"""Offline Fugu Conductor cost & worker-utilization audit.

``scripts/fugu_baseline_eval.py`` writes ``experiments/final/fugu_baseline_<bench>.json``
with a ``cost`` block (``spend_usd``, ``llm_calls``, and a per-model prompt/completion
token + usd breakdown — the ``fugu.cost.CostMeter.report()`` schema), but **nothing
consumes it**. ``docs/fugu/BASELINE_RESULTS.md`` performs this exact analysis **by hand**:

    "The lift is test-time compute, not routing. The Conductor sent ~all work to deepseek
     (deepseek 203k completion tokens vs glm 271, kimi 64) ... Cost of the lift: $1.10 for
     120 tasks, roughly 3.6x a single-shot pass."

This codifies it: per-worker token/cost shares, the **effective number of workers**
(``1/HHI`` on the worker completion-token distribution — ~1.0 means the Conductor collapsed
onto a single worker, i.e. test-time compute rather than routing), the **fanout tax**
(LLM calls per task vs a single-shot pass), **$/correct**, the Conductor's own token
overhead, and a routing-vs-test-time-compute verdict.

Read-only over the persisted cost block; reuses no scoring/fitness math. Pure python — no
torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["WorkerUtilization", "FuguCostSummary", "analyze", "render"]

#: The ``per_model`` key holding the Conductor's OWN tokens (vs the worker models).
_CONDUCTOR_KEY = "<conductor>"


def _num(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0


@dataclass(frozen=True)
class WorkerUtilization:
    """One worker model's token/cost usage under the Conductor."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    usd: float
    completion_share: float   # fraction of all worker completion tokens

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": self.usd,
            "completion_share": self.completion_share,
        }


@dataclass(frozen=True)
class FuguCostSummary:
    """Cost & worker-utilization audit of one Fugu baseline run."""

    benchmark: str
    conductor_model: str | None
    n_tasks: int
    accuracy: float
    parse_rate: float
    spend_usd: float
    llm_calls: int
    workers: list[WorkerUtilization]
    conductor_usd: float
    conductor_token_share: float
    effective_workers: float
    most_used_worker: str | None
    calls_per_task: float
    usd_per_task: float
    usd_per_correct: float | None
    lift_is_test_time_compute: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "conductor_model": self.conductor_model,
            "n_tasks": self.n_tasks,
            "accuracy": self.accuracy,
            "parse_rate": self.parse_rate,
            "spend_usd": self.spend_usd,
            "llm_calls": self.llm_calls,
            "workers": [w.to_dict() for w in self.workers],
            "conductor_usd": self.conductor_usd,
            "conductor_token_share": self.conductor_token_share,
            "effective_workers": self.effective_workers,
            "most_used_worker": self.most_used_worker,
            "calls_per_task": self.calls_per_task,
            "usd_per_task": self.usd_per_task,
            "usd_per_correct": self.usd_per_correct,
            "lift_is_test_time_compute": self.lift_is_test_time_compute,
        }


def analyze(baseline: Mapping[str, Any]) -> FuguCostSummary:
    """Audit a ``fugu_baseline_<bench>.json`` dict's cost block into a summary.

    ``effective_workers = 1 / HHI`` where HHI is the Herfindahl index of the worker
    completion-token shares; ``~1.0`` means the Conductor concentrated on one worker
    (test-time compute), a spread means genuine multi-worker routing. Missing/empty cost
    fields degrade to zeros rather than raising.
    """
    cost = baseline.get("cost") or {}
    per_model: Mapping[str, Any] = cost.get("per_model") or {}
    conductor = per_model.get(_CONDUCTOR_KEY) or {}
    worker_models = {m: v for m, v in per_model.items() if m != _CONDUCTOR_KEY}

    total_wc = sum(_num((v or {}).get("completion_tokens")) for v in worker_models.values())
    workers = [
        WorkerUtilization(
            model=m,
            prompt_tokens=int(_num((v or {}).get("prompt_tokens"))),
            completion_tokens=int(_num((v or {}).get("completion_tokens"))),
            usd=_num((v or {}).get("usd")),
            completion_share=(_num((v or {}).get("completion_tokens")) / total_wc
                              if total_wc > 0 else 0.0),
        )
        for m, v in sorted(worker_models.items(), key=lambda kv: -_num((kv[1] or {}).get("completion_tokens")))
    ]
    hhi = sum(w.completion_share ** 2 for w in workers)
    effective_workers = 1.0 / hhi if hhi > 0 else 0.0

    n_tasks = int(_num(baseline.get("n_tasks")))
    accuracy = _num(baseline.get("accuracy"))
    spend = _num(cost.get("spend_usd"))
    llm_calls = int(_num(cost.get("llm_calls")))
    total_tokens = _num(cost.get("prompt_tokens")) + _num(cost.get("completion_tokens"))
    cond_tokens = _num(conductor.get("prompt_tokens")) + _num(conductor.get("completion_tokens"))
    n_correct = accuracy * n_tasks

    return FuguCostSummary(
        benchmark=str(baseline.get("benchmark", "?")),
        conductor_model=baseline.get("conductor_model"),
        n_tasks=n_tasks,
        accuracy=accuracy,
        parse_rate=_num(baseline.get("parse_rate")),
        spend_usd=spend,
        llm_calls=llm_calls,
        workers=workers,
        conductor_usd=_num(conductor.get("usd")),
        conductor_token_share=(cond_tokens / total_tokens if total_tokens > 0 else 0.0),
        effective_workers=effective_workers,
        most_used_worker=workers[0].model if workers else None,
        calls_per_task=(llm_calls / n_tasks if n_tasks > 0 else 0.0),
        usd_per_task=(spend / n_tasks if n_tasks > 0 else 0.0),
        usd_per_correct=(spend / n_correct if n_correct > 0 else None),
        lift_is_test_time_compute=(0.0 < effective_workers < 1.5),
    )


def render(summary: FuguCostSummary) -> str:
    """Markdown: worker-utilization table + cost-efficiency + the routing-vs-TTC verdict."""
    out = ["# Fugu Conductor cost & worker-utilization audit\n"]
    if summary.n_tasks == 0 and not summary.workers:
        return "".join(out) + "\n_(no cost data)_\n"

    out.append(f"benchmark {summary.benchmark} · conductor {summary.conductor_model} · "
               f"{summary.n_tasks} tasks · accuracy {summary.accuracy:.3f} · "
               f"parse_rate {summary.parse_rate:.3f}\n")
    out.append("| worker | completion tokens | usd | share |")
    out.append("|---|---|---|---|")
    for w in summary.workers:
        out.append(f"| {w.model} | {w.completion_tokens} | ${w.usd:.4f} | {w.completion_share:.3f} |")
    upc = f"${summary.usd_per_correct:.4f}" if summary.usd_per_correct is not None else "—"
    out.append(
        f"\n- spend ${summary.spend_usd:.4f} over {summary.llm_calls} calls "
        f"({summary.calls_per_task:.1f} calls/task fanout) · ${summary.usd_per_task:.4f}/task · "
        f"{upc}/correct"
    )
    out.append(f"- effective workers (1/HHI) = {summary.effective_workers:.2f} "
               f"(most used: {summary.most_used_worker}) · Conductor token overhead "
               f"{summary.conductor_token_share:.1%}")
    if summary.lift_is_test_time_compute:
        out.append(f"\n**Verdict:** the lift is **test-time compute, not routing** — the Conductor "
                   f"concentrated on ~{summary.effective_workers:.1f} effective worker "
                   f"({summary.most_used_worker}).")
    else:
        out.append(f"\n**Verdict:** work is spread across ~{summary.effective_workers:.1f} "
                   f"effective workers — genuine multi-worker routing.")
    return "\n".join(out) + "\n"
