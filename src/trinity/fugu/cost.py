"""API-cost accounting for the Fugu pipeline.

Every Conductor workflow fans out to several worker calls plus the Conductor's
own generation, so cost is the headline operational risk of training a Conductor
by GRPO (each rollout is one proposal plus N worker calls, multiplied by the
group size and the iteration count). This module makes that cost first-class:

* :func:`run_cost` prices a single :class:`~trinity.fugu.workflow.WorkflowRun`
  exactly from its per-model token totals.
* :class:`CostMeter` tracks running spend across many runs and can abort a loop
  when a spend cap is hit (the same guard the oracle-ceiling collector uses).
* :func:`estimate_grpo_cost` / :func:`estimate_eval_cost` PROJECT the spend of a
  run before it starts, so a paid GRPO job is never launched blind.

Prices match ``scripts/oracle_ceiling.py`` (OpenRouter $/1M tokens, in/out). The
runtime ground truth remains the shared cost ledger (set ``TRINITY_COST_LEDGER``
and run ``scripts/cost_report.py``); these functions agree with it and add a
pre-run estimate the ledger cannot give.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trinity.fugu.workflow import CONDUCTOR_KEY

__all__ = [
    "PRICES",
    "price_table",
    "run_cost",
    "CostMeter",
    "estimate_grpo_cost",
    "estimate_eval_cost",
]

# OpenRouter prices, $/1M tokens (prompt, completion). Keep in sync with
# scripts/oracle_ceiling.py::_DEFAULT_PRICES.
PRICES: dict[str, tuple[float, float]] = {
    "qwen3.5-35b-a3b": (0.14, 1.00),
    "minimax-m3": (0.30, 1.20),
    "deepseek-v4-flash": (0.09, 0.18),
}


def _conductor_price(
    lookup: dict[str, tuple[float, float]],
    conductor_model: str | None,
    conductor_local: bool,
) -> tuple[float, float]:
    """Return the ``(price_in, price_out)`` charged for the Conductor's generation.

    A locally-served Conductor costs no API money -- only GPU time, which the caller
    accounts for separately -- so it prices at zero. A hosted Conductor prices at its
    own model's rate.

    Keeping this rule in one place stops :func:`price_table` and the estimators from
    disagreeing about when the Conductor is billed.

    Args:
        lookup: Table used to resolve ``conductor_model``'s per-token price.
        conductor_model: The Conductor's model name, or ``None`` when unspecified.
        conductor_local: Whether the Conductor runs on our own hardware.

    Returns:
        The ``(price_in, price_out)`` pair, in $ per 1M tokens.
    """
    if conductor_local or conductor_model is None:
        return (0.0, 0.0)
    return lookup.get(conductor_model, (0.0, 0.0))


def price_table(
    conductor_model: str | None = None,
    *,
    conductor_local: bool = True,
    extra: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Build a name -> (price_in, price_out) table including the Conductor.

    The Conductor is normally a model WE serve on our own H200 (a GRPO-trained
    HF checkpoint), so its per-token API cost is 0 and only GPU time applies; set
    ``conductor_local=True`` (default). When the Conductor is instead an OpenRouter
    model (the prompted baseline), pass ``conductor_local=False`` and its name so
    its generation is priced.
    """
    table = dict(PRICES)
    if extra:
        table.update(extra)
    table[CONDUCTOR_KEY] = _conductor_price(table, conductor_model, conductor_local)
    return table


def run_cost(
    model_tokens: dict[str, tuple[int, int]],
    prices: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, dict[str, float]]:
    """Exact USD cost of one run's per-model token totals.

    Returns ``(total_usd, per_model_usd)``. Unknown models price at 0 (and are
    surfaced in the breakdown so a missing price is visible, not silent).
    """
    table = prices if prices is not None else price_table()
    breakdown: dict[str, float] = {}
    total = 0.0
    for name, (pt, ct) in model_tokens.items():
        pin, pout = table.get(name, (0.0, 0.0))
        c = pt / 1e6 * pin + ct / 1e6 * pout
        breakdown[name] = c
        total += c
    return total, breakdown


@dataclass
class CostMeter:
    """Running spend tracker across many workflow runs, with an optional cap."""

    prices: dict[str, tuple[float, float]] = field(default_factory=price_table)
    cap_usd: float = 0.0                 # 0 disables the cap
    spend: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    runs: int = 0
    aborted: bool = False
    # name -> [prompt_tokens, completion_tokens, usd]
    per_model: dict[str, list] = field(default_factory=dict)

    def add_run(self, run) -> float:
        """Add one :class:`WorkflowRun`. Returns the run's cost; sets ``aborted``
        if the running total crosses ``cap_usd``."""
        cost, breakdown = run_cost(run.model_tokens, self.prices)
        self.spend += cost
        self.calls += run.n_llm_calls
        self.prompt_tokens += run.prompt_tokens
        self.completion_tokens += run.completion_tokens
        self.runs += 1
        for name, (pt, ct) in run.model_tokens.items():
            row = self.per_model.setdefault(name, [0, 0, 0.0])
            row[0] += pt
            row[1] += ct
            row[2] += breakdown.get(name, 0.0)
        if self.cap_usd > 0 and self.spend > self.cap_usd:
            self.aborted = True
        return cost

    def report(self) -> dict:
        return {
            "spend_usd": round(self.spend, 4),
            "llm_calls": self.calls,
            "runs": self.runs,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cap_usd": self.cap_usd,
            "aborted": self.aborted,
            "per_model": {
                n: {"prompt_tokens": r[0], "completion_tokens": r[1], "usd": round(r[2], 4)}
                for n, r in sorted(self.per_model.items())
            },
        }


def _blended_worker_price(prices: dict[str, tuple[float, float]],
                          worker_names: list[str]) -> tuple[float, float]:
    """Mean (price_in, price_out) across the worker pool, for projections."""
    rows = [prices.get(n, (0.0, 0.0)) for n in worker_names] or [(0.0, 0.0)]
    pin = sum(r[0] for r in rows) / len(rows)
    pout = sum(r[1] for r in rows) / len(rows)
    return pin, pout


def estimate_grpo_cost(
    *,
    worker_names: list[str],
    group_size: int = 64,
    iterations: int = 200,
    questions_per_iter: int = 4,
    avg_steps: float = 2.5,
    avg_worker_prompt_tokens: int = 1200,
    avg_worker_completion_tokens: int = 800,
    avg_conductor_prompt_tokens: int = 700,
    avg_conductor_completion_tokens: int = 250,
    prices: dict[str, tuple[float, float]] | None = None,
    conductor_local: bool = True,
    conductor_model: str | None = None,
) -> dict:
    """Project the API spend of a GRPO Conductor training run before launching it.

    A rollout is one Conductor proposal plus ``avg_steps`` worker calls. The run
    does ``group_size * questions_per_iter`` rollouts per iteration over
    ``iterations`` iterations. With ``conductor_local=True`` the Conductor runs on
    our own GPU, so only the worker calls cost API money; the GPU hours are noted
    separately by the caller.

    ``prices``, when given, overrides the worker prices. It does not disable
    Conductor pricing: unless it carries an explicit ``CONDUCTOR_KEY`` entry, the
    Conductor is still priced from ``conductor_local`` / ``conductor_model``, so a
    projection can never under-state spend by omitting a component it was told to
    include.
    """
    if prices is None:
        table = price_table(conductor_model, conductor_local=conductor_local)
    else:
        # An explicit `prices` table stays authoritative for the WORKER models, but
        # it must not silently disable Conductor pricing: a worker table carries no
        # CONDUCTOR_KEY, so `table.get(CONDUCTOR_KEY, (0, 0))` would bill the
        # Conductor $0 even under `conductor_local=False`. Derive the entry from the
        # same rule `price_table` uses, resolving the model's rate against PRICES
        # overlaid with the caller's table. A caller that supplies CONDUCTOR_KEY
        # explicitly keeps it.
        table = dict(prices)
        if CONDUCTOR_KEY not in table:
            table[CONDUCTOR_KEY] = _conductor_price(
                {**PRICES, **table}, conductor_model, conductor_local
            )
    rollouts = group_size * questions_per_iter * iterations
    worker_calls = int(round(rollouts * avg_steps))

    win, wout = _blended_worker_price(table, worker_names)
    worker_usd = worker_calls * (
        avg_worker_prompt_tokens / 1e6 * win + avg_worker_completion_tokens / 1e6 * wout
    )
    cin, cout = table.get(CONDUCTOR_KEY, (0.0, 0.0))
    conductor_usd = rollouts * (
        avg_conductor_prompt_tokens / 1e6 * cin + avg_conductor_completion_tokens / 1e6 * cout
    )
    return {
        "rollouts": rollouts,
        "worker_calls": worker_calls,
        "conductor_calls": rollouts,
        "worker_usd": round(worker_usd, 2),
        "conductor_api_usd": round(conductor_usd, 2),
        "total_usd": round(worker_usd + conductor_usd, 2),
        "assumptions": {
            "group_size": group_size, "iterations": iterations,
            "questions_per_iter": questions_per_iter, "avg_steps": avg_steps,
            "conductor_local": conductor_local,
            "avg_worker_tokens": [avg_worker_prompt_tokens, avg_worker_completion_tokens],
            "avg_conductor_tokens": [avg_conductor_prompt_tokens, avg_conductor_completion_tokens],
        },
    }


def estimate_eval_cost(
    *,
    worker_names: list[str],
    n_tasks: int,
    reps: int = 1,
    avg_steps: float = 2.5,
    avg_worker_prompt_tokens: int = 1200,
    avg_worker_completion_tokens: int = 800,
    avg_conductor_prompt_tokens: int = 700,
    avg_conductor_completion_tokens: int = 250,
    prices: dict[str, tuple[float, float]] | None = None,
    conductor_local: bool = True,
    conductor_model: str | None = None,
) -> dict:
    """Project the API spend of evaluating the Conductor on ``n_tasks`` x ``reps``."""
    return estimate_grpo_cost(
        worker_names=worker_names,
        group_size=reps,
        iterations=1,
        questions_per_iter=n_tasks,
        avg_steps=avg_steps,
        avg_worker_prompt_tokens=avg_worker_prompt_tokens,
        avg_worker_completion_tokens=avg_worker_completion_tokens,
        avg_conductor_prompt_tokens=avg_conductor_prompt_tokens,
        avg_conductor_completion_tokens=avg_conductor_completion_tokens,
        prices=prices,
        conductor_local=conductor_local,
        conductor_model=conductor_model,
    )
