"""Offline R11 check: does the trained coordinator beat an LLM-as-coordinator?

``docs/SPEC.md`` §1.3 invariant **R11** — *"Trained coordinator > LLM-as-coordinator"*
(Table 8) — is a replication requirement, yet nothing in ``src/`` or ``scripts/``
verifies it. R11 is the justification for training a tiny derivative-free head at
all: a ~10K-param linear head optimized with sep-CMA-ES should route *better* than
prompting a full LLM to pick the model/role each turn. If it doesn't, the whole
coordinator design is unmotivated.

This reads, per benchmark, the trained coordinator's (TRINITY's) accuracy and the
LLM-as-coordinator baseline's accuracy, and reports the margin ``TRINITY - LLM``,
whether TRINITY wins on that benchmark, and the R11 verdict on the equally-weighted
union (matching the composite score; ROADMAP Phase 2). SPEC §6 notes the paper's
LLM-as-coordinator average is **53.76** (Table 8), not the text's 64.14.

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "CoordinatorMargin",
    "analyze_task",
    "analyze_benchmarks",
    "render",
]

# A win must clear this margin to count, so float noise on a tie is not a "win".
_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class CoordinatorMargin:
    """R11 diagnostics for one benchmark: trained coordinator vs LLM-as-coordinator.

    ``margin`` is ``trinity_accuracy - llm_coordinator_accuracy``. ``trinity_wins``
    is the R11 test for this benchmark: ``margin > tol``.
    """

    benchmark: str
    trinity_accuracy: float
    llm_coordinator_accuracy: float
    margin: float
    trinity_wins: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "trinity_accuracy": self.trinity_accuracy,
            "llm_coordinator_accuracy": self.llm_coordinator_accuracy,
            "margin": self.margin,
            "trinity_wins": self.trinity_wins,
        }


def analyze_task(
    benchmark: str,
    trinity_accuracy: float,
    llm_coordinator_accuracy: float,
    *,
    tol: float = _TOL,
) -> CoordinatorMargin:
    """Compute the R11 margin for one benchmark.

    Args:
        benchmark: Name for the report row.
        trinity_accuracy: The trained coordinator's accuracy, in ``[0, 1]``.
        llm_coordinator_accuracy: The LLM-as-coordinator baseline's accuracy.
        tol: A win must exceed the LLM baseline by more than ``tol``.

    Returns:
        A :class:`CoordinatorMargin`.
    """
    ta = float(trinity_accuracy)
    la = float(llm_coordinator_accuracy)
    margin = ta - la
    return CoordinatorMargin(
        benchmark=benchmark,
        trinity_accuracy=ta,
        llm_coordinator_accuracy=la,
        margin=margin,
        trinity_wins=margin > tol,
    )


def analyze_benchmarks(
    tasks: Mapping[str, Any],
    *,
    tol: float = _TOL,
    require_all: bool = True,
) -> dict[str, Any]:
    """Per-benchmark R11 margins plus the equally-weighted union verdict.

    Args:
        tasks: ``{benchmark: entry}`` where ``entry`` is a mapping carrying
            ``trinity`` / ``trinity_accuracy`` and ``llm`` /
            ``llm_coordinator_accuracy`` / ``llm_coordinator``. A benchmark whose
            values are missing/non-numeric is skipped.
        tol: Win margin (see :func:`analyze_task`).
        require_all: When True (the SPEC reading of R11), R11 holds only if the
            trained coordinator wins on *every* scored benchmark. When False, it
            holds if it wins on the equal-weight union average.

    Returns:
        ``{"per_benchmark": [CoordinatorMargin.to_dict, ...], "n_wins": int,
           "n_scored": int, "union_margin": float, "r11_holds": bool,
           "losses": [benchmark, ...]}``.
    """
    results: list[CoordinatorMargin] = []
    for bench, entry in sorted(tasks.items()):
        if not isinstance(entry, Mapping):
            continue
        ta = entry.get("trinity", entry.get("trinity_accuracy"))
        la = entry.get("llm", entry.get("llm_coordinator_accuracy",
                                        entry.get("llm_coordinator")))
        if not (_is_num(ta) and _is_num(la)):
            continue
        results.append(analyze_task(str(bench), float(ta), float(la), tol=tol))

    n_scored = len(results)
    n_wins = sum(r.trinity_wins for r in results)
    union_margin = (sum(r.margin for r in results) / n_scored) if n_scored else 0.0
    losses = [r.benchmark for r in results if not r.trinity_wins]
    if require_all:
        r11_holds = n_scored > 0 and n_wins == n_scored
    else:
        r11_holds = n_scored > 0 and union_margin > tol
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "n_wins": n_wins,
        "n_scored": n_scored,
        "union_margin": union_margin,
        "r11_holds": r11_holds,
        "losses": losses,
    }


def render(
    tasks: Mapping[str, Any], *, tol: float = _TOL, require_all: bool = True,
) -> str:
    """A compact text report of the per-benchmark R11 check and the union verdict."""
    report = analyze_benchmarks(tasks, tol=tol, require_all=require_all)
    lines = ["| benchmark | trained coord | LLM-as-coord | margin | R11 |",
             "|---|---|---|---|---|"]
    for r in report["per_benchmark"]:
        flag = "ok" if r["trinity_wins"] else "loss"
        lines.append(
            f"| {r['benchmark']} | {r['trinity_accuracy']:.3f} | "
            f"{r['llm_coordinator_accuracy']:.3f} | {r['margin']:+.3f} | {flag} |"
        )
    verdict = "HOLDS" if report["r11_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R11 (trained coordinator > LLM-as-coordinator): {verdict} — won "
        f"{report['n_wins']}/{report['n_scored']} benchmarks "
        f"(union mean margin {report['union_margin']:+.3f})"
    )
    if report["losses"]:
        lines.append(f"did not beat the LLM coordinator on: {', '.join(report['losses'])}")
    return "\n".join(lines)
