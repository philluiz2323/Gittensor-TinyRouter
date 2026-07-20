"""Offline R11 check: does the trained coordinator beat an LLM-as-coordinator?

``docs/SPEC.md`` §1.3 invariant **R11** — *"Trained coordinator > LLM-as-coordinator"*
(Table 8) — is a replication requirement, yet nothing in ``src/`` or ``scripts/``
verifies it. R11 is the whole thesis of TRINITY: a tiny (<20K-param) SLM+linear head
trained with sep-CMA-ES should route better than simply *prompting an LLM to act as
the coordinator*. If a frozen LLM picking the models/roles matched or beat the trained
head, the evolved coordinator would not be earning its keep.

This reads, per benchmark, the trained-coordinator accuracy and the
LLM-as-coordinator baseline accuracy, and reports whether the trained coordinator wins
(strictly, within a tolerance) on each benchmark and on the equally-weighted
3-benchmark union (matching the composite score, ROADMAP Phase 2). The SPEC pins the
LLM-as-coordinator average at **53.76** (Table 8 — not the text's 64.14); this module
targets the *invariant* (trained > baseline), not the absolute number.

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "CoordinatorComparison",
    "analyze_pair",
    "analyze_benchmarks",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class CoordinatorComparison:
    """R11 diagnostics for one benchmark: trained coordinator vs LLM-as-coordinator.

    ``margin`` is ``trained - llm_as_coordinator`` (positive means the trained head
    wins). ``holds`` is the R11 verdict for this benchmark: the trained coordinator
    scores strictly above the LLM-as-coordinator baseline (by more than ``tol``).
    ``comparable`` is False when either accuracy is missing/non-numeric, in which case
    the row cannot be judged and does not count toward the union.
    """

    benchmark: str
    trained: float | None
    llm_as_coordinator: float | None
    margin: float | None
    comparable: bool
    holds: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "trained": self.trained,
            "llm_as_coordinator": self.llm_as_coordinator,
            "margin": self.margin,
            "comparable": self.comparable,
            "holds": self.holds,
        }


def analyze_pair(
    trained: Any, llm_as_coordinator: Any, *, benchmark: str = "?", tol: float = _TOL,
) -> CoordinatorComparison:
    """Compare one benchmark's trained-coordinator accuracy to the LLM-as-coordinator.

    Args:
        trained: Trained-coordinator (TRINITY) accuracy on this benchmark.
        llm_as_coordinator: LLM-as-coordinator baseline accuracy on this benchmark.
        benchmark: Name for the report row.
        tol: The trained coordinator must exceed the baseline by more than ``tol`` to
            count as a win (so float noise on a tie is not a spurious R11 pass).

    Returns:
        A :class:`CoordinatorComparison`. When either accuracy is missing or
        non-numeric the row is ``comparable=False`` and ``holds=False``.
    """
    if not (_is_num(trained) and _is_num(llm_as_coordinator)):
        return CoordinatorComparison(
            benchmark=benchmark,
            trained=float(trained) if _is_num(trained) else None,
            llm_as_coordinator=float(llm_as_coordinator) if _is_num(llm_as_coordinator) else None,
            margin=None, comparable=False, holds=False,
        )
    t, b = float(trained), float(llm_as_coordinator)
    margin = t - b
    return CoordinatorComparison(
        benchmark=benchmark, trained=t, llm_as_coordinator=b,
        margin=margin, comparable=True, holds=margin > tol,
    )


def analyze_benchmarks(pairs: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """Per-benchmark R11 comparisons plus the equally-weighted union verdict.

    Args:
        pairs: ``{benchmark: (trained, llm_as_coordinator)}`` — each value a 2-tuple/
            list, or a mapping with keys ``trained`` / ``llm_as_coordinator`` (aliases
            ``trinity`` / ``llm``).
        tol: Win tolerance (see :func:`analyze_pair`).

    Returns:
        ``{"per_benchmark": [CoordinatorComparison.to_dict, ...], "r11_holds": bool,
           "union_trained": float, "union_llm_as_coordinator": float,
           "union_margin": float, "violations": [benchmark, ...]}``. ``r11_holds`` is
        True iff at least one benchmark is comparable and every comparable benchmark
        holds. The union accuracies are equal-weight means over comparable rows.
    """
    results = [
        analyze_pair(*_split(v), benchmark=str(b), tol=tol)
        for b, v in sorted(pairs.items())
    ]
    scored = [r for r in results if r.comparable]
    violations = [r.benchmark for r in results if r.comparable and not r.holds]
    n = len(scored)
    union_trained = sum(r.trained for r in scored) / n if n else 0.0       # type: ignore[misc]
    union_llm = sum(r.llm_as_coordinator for r in scored) / n if n else 0.0  # type: ignore[misc]
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "r11_holds": bool(scored) and all(r.holds for r in scored),
        "union_trained": union_trained,
        "union_llm_as_coordinator": union_llm,
        "union_margin": union_trained - union_llm,
        "violations": violations,
    }


def _split(value: Any) -> tuple[Any, Any]:
    """Coerce a pair value to ``(trained, llm_as_coordinator)``."""
    if isinstance(value, Mapping):
        trained = value.get("trained", value.get("trinity"))
        llm = value.get("llm_as_coordinator", value.get("llm"))
        return trained, llm
    try:
        trained, llm = value
    except (TypeError, ValueError):
        return None, None
    return trained, llm


def render(pairs: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the per-benchmark R11 check and the union verdict."""
    report = analyze_benchmarks(pairs, tol=tol)
    lines = ["| benchmark | trained | llm-as-coord | margin | R11 |",
             "|---|---|---|---|---|"]
    for r in report["per_benchmark"]:
        if not r["comparable"]:
            lines.append(f"| {r['benchmark']} | - | - | - | n/a |")
            continue
        flag = "ok" if r["holds"] else "trained <= baseline"
        lines.append(
            f"| {r['benchmark']} | {r['trained']:.3f} | {r['llm_as_coordinator']:.3f} | "
            f"{r['margin']:+.3f} | {flag} |"
        )
    verdict = "HOLDS" if report["r11_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R11 (trained coordinator > LLM-as-coordinator): {verdict} "
        f"(union {report['union_trained']:.3f} vs {report['union_llm_as_coordinator']:.3f}, "
        f"margin {report['union_margin']:+.3f})"
    )
    if report["violations"]:
        lines.append(f"violations: {', '.join(report['violations'])}")
    return "\n".join(lines)
