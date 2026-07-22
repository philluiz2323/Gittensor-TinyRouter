"""Offline R4 check: does the trained coordinator beat random routing?

``docs/SPEC.md`` §1.3 invariant **R4** — *"TRINITY > random routing"* (RLPR: 0.41 vs
0.32) — is the most basic sanity floor the coordinator must clear: a trained head that
could not beat picking a random model+role each turn would not be routing at all.
``eval.py`` computes R4 live during evaluation (against a multi-seed random-routing
baseline), but there is no offline, reusable module that reads cached per-benchmark
accuracies and checks it — the same role the merged R1/R2/R3/R5/R7/R8/R9/R10/R11/R12/R13
verifier modules play for their invariants.

This reads, per benchmark, TRINITY's accuracy and the random-routing baseline accuracy,
and reports whether TRINITY strictly beats random routing on each benchmark and on the
equally-weighted union (matching the composite score, ROADMAP Phase 2).

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "RandomRoutingComparison",
    "analyze_benchmark",
    "analyze_benchmarks",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class RandomRoutingComparison:
    """R4 diagnostics for one benchmark: TRINITY vs the random-routing baseline.

    ``margin`` is ``trinity - random_routing`` (positive means TRINITY wins). ``holds``
    is the R4 verdict for this benchmark: TRINITY scores strictly above random routing
    (by more than ``tol``). ``comparable`` is False when either accuracy is missing or
    non-numeric.
    """

    benchmark: str
    trinity: float | None
    random_routing: float | None
    margin: float | None
    comparable: bool
    holds: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "trinity": self.trinity,
            "random_routing": self.random_routing,
            "margin": self.margin,
            "comparable": self.comparable,
            "holds": self.holds,
        }


def analyze_benchmark(
    trinity: Any, random_routing: Any, *, benchmark: str = "?", tol: float = _TOL,
) -> RandomRoutingComparison:
    """Compare one benchmark's TRINITY accuracy to the random-routing baseline.

    Args:
        trinity: Trained-coordinator (TRINITY) accuracy on this benchmark.
        random_routing: Random-routing baseline accuracy (typically a multi-seed mean).
        benchmark: Name for the report row.
        tol: TRINITY must exceed random routing by more than ``tol`` to count as a win
            (a tie is not a pass).

    Returns:
        A :class:`RandomRoutingComparison`. When either accuracy is missing or
        non-numeric the row is ``comparable=False`` and ``holds=False``.
    """
    if not (_is_num(trinity) and _is_num(random_routing)):
        return RandomRoutingComparison(
            benchmark=benchmark,
            trinity=float(trinity) if _is_num(trinity) else None,
            random_routing=float(random_routing) if _is_num(random_routing) else None,
            margin=None, comparable=False, holds=False,
        )
    t, r = float(trinity), float(random_routing)
    margin = t - r
    return RandomRoutingComparison(
        benchmark=benchmark, trinity=t, random_routing=r,
        margin=margin, comparable=True, holds=margin > tol,
    )


def analyze_benchmarks(pairs: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """Per-benchmark R4 comparisons plus the equally-weighted union verdict.

    Args:
        pairs: ``{benchmark: (trinity, random_routing)}`` — each value a 2-tuple/list,
            or a mapping with keys ``trinity`` / ``random_routing`` (alias ``random``).
        tol: Win tolerance (see :func:`analyze_benchmark`).

    Returns:
        ``{"per_benchmark": [...], "r4_holds": bool, "union_trinity": float,
           "union_random_routing": float, "union_margin": float, "violations": [...]}``.
        ``r4_holds`` is True iff at least one benchmark is comparable and every
        comparable benchmark holds. The union accuracies are equal-weight means over
        comparable rows.
    """
    results = [
        analyze_benchmark(*_split(v), benchmark=str(b), tol=tol)
        for b, v in sorted(pairs.items())
    ]
    scored = [r for r in results if r.comparable]
    violations = [r.benchmark for r in results if r.comparable and not r.holds]
    n = len(scored)
    union_trinity = sum(r.trinity for r in scored) / n if n else 0.0        # type: ignore[misc]
    union_random = sum(r.random_routing for r in scored) / n if n else 0.0  # type: ignore[misc]
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "r4_holds": bool(scored) and not violations,
        "union_trinity": union_trinity,
        "union_random_routing": union_random,
        "union_margin": union_trinity - union_random,
        "violations": violations,
    }


def _split(value: Any) -> tuple[Any, Any]:
    """Coerce a pair value to ``(trinity, random_routing)``."""
    if isinstance(value, Mapping) and ("trinity" in value or "random_routing" in value or "random" in value):
        return value.get("trinity"), value.get("random_routing", value.get("random"))
    try:
        trinity, random_routing = value
    except (TypeError, ValueError):
        return None, None
    return trinity, random_routing


def render(pairs: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the per-benchmark R4 check and the union verdict."""
    report = analyze_benchmarks(pairs, tol=tol)
    lines = ["| benchmark | trinity | random routing | margin | R4 |", "|---|---|---|---|---|"]
    for r in report["per_benchmark"]:
        if not r["comparable"]:
            lines.append(f"| {r['benchmark']} | - | - | - | n/a |")
            continue
        flag = "ok" if r["holds"] else "trinity <= random"
        lines.append(
            f"| {r['benchmark']} | {r['trinity']:.3f} | {r['random_routing']:.3f} | "
            f"{r['margin']:+.3f} | {flag} |"
        )
    verdict = "HOLDS" if report["r4_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R4 (TRINITY > random routing): {verdict} "
        f"(union {report['union_trinity']:.3f} vs {report['union_random_routing']:.3f}, "
        f"margin {report['union_margin']:+.3f})"
    )
    if report["violations"]:
        lines.append(f"violations: {', '.join(report['violations'])}")
    return "\n".join(lines)
