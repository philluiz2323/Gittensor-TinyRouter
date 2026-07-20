"""Offline R3 check: does TRINITY beat the best multi-agent baseline?

``docs/SPEC.md`` §1.3 invariant **R3** — *"TRINITY > best multi-agent baseline
(MoA / MasRouter / RouterDC / Smoothie)"* (§4.2, Fig. 3) — is a replication
requirement, yet nothing in ``src/`` verifies it. R3 is what separates a *learned
coordinator* from the prior art it is measured against: MoA, MasRouter, RouterDC and
Smoothie are the multi-agent ensembling / routing baselines the paper compares to, and
the whole contribution is that an evolved ~10K-param head routes better than any of
them — not merely better than a single model (R1) or random routing (R4).

This reads, per benchmark, TRINITY's accuracy and the multi-agent baselines'
accuracies, and reports whether TRINITY strictly beats the **best** of them (within a
tolerance) on each benchmark and on the equally-weighted union. The baseline set is
open (any ``{name: score}`` map is accepted); the four SPEC-named baselines are
recognized for reporting.

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "SPEC_BASELINES",
    "MultiAgentComparison",
    "analyze_benchmark",
    "analyze_benchmarks",
    "render",
]

#: The multi-agent baselines the SPEC names for R3 (docs/SPEC.md §1.3, §4.2).
SPEC_BASELINES: tuple[str, ...] = ("MoA", "MasRouter", "RouterDC", "Smoothie")

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class MultiAgentComparison:
    """R3 diagnostics for one benchmark: TRINITY vs the best multi-agent baseline.

    ``best_baseline`` / ``best_baseline_score`` name the strongest baseline present
    (the one TRINITY must clear). ``margin`` is ``trinity - best_baseline_score``.
    ``holds`` is the R3 verdict for this benchmark: TRINITY strictly exceeds the best
    baseline (by more than ``tol``). ``comparable`` is False when TRINITY's score is
    missing/non-numeric or no numeric baseline is present.
    """

    benchmark: str
    trinity: float | None
    best_baseline: str | None
    best_baseline_score: float | None
    margin: float | None
    comparable: bool
    holds: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "trinity": self.trinity,
            "best_baseline": self.best_baseline,
            "best_baseline_score": self.best_baseline_score,
            "margin": self.margin,
            "comparable": self.comparable,
            "holds": self.holds,
        }


def analyze_benchmark(
    trinity: Any, baselines: Mapping[str, Any], *, benchmark: str = "?", tol: float = _TOL,
) -> MultiAgentComparison:
    """Compare TRINITY to the best multi-agent baseline on one benchmark.

    Args:
        trinity: TRINITY's accuracy on this benchmark.
        baselines: ``{baseline_name: accuracy}`` (e.g. MoA / MasRouter / RouterDC /
            Smoothie). Non-numeric entries are ignored; the strongest remaining one is
            the bar TRINITY must clear.
        benchmark: Name for the report row.
        tol: TRINITY must exceed the best baseline by more than ``tol`` (a tie is not a
            win).

    Returns:
        A :class:`MultiAgentComparison`. ``comparable=False`` (and ``holds=False``)
        when TRINITY's score is missing or no numeric baseline is present.
    """
    numeric = {str(n): float(s) for n, s in baselines.items() if _is_num(s)}
    if not (_is_num(trinity) and numeric):
        return MultiAgentComparison(
            benchmark=benchmark,
            trinity=float(trinity) if _is_num(trinity) else None,
            best_baseline=None, best_baseline_score=None,
            margin=None, comparable=False, holds=False,
        )
    best_name = max(numeric, key=lambda n: (numeric[n], n))
    best_score = numeric[best_name]
    t = float(trinity)
    margin = t - best_score
    return MultiAgentComparison(
        benchmark=benchmark, trinity=t,
        best_baseline=best_name, best_baseline_score=best_score,
        margin=margin, comparable=True, holds=margin > tol,
    )


def analyze_benchmarks(pairs: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """Per-benchmark R3 comparisons plus the equally-weighted union verdict.

    Args:
        pairs: ``{benchmark: (trinity, {baseline: score})}`` — each value a
            ``(trinity, baselines)`` pair, or a mapping with keys ``trinity`` and
            ``baselines``.
        tol: Win tolerance (see :func:`analyze_benchmark`).

    Returns:
        ``{"per_benchmark": [...], "r3_holds": bool, "union_trinity": float,
           "union_best_baseline": float, "union_margin": float, "violations": [...]}``.
        ``r3_holds`` is True iff at least one benchmark is comparable and every
        comparable benchmark holds. The union means are over comparable rows, using
        each row's *best* baseline as the bar.
    """
    results = [
        analyze_benchmark(*_split(v), benchmark=str(b), tol=tol)
        for b, v in sorted(pairs.items())
    ]
    scored = [r for r in results if r.comparable]
    violations = [r.benchmark for r in results if r.comparable and not r.holds]
    n = len(scored)
    union_trinity = sum(r.trinity for r in scored) / n if n else 0.0            # type: ignore[misc]
    union_baseline = sum(r.best_baseline_score for r in scored) / n if n else 0.0  # type: ignore[misc]
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "r3_holds": bool(scored) and not violations,
        "union_trinity": union_trinity,
        "union_best_baseline": union_baseline,
        "union_margin": union_trinity - union_baseline,
        "violations": violations,
    }


def _split(value: Any) -> tuple[Any, Mapping[str, Any]]:
    """Coerce a pair value to ``(trinity, baselines)``."""
    if isinstance(value, Mapping) and ("trinity" in value or "baselines" in value):
        base = value.get("baselines", {})
        return value.get("trinity"), base if isinstance(base, Mapping) else {}
    try:
        trinity, base = value
    except (TypeError, ValueError):
        return None, {}
    return trinity, base if isinstance(base, Mapping) else {}


def render(pairs: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the per-benchmark R3 check and the union verdict."""
    report = analyze_benchmarks(pairs, tol=tol)
    lines = ["| benchmark | trinity | best baseline | margin | R3 |", "|---|---|---|---|---|"]
    for r in report["per_benchmark"]:
        if not r["comparable"]:
            lines.append(f"| {r['benchmark']} | - | - | - | n/a |")
            continue
        base = f"{r['best_baseline']} {r['best_baseline_score']:.3f}"
        flag = "ok" if r["holds"] else "trinity <= baseline"
        lines.append(
            f"| {r['benchmark']} | {r['trinity']:.3f} | {base} | {r['margin']:+.3f} | {flag} |"
        )
    verdict = "HOLDS" if report["r3_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R3 (TRINITY > best multi-agent baseline): {verdict} "
        f"(union {report['union_trinity']:.3f} vs best-baseline {report['union_best_baseline']:.3f}, "
        f"margin {report['union_margin']:+.3f})"
    )
    if report["violations"]:
        lines.append(f"violations: {', '.join(report['violations'])}")
    return "\n".join(lines)
