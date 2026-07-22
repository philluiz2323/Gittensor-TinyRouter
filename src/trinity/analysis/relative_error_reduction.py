"""Offline R13 check: the relative-error-reduction of TRINITY over the best single agent.

``docs/SPEC.md`` §1.3 invariant **R13** — *"Mean relative-error-reduction ≈ 21.9% vs
2nd-best (ballpark, pool-dependent)"* — with the metric pinned in §6 (L356):

    RER = (Z − S*) / (1 − S*)

where ``Z`` is the coordinated (TRINITY) score and ``S*`` is the best single-agent
score on the subset. RER is the fraction of the *remaining* error above the strongest
single model that the coordinator closes: 0.0 means TRINITY only matches the best
single model, 1.0 means it reaches a perfect score, negative means it does worse.

The absolute 21.9% is explicitly *ballpark, pool-dependent*, so this module targets the
**invariant** — TRINITY closes positive error above the best single agent on every
task — and reports the per-task and mean RER so a caller can compare to the paper's
ballpark. Nothing in ``src/`` computed RER before (``eval.py`` checks R1/R2/R4 live but
not this metric).

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ErrorReduction",
    "relative_error_reduction",
    "analyze_task",
    "analyze_tasks",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def relative_error_reduction(trinity: float, best_single: float) -> float:
    """``RER = (trinity - best_single) / (1 - best_single)`` (docs/SPEC.md §6, L356).

    The caller must ensure ``best_single < 1`` (a perfect single model leaves no error
    to reduce); :func:`analyze_task` guards that case.
    """
    return (float(trinity) - float(best_single)) / (1.0 - float(best_single))


@dataclass(frozen=True)
class ErrorReduction:
    """R13 diagnostics for one task's TRINITY vs best-single-agent scores.

    ``best_single`` is the strongest single-agent score (the bar). ``rer`` is the
    relative error reduction above it (``None`` when it cannot be computed — missing
    scores, or a perfect single model that leaves no error to close). ``holds`` is the
    R13 verdict for this task: a strictly positive RER (TRINITY reduces error).
    """

    task: str
    trinity: float | None
    best_single: float | None
    rer: float | None
    comparable: bool
    holds: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "task": self.task,
            "trinity": self.trinity,
            "best_single": self.best_single,
            "rer": self.rer,
            "comparable": self.comparable,
            "holds": self.holds,
        }


def _best_single(single: Any) -> float | None:
    """Best single-agent score from a scalar or a ``{model: score}`` map."""
    if _is_num(single):
        return float(single)
    if isinstance(single, Mapping):
        vals = [float(v) for v in single.values() if _is_num(v)]
        return max(vals) if vals else None
    return None


def analyze_task(
    trinity: Any, best_single: Any, *, task: str = "?", tol: float = _TOL,
) -> ErrorReduction:
    """Compute the R13 relative-error-reduction for one task.

    Args:
        trinity: TRINITY (coordinated) accuracy on this task.
        best_single: The best single-agent accuracy — a scalar, or a
            ``{model: accuracy}`` map whose max is taken.
        task: Name for the report row.
        tol: RER must exceed ``tol`` to count as a positive reduction.

    Returns:
        An :class:`ErrorReduction`. ``comparable=False`` when TRINITY is missing, no
        numeric single-agent score is present, or the best single already scores ``>=
        1`` (no error to reduce).
    """
    s = _best_single(best_single)
    if not (_is_num(trinity) and s is not None) or s >= 1.0 - tol:
        return ErrorReduction(
            task=task,
            trinity=float(trinity) if _is_num(trinity) else None,
            best_single=s, rer=None, comparable=False, holds=False,
        )
    rer = relative_error_reduction(float(trinity), s)
    return ErrorReduction(
        task=task, trinity=float(trinity), best_single=s,
        rer=rer, comparable=True, holds=rer > tol,
    )


def analyze_tasks(tasks: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """Per-task R13 reductions plus the mean and the across-tasks verdict.

    Args:
        tasks: ``{task: (trinity, best_single)}`` — each value a ``(trinity,
            best_single)`` pair, or a mapping with keys ``trinity`` / ``best_single``
            (alias ``singles`` for a ``{model: score}`` map).
        tol: Positive-reduction tolerance (see :func:`analyze_task`).

    Returns:
        ``{"per_task": [...], "r13_holds": bool, "mean_rer": float,
           "n_tasks_scored": int, "violations": [task, ...]}``. ``r13_holds`` is True
        iff at least one task is comparable and **every** comparable task has a positive
        RER. ``mean_rer`` is the equal-weight mean RER over comparable tasks.
    """
    results = [analyze_task(*_split(v), task=str(t), tol=tol) for t, v in sorted(tasks.items())]
    scored = [r for r in results if r.comparable]
    violations = [r.task for r in scored if not r.holds]
    mean_rer = sum(r.rer for r in scored) / len(scored) if scored else 0.0  # type: ignore[misc]
    return {
        "per_task": [r.to_dict() for r in results],
        "r13_holds": bool(scored) and not violations,
        "mean_rer": mean_rer,
        "n_tasks_scored": len(scored),
        "violations": violations,
    }


def _split(value: Any) -> tuple[Any, Any]:
    """Coerce a pair value to ``(trinity, best_single)``."""
    if isinstance(value, Mapping) and ("trinity" in value or "best_single" in value or "singles" in value):
        return value.get("trinity"), value.get("best_single", value.get("singles"))
    try:
        trinity, best_single = value
    except (TypeError, ValueError):
        return None, None
    return trinity, best_single


def render(tasks: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the per-task R13 reduction and the mean/verdict."""
    report = analyze_tasks(tasks, tol=tol)
    lines = ["| task | trinity | best single | RER | R13 |", "|---|---|---|---|---|"]
    for r in report["per_task"]:
        if not r["comparable"]:
            lines.append(f"| {r['task']} | - | - | - | n/a |")
            continue
        flag = "ok" if r["holds"] else "no reduction"
        lines.append(
            f"| {r['task']} | {r['trinity']:.3f} | {r['best_single']:.3f} | "
            f"{r['rer'] * 100:+.1f}% | {flag} |"
        )
    verdict = "HOLDS" if report["r13_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R13 (relative-error-reduction > 0 vs best single): {verdict} "
        f"(mean RER {report['mean_rer'] * 100:+.1f}%, ballpark ~21.9%)"
    )
    if report["violations"]:
        lines.append(f"violations: {', '.join(report['violations'])}")
    return "\n".join(lines)
