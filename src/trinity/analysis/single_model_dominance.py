"""Offline R1/R2 check: does TRINITY beat the single models, on average and per task?

``docs/SPEC.md`` §1.3 invariants **R1** — *"TRINITY avg > best single model avg
(budget-matched 5×)"* — and **R2** — *"TRINITY > every single model on every task"*
(Tables 1, 2). ``eval.py`` computes a single R1 avg-boolean live during evaluation, but
there is no offline, reusable module that reads cached per-task/per-model accuracies and
checks **both**: the averaged R1 claim *and* the stricter per-task/per-model R2 claim
(TRINITY dominates *every* model on *every* task, not just on average).

This reads ``{task: (trinity, {model: score})}`` and reports:

* **R2** — per task, whether TRINITY strictly beats the best single model on that task
  (equivalently, beats every model), and which tasks violate it;
* **R1** — TRINITY's mean vs the best single model's mean (the model with the highest
  average across tasks), the classic "beats the best single model on average" claim.

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "TaskDominance",
    "analyze_task",
    "analyze_tasks",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class TaskDominance:
    """R2 diagnostics for one task: TRINITY vs the single models on that task.

    ``best_model`` / ``best_model_score`` name the strongest single model on this task
    (the one TRINITY must clear to beat *every* model). ``margin`` is ``trinity -
    best_model_score``. ``dominates`` is the per-task R2 verdict: TRINITY strictly
    exceeds the best single model (so it beats them all). ``comparable`` is False when
    TRINITY is missing or no numeric model score is present.
    """

    task: str
    trinity: float | None
    best_model: str | None
    best_model_score: float | None
    margin: float | None
    comparable: bool
    dominates: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "task": self.task,
            "trinity": self.trinity,
            "best_model": self.best_model,
            "best_model_score": self.best_model_score,
            "margin": self.margin,
            "comparable": self.comparable,
            "dominates": self.dominates,
        }


def analyze_task(
    trinity: Any, singles: Mapping[str, Any], *, task: str = "?", tol: float = _TOL,
) -> TaskDominance:
    """R2 for one task: does TRINITY beat every single model on it?

    Args:
        trinity: TRINITY's accuracy on this task.
        singles: ``{model: accuracy}`` for the single models. Non-numeric entries are
            ignored; the strongest remaining one is the bar (beating it beats them all).
        task: Name for the report row.
        tol: TRINITY must exceed the best model by more than ``tol`` (a tie is not a win).

    Returns:
        A :class:`TaskDominance`. ``comparable=False`` (and ``dominates=False``) when
        TRINITY is missing or no numeric model score is present.
    """
    numeric = {str(m): float(s) for m, s in singles.items() if _is_num(s)}
    if not (_is_num(trinity) and numeric):
        return TaskDominance(
            task=task, trinity=float(trinity) if _is_num(trinity) else None,
            best_model=None, best_model_score=None, margin=None,
            comparable=False, dominates=False,
        )
    best = max(numeric, key=lambda m: (numeric[m], m))
    best_score = numeric[best]
    margin = float(trinity) - best_score
    return TaskDominance(
        task=task, trinity=float(trinity), best_model=best, best_model_score=best_score,
        margin=margin, comparable=True, dominates=margin > tol,
    )


def analyze_tasks(tasks: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """R2 per-task dominance plus the averaged R1 claim and the combined verdict.

    Args:
        tasks: ``{task: (trinity, {model: score})}`` — each value a ``(trinity,
            singles)`` pair, or a mapping with keys ``trinity`` / ``singles``.
        tol: Win tolerance (see :func:`analyze_task`).

    Returns:
        ``{"per_task": [...], "r2_holds": bool, "r2_violations": [task, ...],
           "trinity_avg": float, "best_single_model": str|None, "best_single_avg": float,
           "r1_holds": bool, "r1r2_holds": bool}``. **R2** holds iff every comparable
        task dominates. **R1** compares TRINITY's mean to the best single model's mean
        (the model with the highest average over the comparable tasks).
    """
    results = [analyze_task(*_split(v), task=str(t), tol=tol) for t, v in sorted(tasks.items())]
    scored = [r for r in results if r.comparable]
    r2_violations = [r.task for r in scored if not r.dominates]

    # R1: TRINITY mean vs the best single model's mean over the comparable tasks.
    trinity_avg = sum(r.trinity for r in scored) / len(scored) if scored else 0.0  # type: ignore[misc]
    per_model_sum: dict[str, float] = {}
    per_model_n: dict[str, int] = {}
    for t, v in sorted(tasks.items()):
        r = analyze_task(*_split(v), task=str(t), tol=tol)
        if not r.comparable:
            continue
        _, singles = _split(v)
        for model, score in singles.items():
            if _is_num(score):
                per_model_sum[str(model)] = per_model_sum.get(str(model), 0.0) + float(score)
                per_model_n[str(model)] = per_model_n.get(str(model), 0) + 1
    per_model_avg = {m: per_model_sum[m] / per_model_n[m] for m in per_model_sum}
    best_single_model = max(per_model_avg, key=lambda m: (per_model_avg[m], m)) if per_model_avg else None
    best_single_avg = per_model_avg[best_single_model] if best_single_model is not None else 0.0
    r1_holds = bool(scored) and (trinity_avg - best_single_avg) > tol
    r2_holds = bool(scored) and not r2_violations
    return {
        "per_task": [r.to_dict() for r in results],
        "r2_holds": r2_holds,
        "r2_violations": r2_violations,
        "trinity_avg": trinity_avg,
        "best_single_model": best_single_model,
        "best_single_avg": best_single_avg,
        "r1_holds": r1_holds,
        "r1r2_holds": r1_holds and r2_holds,
    }


def _split(value: Any) -> tuple[Any, Mapping[str, Any]]:
    """Coerce a pair value to ``(trinity, singles)``."""
    if isinstance(value, Mapping) and ("trinity" in value or "singles" in value):
        s = value.get("singles", {})
        return value.get("trinity"), s if isinstance(s, Mapping) else {}
    try:
        trinity, singles = value
    except (TypeError, ValueError):
        return None, {}
    return trinity, singles if isinstance(singles, Mapping) else {}


def render(tasks: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the R2 per-task dominance and the R1 average claim."""
    report = analyze_tasks(tasks, tol=tol)
    lines = ["| task | trinity | best model | margin | R2 |", "|---|---|---|---|---|"]
    for r in report["per_task"]:
        if not r["comparable"]:
            lines.append(f"| {r['task']} | - | - | - | n/a |")
            continue
        base = f"{r['best_model']} {r['best_model_score']:.3f}"
        flag = "ok" if r["dominates"] else "trinity <= a model"
        lines.append(
            f"| {r['task']} | {r['trinity']:.3f} | {base} | {r['margin']:+.3f} | {flag} |"
        )
    r2 = "HOLDS" if report["r2_holds"] else "VIOLATED"
    r1 = "HOLDS" if report["r1_holds"] else "VIOLATED"
    lines.append("")
    lines.append(f"R2 (TRINITY > every single model on every task): {r2}")
    if report["r2_violations"]:
        lines.append(f"  violations: {', '.join(report['r2_violations'])}")
    best = report["best_single_model"] or "-"
    lines.append(
        f"R1 (TRINITY avg > best single model avg): {r1} "
        f"(trinity {report['trinity_avg']:.3f} vs {best} {report['best_single_avg']:.3f})"
    )
    return "\n".join(lines)
