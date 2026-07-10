"""Offline efficiency and composite-score analysis for a Conductor.

Why this exists
---------------
The competition score has four weighted parts (CONTRIBUTING.md / SUBMITTING.md):

    score = 0.70 * hidden_acc      # cached single-turn accuracy
          + 0.15 * live_acc        # live multi-turn accuracy
          + 0.10 * efficiency      # fewer turns per correct answer
          + 0.05 * novelty         # different routing choices from other miners

The **efficiency** term (10% of the score) rewards reaching a correct answer in
fewer turns:

    efficiency = max(0, (max_turns - avg_turns) / (max_turns - 1)) * live_acc

Today this formula lives only inside ``scripts/pr_eval.py::_compute_score`` (the
maintainer's hidden scorer). A contributor iterating on a routing head has no way
to see their own efficiency, or to predict how a change in turn count moves the
composite score, without the hidden benchmark.

This module makes both first-class and offline. It re-implements the *same*
formula (kept byte-for-byte in sync with ``pr_eval``; a test pins them equal),
adds the per-answer efficiency analytics the aggregate score hides
(turns/calls/cost per correct answer), and can read the turn counts straight off
the ``Trajectory`` objects the pipeline already produces.

Pure / deterministic / no network / no GPU / no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "DEFAULT_MAX_TURNS",
    "SCORE_WEIGHTS",
    "TurnRecord",
    "ScoreBreakdown",
    "EfficiencySummary",
    "turn_efficiency",
    "composite_score",
    "avg_turns",
    "summarize_efficiency",
    "trajectory_turn_records",
]

#: Turn budget K used by the live multi-turn scorer (SPEC / pr_eval).
DEFAULT_MAX_TURNS = 5

#: Composite-score weights (hidden, live, efficiency, novelty). Sum to 1.0.
SCORE_WEIGHTS = {"hidden": 0.70, "live": 0.15, "efficiency": 0.10, "novelty": 0.05}


def turn_efficiency(avg_turns_used: float, live_acc: float, *,
                    max_turns: int = DEFAULT_MAX_TURNS) -> float:
    """The efficiency term of the composite score.

    ``max(0, (max_turns - avg_turns) / (max_turns - 1)) * live_acc``, and ``0``
    when ``live_acc <= 0`` (an inefficient wrong answer earns nothing). This is
    the same expression the hidden scorer uses
    (``scripts/pr_eval.py::_compute_score``); a unit test pins them equal.

    A single-turn correct answer (``avg_turns == 1``) scores the full ``live_acc``;
    using the whole budget (``avg_turns == max_turns``) scores ``0``.

    Args:
        avg_turns_used: Mean turns per task on the live multi-turn eval.
        live_acc: Live multi-turn accuracy in ``[0, 1]``.
        max_turns: The turn budget ``K`` (>= 2).

    Returns:
        The efficiency component in ``[0, live_acc]``.
    """
    if live_acc <= 0.0:
        return 0.0
    denom = max(1, int(max_turns) - 1)
    frac = (max_turns - avg_turns_used) / denom
    return max(0.0, frac) * live_acc


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every component of the composite score, plus its weighted total.

    ``efficiency`` here is the raw efficiency term (before its 0.10 weight); the
    ``weighted`` map holds each part's contribution to ``total`` so a caller can
    see exactly where the score comes from.
    """

    hidden_acc: float
    live_acc: float
    efficiency: float
    novelty: float
    total: float
    weighted: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "hidden_acc": self.hidden_acc,
            "live_acc": self.live_acc,
            "efficiency": self.efficiency,
            "novelty": self.novelty,
            "weighted": dict(self.weighted),
            "total": self.total,
        }


def composite_score(
    *,
    hidden_acc: float,
    live_acc: float,
    avg_turns_used: float,
    novelty: float = 0.0,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> ScoreBreakdown:
    """Predict the composite competition score from its four inputs.

    Mirrors ``scripts/pr_eval.py::_compute_score`` exactly, but returns the full
    breakdown (each component and its weighted contribution) instead of only the
    scalar, so the result is inspectable offline.

    Args:
        hidden_acc: Cached single-turn accuracy (70%).
        live_acc: Live multi-turn accuracy (15%).
        avg_turns_used: Mean turns per task on the live eval (feeds efficiency).
        novelty: Novelty vs the current king (5%); ``0`` if unknown.
        max_turns: Live turn budget ``K``.

    Returns:
        A :class:`ScoreBreakdown`.
    """
    eff = turn_efficiency(avg_turns_used, live_acc, max_turns=max_turns)
    weighted = {
        "hidden": SCORE_WEIGHTS["hidden"] * hidden_acc,
        "live": SCORE_WEIGHTS["live"] * live_acc,
        "efficiency": SCORE_WEIGHTS["efficiency"] * eff,
        "novelty": SCORE_WEIGHTS["novelty"] * novelty,
    }
    return ScoreBreakdown(
        hidden_acc=hidden_acc, live_acc=live_acc, efficiency=eff, novelty=novelty,
        total=sum(weighted.values()), weighted=weighted,
    )


@dataclass(frozen=True)
class TurnRecord:
    """One task's live outcome: was it correct, and at what cost.

    ``turns`` and ``llm_calls`` are distinct: a turn is one Thinker/Worker/
    Verifier round, while ``llm_calls`` counts every model call the turn made
    (a workflow step can fan out to several workers). Either can be missing.
    """

    correct: bool
    turns: int
    llm_calls: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class EfficiencySummary:
    """Per-answer efficiency over a batch of live tasks.

    The composite score's efficiency term is an aggregate; these are the
    per-*correct-answer* costs it hides, which is what a contributor tunes.
    ``*_per_correct`` are ``inf`` when nothing was solved (an honest "undefined",
    not a silent 0).
    """

    n_tasks: int
    n_correct: int
    accuracy: float
    avg_turns: float
    avg_turns_correct: float
    turns_per_correct: float
    calls_per_correct: float | None
    cost_per_correct: float | None
    efficiency: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_tasks": self.n_tasks,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "avg_turns": self.avg_turns,
            "avg_turns_correct": self.avg_turns_correct,
            "turns_per_correct": self.turns_per_correct,
            "calls_per_correct": self.calls_per_correct,
            "cost_per_correct": self.cost_per_correct,
            "efficiency": self.efficiency,
        }


def avg_turns(
    records: Sequence[TurnRecord], *, max_turns: int = DEFAULT_MAX_TURNS,
    penalize_missing: bool = True,
) -> float:
    """Mean turns per task, mirroring the hidden scorer's accounting.

    ``pr_eval`` charges a failed/errored trajectory the full ``max_turns`` rather
    than dropping it, so an answer that never terminated cannot look efficient.
    With ``penalize_missing`` (default) a record whose ``turns <= 0`` is counted
    as ``max_turns``. Returns ``0.0`` for an empty batch.
    """
    if not records:
        return 0.0
    total = 0
    for r in records:
        total += max_turns if (penalize_missing and r.turns <= 0) else r.turns
    return total / len(records)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_efficiency(
    records: Sequence[TurnRecord], *, max_turns: int = DEFAULT_MAX_TURNS,
) -> EfficiencySummary:
    """Aggregate live task outcomes into per-answer efficiency analytics.

    ``turns_per_correct`` / ``calls_per_correct`` / ``cost_per_correct`` divide the
    total turns / calls / cost across ALL tasks by the number of correct answers —
    the real unit price of a right answer, including the tries that missed. They
    are ``inf`` / ``None`` when nothing was solved.
    """
    n = len(records)
    correct = [r for r in records if r.correct]
    n_correct = len(correct)
    acc = n_correct / n if n else 0.0

    mean_turns = avg_turns(records, max_turns=max_turns)
    eff = turn_efficiency(mean_turns, acc, max_turns=max_turns)

    total_turns = sum(
        (max_turns if r.turns <= 0 else r.turns) for r in records
    )
    turns_per_correct = total_turns / n_correct if n_correct else float("inf")

    calls = [r.llm_calls for r in records if r.llm_calls is not None]
    calls_per_correct: float | None
    if calls and len(calls) == n:
        calls_per_correct = sum(calls) / n_correct if n_correct else float("inf")
    else:
        calls_per_correct = None

    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    cost_per_correct: float | None
    if costs and len(costs) == n:
        cost_per_correct = sum(costs) / n_correct if n_correct else float("inf")
    else:
        cost_per_correct = None

    return EfficiencySummary(
        n_tasks=n,
        n_correct=n_correct,
        accuracy=acc,
        avg_turns=mean_turns,
        avg_turns_correct=_mean([float(r.turns) for r in correct]),
        turns_per_correct=turns_per_correct,
        calls_per_correct=calls_per_correct,
        cost_per_correct=cost_per_correct,
        efficiency=eff,
    )


def trajectory_turn_records(
    trajectories: Iterable[Any],
    correctness: Iterable[float] | None = None,
    *,
    score_fn: Any = None,
) -> list[TurnRecord]:
    """Build :class:`TurnRecord`\\ s from the pipeline's ``Trajectory`` objects.

    A ``Trajectory`` exposes ``n_turns`` (``len(turns)``); correctness is taken
    from ``correctness`` when provided (one value per trajectory), otherwise from
    ``score_fn(traj)`` (defaults to :func:`trinity.orchestration.reward.score`,
    imported lazily so this module needs no heavy deps to import).

    Args:
        trajectories: Completed trajectories.
        correctness: Optional per-trajectory ``{0,1}`` / bool scores.
        score_fn: Override the grader; must return a float where ``> 0`` is correct.

    Returns:
        One record per trajectory (``llm_calls``/``cost`` left ``None``).
    """
    trajs = list(trajectories)
    if correctness is not None:
        marks = [bool(c) for c in correctness]
        if len(marks) != len(trajs):
            raise ValueError(
                f"correctness has {len(marks)} entries, expected {len(trajs)}"
            )
    else:
        if score_fn is None:
            from trinity.orchestration.reward import score as score_fn  # lazy
        marks = [score_fn(t) > 0.0 for t in trajs]

    return [
        TurnRecord(correct=marks[i], turns=int(getattr(t, "n_turns", 0)))
        for i, t in enumerate(trajs)
    ]
