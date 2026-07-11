"""Offline scorecard: a predicted composite-score RANGE from cached answers.

Why this exists
---------------
The competition score is ``0.70*hidden + 0.15*live + 0.10*efficiency +
0.05*novelty`` (CONTRIBUTING.md). :mod:`trinity.efficiency` already turns the
last three into a number, but the dominant term — ``hidden`` (70%), the routed
head's cached single-turn accuracy — needs the trained head to evaluate.

What a contributor CAN compute offline, for free, from the cached
``model_answers`` in a built benchmark, is a *bound* on that routed accuracy:

* the **best single model**'s cached accuracy is a floor — a router that always
  picks the strongest worker already reaches it;
* the **any-model oracle** (a question is solved if *some* worker solves it) is
  the ceiling — no router without the labels can beat it.

The routed head's real hidden accuracy lies in ``[best_single, oracle]``. This
module grades the cached answers (reusing :mod:`trinity.analysis.agreement`),
takes those two bounds, and runs each through
:func:`trinity.efficiency.composite_score` to report the composite score the
submission would earn at the floor and at the ceiling — a predicted **range**
rather than a false point estimate. It also surfaces the routing **headroom**
(``oracle - best_single``): the accuracy a smarter router could still capture.

Zero API cost; composes already-merged analyses. No torch/network/GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from trinity.efficiency import DEFAULT_MAX_TURNS, ScoreBreakdown, composite_score

__all__ = ["ScoreCard", "scorecard"]


@dataclass(frozen=True)
class ScoreCard:
    """A predicted composite-score range plus the routing bounds behind it.

    ``score_floor`` uses ``hidden = best_single`` (always-pick-the-strongest);
    ``score_ceiling`` uses ``hidden = oracle_any`` (a perfect router). The trained
    head's real score lies between them. ``headroom`` is the accuracy a smarter
    router could still capture over the best single model.
    """

    n_questions: int
    models: list[str]
    per_model_accuracy: dict[str, float]
    best_single_model: str | None
    best_single_accuracy: float
    oracle_accuracy: float
    headroom: float
    live_acc: float
    avg_turns: float
    novelty: float
    score_floor: ScoreBreakdown
    score_ceiling: ScoreBreakdown

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_questions": self.n_questions,
            "models": list(self.models),
            "per_model_accuracy": dict(self.per_model_accuracy),
            "best_single_model": self.best_single_model,
            "best_single_accuracy": self.best_single_accuracy,
            "oracle_accuracy": self.oracle_accuracy,
            "headroom": self.headroom,
            "inputs": {
                "live_acc": self.live_acc,
                "avg_turns": self.avg_turns,
                "novelty": self.novelty,
            },
            "predicted_score": {
                "floor": self.score_floor.to_dict(),
                "ceiling": self.score_ceiling.to_dict(),
            },
        }


def scorecard(
    items: Iterable[Mapping[str, Any]],
    *,
    live_acc: float = 0.0,
    avg_turns: float = 1.0,
    novelty: float = 0.0,
    max_turns: int = DEFAULT_MAX_TURNS,
    score_fn: Any = None,
) -> ScoreCard:
    """Predict the composite-score range from cached answers + live inputs.

    The hidden (70%) term is bounded by grading the cached ``model_answers``:
    ``best_single`` accuracy (floor) and the ``any-model`` oracle (ceiling). The
    live / efficiency / novelty terms come from the caller (they need real API
    calls the offline path cannot make), defaulting to a conservative ``0``.

    Args:
        items: Built benchmark items with cached ``model_answers``.
        live_acc: Live multi-turn accuracy (15%).
        avg_turns: Mean turns per task on the live eval (feeds efficiency).
        novelty: Novelty vs the current king (5%).
        max_turns: Live turn budget.
        score_fn: Optional grader override for the cached answers.

    Returns:
        A :class:`ScoreCard`.

    Raises:
        ValueError: If no item carries cached answers (nothing to bound the
            hidden term with).
    """
    # Lazy import: the adapter registry the grader pulls in has optional deps.
    from trinity.analysis.agreement import grade_items, summarize

    records = grade_items(items, score_fn=score_fn) if score_fn else grade_items(items)
    if not records:
        raise ValueError("no cached model answers found; cannot bound the hidden term")
    summary = summarize(records)

    def _score(hidden: float) -> ScoreBreakdown:
        return composite_score(
            hidden_acc=hidden, live_acc=live_acc, avg_turns_used=avg_turns,
            novelty=novelty, max_turns=max_turns,
        )

    return ScoreCard(
        n_questions=summary.n_questions,
        models=list(summary.models),
        per_model_accuracy=dict(summary.per_model_accuracy),
        best_single_model=summary.best_single_model,
        best_single_accuracy=summary.best_single_accuracy,
        oracle_accuracy=summary.oracle_any,
        headroom=summary.headroom,
        live_acc=live_acc,
        avg_turns=avg_turns,
        novelty=novelty,
        score_floor=_score(summary.best_single_accuracy),
        score_ceiling=_score(summary.oracle_any),
    )
