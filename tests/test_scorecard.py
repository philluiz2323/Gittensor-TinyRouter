"""Offline tests for the predicted-score-range scorecard. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.efficiency import composite_score
from trinity.scorecard import scorecard


def _score(benchmark: str, candidate: str, reference: object) -> float:
    """Stub grader: correct iff the candidate equals the reference string."""
    del benchmark
    return 1.0 if candidate == str(reference) else 0.0


def _item(qid: str, answers: dict, gold: str) -> dict:
    return {"question_id": qid, "benchmark": "math500",
            "correct_answer": gold, "model_answers": answers}


def _complementary_items():
    # Two models, each solves one of two questions -> best_single 0.5, oracle 1.0.
    return [
        _item("q0", {"a": "4", "b": "x"}, "4"),   # only a
        _item("q1", {"a": "x", "b": "6"}, "6"),   # only b
    ]


def test_bounds_come_from_best_single_and_oracle():
    card = scorecard(_complementary_items(), score_fn=_score)
    assert card.best_single_accuracy == 0.5
    assert card.oracle_accuracy == 1.0
    assert card.headroom == 0.5
    assert set(card.models) == {"a", "b"}


def test_predicted_score_range_uses_the_two_hidden_bounds():
    card = scorecard(_complementary_items(), score_fn=_score,
                     live_acc=0.6, avg_turns=2.0, novelty=0.2)
    floor = composite_score(hidden_acc=0.5, live_acc=0.6, avg_turns_used=2.0, novelty=0.2)
    ceil = composite_score(hidden_acc=1.0, live_acc=0.6, avg_turns_used=2.0, novelty=0.2)
    assert card.score_floor.total == pytest.approx(floor.total)
    assert card.score_ceiling.total == pytest.approx(ceil.total)
    # The ceiling must dominate the floor (higher hidden accuracy).
    assert card.score_ceiling.total > card.score_floor.total


def test_no_headroom_when_one_model_dominates():
    # 'a' solves both; oracle == best_single -> zero headroom, floor == ceiling.
    items = [_item("q0", {"a": "4", "b": "x"}, "4"), _item("q1", {"a": "6", "b": "x"}, "6")]
    card = scorecard(items, score_fn=_score, live_acc=0.5, avg_turns=1.0)
    assert card.best_single_accuracy == 1.0 and card.oracle_accuracy == 1.0
    assert card.headroom == 0.0
    assert card.score_floor.total == pytest.approx(card.score_ceiling.total)


def test_live_efficiency_novelty_flow_into_the_prediction():
    items = _complementary_items()
    lazy = scorecard(items, score_fn=_score, live_acc=0.5, avg_turns=5.0)   # max turns -> eff 0
    quick = scorecard(items, score_fn=_score, live_acc=0.5, avg_turns=1.0)  # 1 turn -> full eff
    assert quick.score_floor.total > lazy.score_floor.total


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no cached"):
        scorecard([{"question_id": "q", "benchmark": "math500",
                    "correct_answer": "4", "model_answers": {}}], score_fn=_score)


def test_scorecard_roundtrips_to_dict():
    card = scorecard(_complementary_items(), score_fn=_score, live_acc=0.3)
    d = card.to_dict()
    assert d["best_single_accuracy"] == 0.5 and d["oracle_accuracy"] == 1.0
    assert "floor" in d["predicted_score"] and "ceiling" in d["predicted_score"]
    assert d["inputs"]["live_acc"] == 0.3


def test_default_grader_is_the_real_one():
    # Without a stub, grading goes through the real adapter (math \boxed{}).
    items = [
        {"question_id": "q0", "benchmark": "math500", "correct_answer": "42",
         "model_answers": {"a": r"\boxed{42}", "b": r"\boxed{7}"}},
    ]
    card = scorecard(items, live_acc=0.0)
    assert card.per_model_accuracy["a"] == 1.0
    assert card.per_model_accuracy["b"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
