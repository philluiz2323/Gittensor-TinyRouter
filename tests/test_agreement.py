"""Offline tests for the per-question agreement / contested-subset analysis.

No network, no GPU, no torch: a stub scorer stands in for the benchmark adapter.
"""
from __future__ import annotations

import pytest

from trinity.analysis import (
    contested_ids,
    grade_item,
    grade_items,
    summarize,
    to_oracle_matrix,
)


def _score(benchmark: str, candidate: str, reference: object) -> float:
    """Stub grader: an answer is correct iff it equals the reference string."""
    del benchmark
    return 1.0 if candidate == str(reference) else 0.0


def _item(qid: str, answers: dict, *, gold: str = "4", **extra):
    item = {
        "question_id": qid,
        "benchmark": "math500",
        "question_text": "2+2?",
        "correct_answer": gold,
        "model_answers": answers,
    }
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# grade_item
# ---------------------------------------------------------------------------
def test_grade_item_marks_each_model_right_or_wrong():
    rec = grade_item(_item("q0", {"a": "4", "b": "5"}), score_fn=_score)
    assert rec.per_model_correct == {"a": 1, "b": 0}
    assert rec.question_id == "q0" and rec.benchmark == "math500"
    assert rec.n_models == 2 and rec.n_correct == 1


def test_grade_item_prefers_cached_scores_over_regrading():
    # model_scores says "a" was right even though its text would grade wrong.
    item = _item("q0", {"a": "wrong-text"}, model_scores={"a": 1.0})
    assert grade_item(item, score_fn=_score).per_model_correct == {"a": 1}
    # ...and re-grading can be forced.
    assert grade_item(item, score_fn=_score, use_cached_scores=False).per_model_correct == {"a": 0}


def test_grade_item_treats_a_null_answer_as_wrong():
    rec = grade_item(_item("q0", {"a": None}), score_fn=_score)
    assert rec.per_model_correct == {"a": 0}


# ---------------------------------------------------------------------------
# contested / unanimous classification
# ---------------------------------------------------------------------------
def test_contested_is_exactly_the_split_decisions():
    contested = grade_item(_item("q", {"a": "4", "b": "5"}), score_fn=_score)
    all_right = grade_item(_item("q", {"a": "4", "b": "4"}), score_fn=_score)
    all_wrong = grade_item(_item("q", {"a": "5", "b": "6"}), score_fn=_score)

    assert contested.is_contested
    assert not contested.is_unanimous_correct and not contested.is_unanimous_wrong

    assert all_right.is_unanimous_correct and not all_right.is_contested
    assert all_wrong.is_unanimous_wrong and not all_wrong.is_contested


def test_contested_ids_returns_only_the_disagreement_subset():
    items = [
        _item("q-contested", {"a": "4", "b": "5"}),
        _item("q-all-right", {"a": "4", "b": "4"}),
        _item("q-all-wrong", {"a": "9", "b": "9"}),
    ]
    records = grade_items(items, score_fn=_score)
    assert contested_ids(records) == ["q-contested"]


# ---------------------------------------------------------------------------
# grade_items guards
# ---------------------------------------------------------------------------
def test_items_without_cached_answers_are_skipped_not_counted_wrong():
    items = [_item("q0", {"a": "4"}), _item("q-live", {})]
    records = grade_items(items, score_fn=_score)
    assert [r.question_id for r in records] == ["q0"]


def test_a_ragged_model_pool_is_an_error():
    items = [_item("q0", {"a": "4", "b": "4"}), _item("q1", {"a": "4"})]
    with pytest.raises(ValueError, match="expected"):
        grade_items(items, score_fn=_score)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summary_counts_rates_best_single_and_headroom():
    items = [
        _item("q0", {"a": "4", "b": "5"}),   # contested, a right
        _item("q1", {"a": "4", "b": "5"}),   # contested, a right
        _item("q2", {"a": "4", "b": "4"}),   # all right
        _item("q3", {"a": "9", "b": "9"}),   # all wrong
    ]
    s = summarize(grade_items(items, score_fn=_score))

    assert s.n_questions == 4 and s.models == ["a", "b"]
    assert (s.n_contested, s.n_unanimous_correct, s.n_unanimous_wrong) == (2, 1, 1)
    assert s.disagreement_rate == 0.5
    assert s.per_model_accuracy == {"a": 0.75, "b": 0.25}
    assert s.best_single_model == "a" and s.best_single_accuracy == 0.75
    # 3 of 4 questions are solved by at least one model.
    assert s.oracle_any == 0.75
    # A perfect router cannot beat the best single model here.
    assert s.headroom == 0.0


def test_headroom_is_positive_when_models_are_complementary():
    items = [
        _item("q0", {"a": "4", "b": "5"}),   # only a
        _item("q1", {"a": "5", "b": "4"}),   # only b
    ]
    s = summarize(grade_items(items, score_fn=_score))
    assert s.best_single_accuracy == 0.5
    assert s.oracle_any == 1.0
    assert s.headroom == 0.5


def test_summary_of_nothing_is_all_zero_not_an_error():
    s = summarize([])
    assert s.n_questions == 0 and s.models == [] and s.best_single_model is None
    assert s.disagreement_rate == 0.0 and s.headroom == 0.0
    assert s.to_dict()["n_questions"] == 0


def test_headroom_never_goes_negative():
    s = summarize(grade_items([_item("q0", {"a": "4"})], score_fn=_score))
    assert s.headroom == 0.0


# ---------------------------------------------------------------------------
# to_oracle_matrix: must satisfy oracle_ceiling.matrix_to_tensor
# ---------------------------------------------------------------------------
def test_matrix_shape_matches_the_oracle_ceiling_contract():
    items = [_item("q0", {"a": "4", "b": "5"}), _item("q1", {"a": "5", "b": "4"})]
    matrix = to_oracle_matrix(grade_items(items, score_fn=_score))

    assert matrix["benchmark"] == "math500"
    assert matrix["n_samples"] == 1
    assert [t["id"] for t in matrix["tasks"]] == ["q0", "q1"]
    # Every (question, model) cell is a length-1 list of 0/1 -> uniform K == 1.
    for task in matrix["tasks"]:
        assert sorted(task["per_model"]) == ["a", "b"]
        for cell in task["per_model"].values():
            assert isinstance(cell, list) and len(cell) == 1
            assert cell[0] in (0, 1)
    assert matrix["tasks"][0]["per_model"] == {"a": [1], "b": [0]}


def test_matrix_benchmark_name_can_be_overridden():
    records = grade_items([_item("q0", {"a": "4"})], score_fn=_score)
    assert to_oracle_matrix(records, benchmark="mmlu")["benchmark"] == "mmlu"


def test_empty_matrix_is_still_well_formed():
    matrix = to_oracle_matrix([])
    assert matrix["tasks"] == [] and matrix["n_samples"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
