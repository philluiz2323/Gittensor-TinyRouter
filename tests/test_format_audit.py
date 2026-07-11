"""Offline tests for the per-model format (parse-rate) audit. No network, no GPU."""
from __future__ import annotations

from trinity.format_audit import audit_items


def _has_answer(benchmark: str, text: str) -> bool:
    """Stub grader predicate: an answer is parseable iff it contains 'ANS='."""
    del benchmark
    return "ANS=" in text


def _item(benchmark: str, answers: dict) -> dict:
    return {"benchmark": benchmark, "question_id": "q", "model_answers": answers}


# ---------------------------------------------------------------------------
# per-model parse rate
# ---------------------------------------------------------------------------
def test_parse_rate_counts_extractable_answers_per_model():
    items = [
        _item("math500", {"a": "ANS=4", "b": "no answer here"}),
        _item("math500", {"a": "ANS=6", "b": "ANS=6"}),
    ]
    audit = audit_items(items, has_answer_fn=_has_answer)
    a, b = audit.per_model["a"], audit.per_model["b"]
    assert (a.n_answers, a.n_parseable, a.parse_rate) == (2, 2, 1.0)
    assert (b.n_answers, b.n_parseable, b.parse_rate) == (2, 1, 0.5)


def test_worst_model_is_the_lowest_parse_rate():
    items = [_item("math500", {"good": "ANS=1", "bad": "waffle"})]
    audit = audit_items(items, has_answer_fn=_has_answer)
    assert audit.worst_model() == "bad"


def test_overall_parse_rate_pools_every_answer():
    items = [
        _item("math500", {"a": "ANS=1", "b": "nope"}),
        _item("mmlu", {"a": "ANS=B", "b": "ANS=C"}),
    ]
    audit = audit_items(items, has_answer_fn=_has_answer)
    # 3 of 4 answers parse.
    assert audit.overall_parse_rate == 0.75
    assert audit.n_items == 2


# ---------------------------------------------------------------------------
# per-benchmark split
# ---------------------------------------------------------------------------
def test_a_model_can_be_fine_on_one_benchmark_and_bad_on_another():
    items = [
        _item("math500", {"a": "ANS=4"}),
        _item("mmlu", {"a": "letter B, unmarked"}),
    ]
    audit = audit_items(items, has_answer_fn=_has_answer)
    assert audit.per_benchmark_model["math500"]["a"].parse_rate == 1.0
    assert audit.per_benchmark_model["mmlu"]["a"].parse_rate == 0.0
    # Pooled across both benchmarks it is 0.5.
    assert audit.per_model["a"].parse_rate == 0.5


# ---------------------------------------------------------------------------
# empty / missing answers
# ---------------------------------------------------------------------------
def test_empty_and_null_answers_count_as_unparseable_and_empty():
    items = [_item("math500", {"a": "", "b": None, "c": "ANS=1"})]
    audit = audit_items(items, has_answer_fn=_has_answer)
    assert audit.per_model["a"].n_empty == 1 and audit.per_model["a"].n_parseable == 0
    assert audit.per_model["b"].n_empty == 1 and audit.per_model["b"].n_parseable == 0
    assert audit.per_model["c"].n_empty == 0 and audit.per_model["c"].n_parseable == 1


def test_n_unparseable_is_answers_minus_parseable():
    items = [_item("math500", {"a": "ANS=1"}), _item("math500", {"a": "no"})]
    s = audit_items(items, has_answer_fn=_has_answer).per_model["a"]
    assert s.n_answers == 2 and s.n_parseable == 1 and s.n_unparseable == 1


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_items_without_cached_answers_are_skipped():
    items = [_item("math500", {}), _item("math500", {"a": "ANS=1"})]
    audit = audit_items(items, has_answer_fn=_has_answer)
    assert audit.n_items == 1 and audit.per_model["a"].n_answers == 1


def test_empty_audit_is_all_zero_and_has_no_worst_model():
    audit = audit_items([], has_answer_fn=_has_answer)
    assert audit.n_items == 0
    assert audit.overall_parse_rate == 0.0
    assert audit.worst_model() is None
    assert audit.to_dict()["per_model"] == {}


# ---------------------------------------------------------------------------
# the default predicate is the grader's own has_answer
# ---------------------------------------------------------------------------
def test_default_predicate_uses_the_real_grader():
    # \boxed{...} is extractable for math500; bare prose is not.
    items = [_item("math500", {"a": r"The answer is \boxed{42}", "b": "I think it is fine"})]
    audit = audit_items(items)  # no stub -> real reward.has_answer
    assert audit.per_model["a"].n_parseable == 1
    assert audit.per_model["b"].n_parseable == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
