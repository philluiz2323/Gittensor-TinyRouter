"""Offline tests for the BIG-Bench Hard (BBH) adapter (issue #269).

No network, no GPU, no torch, and NO code execution (BBH is a pure-text benchmark).
Covers schema round-trip, the guarded loader + toy fallback, prompt shape, and the
two scoring formats (multiple-choice letter + exact-match string).
"""
from __future__ import annotations

import sys
import types

import pytest

from trinity.adapters import available_adapters, get_adapter
from trinity.adapters.base import TaskType
from trinity.adapters.bbh import (
    BENCHMARK,
    SUBTASKS,
    BBHAdapter,
    BBHReference,
    build_bbh_prompt,
    load_bbh_tasks,
    score_bbh,
)


def _fake_datasets(monkeypatch, rows_by_subtask):
    module = types.ModuleType("datasets")

    def load_dataset(path, name=None, split=None):
        return rows_by_subtask.get(name, [])

    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --- registry / schema ---


def test_registered_and_task_type():
    assert BENCHMARK in available_adapters()
    assert isinstance(get_adapter(BENCHMARK), BBHAdapter)
    assert get_adapter(BENCHMARK).task_type() is TaskType.MCQ
    assert len(SUBTASKS) == 27


def test_reference_roundtrips_and_validates():
    ref = BBHReference(answer="(B)", answer_type="multiple_choice", subtask="date_understanding")
    assert BBHReference.from_dict(ref.to_dict()) == ref
    assert ref.is_valid()
    assert not BBHReference(answer="", answer_type="exact_match").is_valid()
    assert not BBHReference(answer="x", answer_type="bogus").is_valid()


# --- loader / toy fallback ---


def test_toy_fallback_when_datasets_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)   # import raises -> toy
    tasks = load_bbh_tasks("test", None, seed=0)
    assert len(tasks) == 3
    kinds = {t.answer["answer_type"] for t in tasks}
    assert kinds == {"multiple_choice", "exact_match"}
    for t in tasks:
        assert t.benchmark == BENCHMARK and t.answer["answer"]


def test_loader_parses_hf_rows(monkeypatch):
    _fake_datasets(monkeypatch, {
        "date_understanding": [{"input": "What date?\n(A) x\n(B) y", "target": "(B)"}],
        "object_counting": [{"input": "How many?", "target": "4"}],
    })
    tasks = load_bbh_tasks("test", None, seed=0)
    by_sub = {t.meta["subtask"]: t for t in tasks}
    assert by_sub["date_understanding"].answer["answer_type"] == "multiple_choice"
    assert by_sub["date_understanding"].answer["answer"] == "(B)"
    assert by_sub["object_counting"].answer["answer_type"] == "exact_match"


def test_loader_is_deterministic_and_truncates(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    a = load_bbh_tasks("test", max_items=2, seed=7)
    b = load_bbh_tasks("test", max_items=2, seed=7)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert len(a) == 2


# --- prompt ---


def test_prompt_instruction_matches_answer_type():
    assert "letter of the correct option" in build_bbh_prompt("Q?", "multiple_choice")
    assert "only the final" in build_bbh_prompt("Q?", "exact_match")


# --- scoring: multiple-choice ---


def test_multiple_choice_scoring():
    ref = {"answer": "(B)", "answer_type": "multiple_choice"}
    assert score_bbh("Reasoning... Answer: (B)", ref) == 1.0
    assert score_bbh("The answer is B", ref) == 1.0           # bare letter
    assert score_bbh("Answer: (C)", ref) == 0.0
    assert score_bbh("no letter here", ref) == 0.0


def test_multiple_choice_supports_high_letters():
    ref = {"answer": "(G)", "answer_type": "multiple_choice"}   # up to 7-object logical deduction
    assert score_bbh("... Answer: (G)", ref) == 1.0
    assert score_bbh("... Answer: (A)", ref) == 0.0


# --- scoring: exact-match ---


def test_exact_match_scoring_boolean_and_number():
    assert score_bbh("so the final answer is True.", {"answer": "True", "answer_type": "exact_match"}) == 1.0
    assert score_bbh("Answer: False", {"answer": "True", "answer_type": "exact_match"}) == 0.0
    assert score_bbh("Counting... Answer: 4", {"answer": "4", "answer_type": "exact_match"}) == 1.0
    assert score_bbh("Answer: 5", {"answer": "4", "answer_type": "exact_match"}) == 0.0


def test_exact_match_is_normalized():
    ref = {"answer": "valid", "answer_type": "exact_match"}
    assert score_bbh("Answer: Valid.", ref) == 1.0          # case + trailing punctuation
    assert score_bbh("bird cat dog", {"answer": "bird cat dog", "answer_type": "exact_match"}) == 1.0
    # last-line fallback when there is no explicit "answer:" lead
    assert score_bbh("reasoning line\nTrue", {"answer": "True", "answer_type": "exact_match"}) == 1.0


def test_bare_string_reference_is_exact_match():
    assert score_bbh("Answer: 16", "16") == 1.0
    assert score_bbh("Answer: 17", "16") == 0.0


# --- adapter wiring ---


def test_adapter_scores_via_score_bbh_no_execution():
    a = BBHAdapter()
    ref = {"answer": "(A)", "answer_type": "multiple_choice"}
    assert a.score_output("Answer: (A)", ref) == 1.0
    assert a.score_output("Answer: (B)", ref) == 0.0


def test_serialize_task_shape(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    task = load_bbh_tasks("test", 1, 0)[0]
    d = BBHAdapter().serialize_task(task)
    assert set(d) == {"task_id", "benchmark", "prompt", "reference", "task_type", "meta"}
    assert d["reference"]["answer_type"] in ("multiple_choice", "exact_match")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
