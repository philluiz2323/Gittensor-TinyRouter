"""Offline tests for the DROP reading-comprehension adapter (issue #275).

No network, no GPU, no torch, and NO code execution (DROP is pure-text). Covers schema
round-trip, the guarded loader + toy fallback, prompt shape, and the DROP EM/token-F1
metric. All loader calls fake ``datasets`` so CI is network-independent.
"""
from __future__ import annotations

import sys
import types

import pytest

from trinity.adapters import available_adapters, get_adapter
from trinity.adapters.base import TaskType
from trinity.adapters.drop import (
    BENCHMARK,
    DropAdapter,
    DropReference,
    build_drop_prompt,
    drop_em_f1,
    load_drop_tasks,
    score_drop,
)


def _fake_datasets(monkeypatch, rows):
    module = types.ModuleType("datasets")
    module.load_dataset = lambda path, name=None, split=None: rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --- registry / schema ---


def test_registered_and_task_type():
    assert BENCHMARK in available_adapters()
    assert isinstance(get_adapter(BENCHMARK), DropAdapter)
    assert get_adapter(BENCHMARK).task_type() is TaskType.MATH


def test_reference_roundtrips_and_validates():
    ref = DropReference(gold_answers=["21", "twenty one"])
    assert DropReference.from_dict(ref.to_dict()) == ref
    assert ref.is_valid()
    assert not DropReference(gold_answers=[]).is_valid()
    assert not DropReference(gold_answers=["", "  "]).is_valid()


# --- loader / toy fallback ---


def test_toy_fallback_when_datasets_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)   # import raises -> toy
    tasks = load_drop_tasks("test", None, seed=0)
    assert len(tasks) == 2
    for t in tasks:
        assert t.benchmark == BENCHMARK and t.answer["gold_answers"]
        assert "Passage:" in t.prompt and "Question:" in t.prompt


def test_loader_parses_hf_rows_spans_and_number(monkeypatch):
    _fake_datasets(monkeypatch, [
        {"passage": "P.", "question": "Who?", "answers_spans": {"spans": ["Ana Ruiz"], "types": ["span"]}},
        {"passage": "P.", "question": "How many?", "answers_spans": {"spans": [], "number": "21"}},
        {"passage": "", "question": "skip?", "answers_spans": {"spans": ["x"]}},   # no passage -> skipped
    ])
    tasks = load_drop_tasks("validation", None, seed=0)
    golds = {tuple(t.answer["gold_answers"]) for t in tasks}
    assert ("Ana Ruiz",) in golds and ("21",) in golds and len(tasks) == 2


def test_loader_is_deterministic_and_truncates(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    a = load_drop_tasks("test", max_items=1, seed=5)
    b = load_drop_tasks("test", max_items=1, seed=5)
    assert [t.task_id for t in a] == [t.task_id for t in b] and len(a) == 1


# --- prompt ---


def test_prompt_shape():
    p = build_drop_prompt("Some passage.", "What is X?")
    assert p.startswith("Passage:") and "Question: What is X?" in p and "Answer:" in p


# --- DROP metric ---


def test_em_f1_exact_and_partial():
    assert drop_em_f1("21", ["21"]) == (1.0, 1.0)
    em, f1 = drop_em_f1("Ana", ["Ana Ruiz"])
    assert em == 0.0 and 0.0 < f1 < 1.0
    # best over a gold set
    assert drop_em_f1("blue", ["red", "blue"]) == (1.0, 1.0)


def test_number_normalization():
    assert score_drop("Answer: 16", {"gold_answers": ["16.0"]}) == 1.0
    assert score_drop("Answer: 16.0", {"gold_answers": ["16"]}) == 1.0
    assert score_drop("Answer: 17", {"gold_answers": ["16"]}) == 0.0


def test_article_punctuation_case_folding():
    ref = {"gold_answers": ["Ana Ruiz"]}
    assert score_drop("Answer: the ana ruiz.", ref) == 1.0   # article + case + punctuation
    ref2 = {"gold_answers": ["New York City"]}
    assert score_drop("Answer: new york city", ref2) == 1.0


def test_partial_and_extra_tokens_score_zero_under_binary_rule():
    ref = {"gold_answers": ["21"]}
    assert score_drop("Answer: 21 houses", ref) == 0.0   # extra token -> F1 < 1
    assert score_drop("Answer: Ana", {"gold_answers": ["Ana Ruiz"]}) == 0.0   # missing token


def test_multi_span_bag_match():
    # A multi-span gold answer; the correct set (order-independent) matches.
    ref = {"gold_answers": ["Ana Ruiz Ben Cho"]}
    assert score_drop("Answer: Ben Cho Ana Ruiz", ref) == 1.0
    assert score_drop("Answer: Ana Ruiz", ref) == 0.0


def test_bare_and_list_references():
    assert score_drop("Answer: 21", "21") == 1.0
    assert score_drop("Answer: blue", ["red", "blue"]) == 1.0
    assert score_drop("Answer: green", ["red", "blue"]) == 0.0
    assert score_drop("Answer: x", {"gold_answers": []}) == 0.0


# --- adapter wiring ---


def test_adapter_scores_via_score_drop():
    a = DropAdapter()
    assert a.score_output("Answer: 21", {"gold_answers": ["21"]}) == 1.0
    assert a.score_output("Answer: 22", {"gold_answers": ["21"]}) == 0.0


def test_serialize_task_shape(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    task = load_drop_tasks("test", 1, 0)[0]
    d = DropAdapter().serialize_task(task)
    assert set(d) == {"task_id", "benchmark", "prompt", "reference", "task_type", "meta"}
    assert d["task_type"] == TaskType.MATH.value and d["reference"]["gold_answers"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
