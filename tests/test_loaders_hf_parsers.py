"""Offline coverage for the per-benchmark HuggingFace row parsers in loaders.py.

`trinity.adapters.loaders` turns raw dataset rows into `Task` objects for
math500 / mmlu / gpqa / livecodebench. The existing suite exercises the split
resolution and the toy fallback, but the actual row-parsing loops (and the
`_try_load_hf` / `_row_get` guards) were uncovered — `loaders.py` sat at 70%.

These tests fake the `datasets` module so the *online* parsing path runs with no
network: `_try_load_hf` imports `datasets` lazily, so injecting
`sys.modules["datasets"]` is enough. `loaders.py` imports no torch, so this file
has no ordering constraint.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from trinity.adapters import loaders


def _fake_datasets(monkeypatch, handler):
    """Install a fake `datasets` whose `load_dataset` delegates to `handler`."""
    module = types.ModuleType("datasets")
    module.load_dataset = handler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def _rows(monkeypatch, path_rows):
    """Fake `datasets` that returns rows keyed by dataset path (raises otherwise)."""

    def load_dataset(path, name=None, split=None):
        if path not in path_rows:
            raise ValueError(f"no fake rows for {path!r}")
        return path_rows[path]

    _fake_datasets(monkeypatch, load_dataset)


# --------------------------------------------------------------------------- #
# _try_load_hf / _row_get guards
# --------------------------------------------------------------------------- #
def test_try_load_hf_returns_none_when_datasets_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)  # import raises
    assert loaders._try_load_hf("any/dataset") is None


def test_try_load_hf_returns_none_on_load_error(monkeypatch):
    def boom(path, name=None, split=None):
        raise RuntimeError("gated / offline / unknown id")

    _fake_datasets(monkeypatch, boom)
    assert loaders._try_load_hf("any/dataset") is None


def test_row_get_first_present_key():
    assert loaders._row_get({"b": 2, "a": 1}, "a", "b") == 1
    assert loaders._row_get({"b": 2}, "a", "b") == 2


def test_row_get_skips_none_values():
    assert loaders._row_get({"a": None, "b": 5}, "a", "b") == 5


def test_row_get_non_mapping_row_returns_default():
    # `"k" in 123` raises TypeError -> the loop breaks -> default is returned.
    assert loaders._row_get(123, "k", default="d") == "d"


# --------------------------------------------------------------------------- #
# math500
# --------------------------------------------------------------------------- #
def test_math500_parses_primary_schema(monkeypatch):
    _rows(monkeypatch, {"HuggingFaceH4/MATH-500": [
        {"problem": "2+2?", "answer": "4", "subject": "algebra", "level": 3},
        {"question": "3+3?", "solution": "6"},   # alternate field names
        {"problem": "", "answer": "x"},           # no problem -> skipped
    ]})
    tasks = loaders._load_math500_hf("test")
    assert [t.task_id for t in tasks] == ["math500-0", "math500-1"]
    assert tasks[0].benchmark == "math500"
    assert tasks[0].answer == "4"
    assert tasks[0].meta["source"] == "HuggingFaceH4/MATH-500"
    assert tasks[0].meta["subject"] == "algebra"
    assert tasks[1].prompt == "3+3?" and tasks[1].answer == "6"


def test_math500_falls_back_to_competition_math(monkeypatch):
    def load_dataset(path, name=None, split=None):
        if path == "HuggingFaceH4/MATH-500":
            raise ValueError("gated")
        if path == "qwedsacf/competition_math":
            return [{"problem": "q", "solution": "a"}]
        raise ValueError(path)

    _fake_datasets(monkeypatch, load_dataset)
    tasks = loaders._load_math500_hf("test")
    assert len(tasks) == 1
    assert tasks[0].meta["source"] == "qwedsacf/competition_math"


def test_math500_all_rows_empty_returns_none(monkeypatch):
    _rows(monkeypatch, {"HuggingFaceH4/MATH-500": [{"problem": ""}]})
    assert loaders._load_math500_hf("test") is None


# --------------------------------------------------------------------------- #
# mmlu skip branches
# --------------------------------------------------------------------------- #
def test_mmlu_skips_malformed_rows_and_parses_valid(monkeypatch):
    _rows(monkeypatch, {"cais/mmlu": [
        {"question": "", "choices": ["a", "b"], "answer": 0},        # no question
        {"question": "q", "choices": ["a", "b"], "answer": "x"},     # answer not int
        {"question": "q", "choices": ["a"], "answer": 99},           # index out of range
        {"question": "ok", "choices": ["a", "b", "c", "d"], "answer": 2},  # valid -> C
    ]})
    tasks = loaders._load_mmlu_hf("test")
    assert len(tasks) == 1
    assert tasks[0].answer == "C"
    assert tasks[0].meta["choices"] == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------- #
# gpqa
# --------------------------------------------------------------------------- #
def test_gpqa_parses_and_shuffles_options(monkeypatch):
    _rows(monkeypatch, {"Idavidrein/gpqa": [
        {"Question": "too few", "Correct Answer": "c", "Incorrect Answer 1": "x"},  # <3 distractors -> skip
        {
            "Question": "real?", "Correct Answer": "CORRECT",
            "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2",
            "Incorrect Answer 3": "w3",
        },
    ]})
    tasks = loaders._load_gpqa_hf("train")
    assert len(tasks) == 1
    t = tasks[0]
    assert t.benchmark == "gpqa"
    assert t.answer in ("A", "B", "C", "D")
    # The four options are a permutation of correct + three distractors.
    assert sorted(t.meta["choices"]) == sorted(["CORRECT", "w1", "w2", "w3"])
    # The answer letter points at the correct option in the shuffled order.
    assert t.meta["choices"]["ABCD".index(t.answer)] == "CORRECT"


def test_gpqa_shuffle_is_deterministic(monkeypatch):
    row = {
        "Question": "q", "Correct Answer": "c",
        "Incorrect Answer 1": "a", "Incorrect Answer 2": "b", "Incorrect Answer 3": "d",
    }
    _rows(monkeypatch, {"Idavidrein/gpqa": [row]})
    first = loaders._load_gpqa_hf("train")[0]
    _rows(monkeypatch, {"Idavidrein/gpqa": [row]})
    second = loaders._load_gpqa_hf("train")[0]
    assert first.answer == second.answer
    assert first.meta["choices"] == second.meta["choices"]


# --------------------------------------------------------------------------- #
# livecodebench
# --------------------------------------------------------------------------- #
def test_livecodebench_parses_via_config_name(monkeypatch):
    _rows(monkeypatch, {"livecodebench/code_generation_lite": [{
        "question_content": "sum stdin", "question_id": "lcb-42",
        "public_test_cases": json.dumps([{"input": "2 3\n", "output": "5\n"}]),
        "fn_name": "solve", "starter_code": "def solve(): ...",
        "platform": "leetcode", "difficulty": "easy",
    }]})
    tasks = loaders._load_livecodebench_hf("test")
    assert len(tasks) == 1
    t = tasks[0]
    assert t.task_id == "lcb-42"
    assert t.benchmark == "livecodebench"
    assert t.answer["tests"] == [{"input": "2 3\n", "output": "5\n"}]
    assert t.answer["fn_name"] == "solve"
    assert t.answer["starter_code"] == "def solve(): ..."
    assert t.meta["version"] == "release_v6"  # "test" -> v6


def test_livecodebench_falls_back_to_split_mirror(monkeypatch):
    def load_dataset(path, name=None, split=None):
        # Config-name form fails; the mirror exposes the version via `split`.
        if name is not None:
            raise ValueError("no such config")
        if split == "release_v6":
            return [{"question_content": "q", "question_id": "m1"}]
        raise ValueError(f"unexpected split {split!r}")

    _fake_datasets(monkeypatch, load_dataset)
    tasks = loaders._load_livecodebench_hf("test")
    assert len(tasks) == 1
    assert tasks[0].task_id == "m1"


def test_livecodebench_missing_question_is_skipped_and_id_falls_back(monkeypatch):
    _rows(monkeypatch, {"livecodebench/code_generation_lite": [
        {"question_content": ""},                       # skipped
        {"question_content": "has content"},            # no question_id -> lcb-<i>
    ]})
    tasks = loaders._load_livecodebench_hf("v1")
    assert len(tasks) == 1
    assert tasks[0].task_id == "lcb-1"  # index-based fallback id
    assert tasks[0].meta["version"] == "release_v1"


# --------------------------------------------------------------------------- #
# _parse_lcb_tests variants
# --------------------------------------------------------------------------- #
def test_parse_lcb_tests_from_json_string():
    row = {"public_test_cases": json.dumps([{"input": "a", "output": "b"}])}
    assert loaders._parse_lcb_tests(row) == [{"input": "a", "output": "b"}]


def test_parse_lcb_tests_from_list_with_alt_keys():
    row = {"tests": [{"stdin": "x", "expected_output": "y"}]}
    assert loaders._parse_lcb_tests(row) == [{"input": "x", "output": "y"}]


def test_parse_lcb_tests_bad_json_returns_empty():
    assert loaders._parse_lcb_tests({"test_cases": "not json {"}) == []


def test_parse_lcb_tests_absent_returns_empty():
    assert loaders._parse_lcb_tests({}) == []


def test_parse_lcb_tests_non_list_returns_empty():
    assert loaders._parse_lcb_tests({"tests": {"input": "a"}}) == []


# --------------------------------------------------------------------------- #
# _lcb_version_for_split
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", ["test", "eval", "v6", "release_v6"])
def test_lcb_version_v6(split):
    assert loaders._lcb_version_for_split(split) == "release_v6"


@pytest.mark.parametrize("split", ["train", "v1", "", "anything"])
def test_lcb_version_defaults_to_v1(split):
    assert loaders._lcb_version_for_split(split) == "release_v1"


# --------------------------------------------------------------------------- #
# load_split / _toy_tasks unknown-benchmark guards
# --------------------------------------------------------------------------- #
def test_load_split_unknown_benchmark_raises():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        loaders.load_split("not_a_benchmark", "test", max_items=1, seed=0)


def test_toy_tasks_unknown_benchmark_raises():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        loaders._toy_tasks("not_a_benchmark")


def test_load_split_real_data_shuffles_and_truncates(monkeypatch):
    """The canonical path: real rows -> resolved split -> shuffle -> truncate."""
    _rows(monkeypatch, {"HuggingFaceH4/MATH-500": [
        {"problem": f"q{i}", "answer": str(i)} for i in range(10)
    ]})
    tasks = loaders.load_split("math500", "test", max_items=3, seed=0)
    assert len(tasks) == 3
    assert all(t.benchmark == "math500" for t in tasks)
    # Deterministic for a fixed seed.
    _rows(monkeypatch, {"HuggingFaceH4/MATH-500": [
        {"problem": f"q{i}", "answer": str(i)} for i in range(10)
    ]})
    again = loaders.load_split("math500", "test", max_items=3, seed=0)
    assert [t.task_id for t in tasks] == [t.task_id for t in again]
