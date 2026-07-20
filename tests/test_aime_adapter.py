"""Offline tests for the AIME adapter (ROADMAP "expand benchmark suite"; SPEC §6.2).

``reward.MATH_BENCHMARKS`` has always graded ``aime``, but there was no adapter or loader,
so ``get_adapter("aime")`` raised and the benchmark could not be built or evaluated — the
grader knew it, the data path did not. This adds only the data path; scoring stays the
shared math extractor, so these tests pin the *loading* contract:

* the single upstream split is aliased for logical ``test`` and carved into DISJOINT
  train/test subsets — the failure mode fixed for MMLU (#35), MMLU-Pro (#50), GPQA (#95)
  and SWE-bench (#196), where a missing ``test`` split silently served the 2-item toy set;
* both published row schemas parse (``problem``/``answer`` and ``Problem``/``Answer``);
* the gold answer is stored verbatim, because ``math_equal`` treats a leading zero as
  significant for AIME's 0-999 range — re-padding or stripping it here would change what
  grades correct.

The HF path is exercised with a faked ``datasets`` module (same trick as
``test_loaders_hf_parsers.py``), so nothing here touches the network.
"""
from __future__ import annotations

import sys
import types

import pytest

from trinity.adapters import get_adapter, loaders
from trinity.adapters.base import TaskType
from trinity.adapters.split_policy import ToyFallbackWarning, resolve_split
from trinity.orchestration.reward import MATH_BENCHMARKS, score_text


def _fake_datasets(monkeypatch, handler):
    module = types.ModuleType("datasets")
    module.load_dataset = handler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def _no_datasets(monkeypatch):
    """Force every HF load to fail, so the toy fallback runs."""
    def boom(*a, **k):
        raise RuntimeError("offline")
    _fake_datasets(monkeypatch, boom)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_aime_is_registered_and_is_a_math_benchmark():
    a = get_adapter("aime")
    assert a.name == "aime"
    assert a.task_type() == TaskType.MATH
    assert "aime" in MATH_BENCHMARKS          # the grader already knew it


def test_aime_is_a_supported_loader_benchmark():
    assert "aime" in loaders.SUPPORTED_BENCHMARKS


# --------------------------------------------------------------------------- #
# split policy — the toy-fallback bug family (#35 / #50 / #95 / #196)
# --------------------------------------------------------------------------- #
def test_logical_test_reads_the_single_upstream_train_split():
    # AIME publishes only `train`; without the alias, logical `test` finds no rows and
    # the loader silently serves the toy set.
    assert resolve_split("aime", "test") == "train"
    assert resolve_split("aime", "train") == "train"


def test_train_and_test_are_disjoint(monkeypatch):
    rows = [{"problem": f"p{i}", "answer": str(i)} for i in range(40)]
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: rows)
    a = get_adapter("aime")
    train = {t.task_id for t in a.load_tasks("train", None, seed=0)}
    test = {t.task_id for t in a.load_tasks("test", None, seed=0)}
    assert train and test
    assert train.isdisjoint(test), "train/test overlap would leak eval rows into training"


def test_holdout_is_deterministic(monkeypatch):
    rows = [{"problem": f"p{i}", "answer": str(i)} for i in range(40)]
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: rows)
    a = get_adapter("aime")
    first = [t.task_id for t in a.load_tasks("test", None, seed=3)]
    second = [t.task_id for t in a.load_tasks("test", None, seed=3)]
    assert first == second


# --------------------------------------------------------------------------- #
# row parsing — both published schemas
# --------------------------------------------------------------------------- #
def test_parses_the_lowercase_schema(monkeypatch):
    rows = [{"problem": "Find N.", "answer": 204, "url": "u", "year": 2024}]
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: rows)
    tasks = loaders._load_aime_hf("train")
    assert tasks is not None and len(tasks) == 1
    t = tasks[0]
    assert t.benchmark == "aime" and t.prompt == "Find N." and t.answer == "204"
    assert t.meta["source"] == "AI-MO/aimo-validation-aime"


def test_parses_the_capitalised_fallback_schema(monkeypatch):
    def load_dataset(path, name=None, split=None):
        if path == "AI-MO/aimo-validation-aime":
            raise RuntimeError("unavailable")
        return [{"Problem": "Compute x.", "Answer": 42}]

    _fake_datasets(monkeypatch, load_dataset)
    tasks = loaders._load_aime_hf("train")
    assert tasks is not None and tasks[0].answer == "42"
    assert tasks[0].meta["source"] == "Maxwell-Jia/AIME_2024"


def test_rows_without_a_problem_or_answer_are_skipped(monkeypatch):
    rows = [
        {"problem": "", "answer": "1"},          # no problem
        {"problem": "ok", "answer": ""},         # blank answer
        {"problem": "keep", "answer": 7},        # good
    ]
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: rows)
    tasks = loaders._load_aime_hf("train")
    assert [t.answer for t in tasks] == ["7"]


def test_zero_padded_gold_is_stored_verbatim(monkeypatch):
    # math_equal treats a leading zero as significant for AIME; the loader must not
    # rewrite the gold, in either direction.
    rows = [{"problem": "p", "answer": "005"}]
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: rows)
    (task,) = loaders._load_aime_hf("train")
    assert task.answer == "005"
    assert score_text("aime", r"\boxed{005}", task.answer) == 1.0
    assert score_text("aime", r"\boxed{5}", task.answer) == 0.0     # padding is significant


def test_no_rows_returns_none_so_the_toy_fallback_runs(monkeypatch):
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: [])
    assert loaders._load_aime_hf("train") is None


# --------------------------------------------------------------------------- #
# offline toy fallback
# --------------------------------------------------------------------------- #
def test_toy_fallback_serves_aime_tasks_and_warns(monkeypatch):
    _no_datasets(monkeypatch)
    a = get_adapter("aime")
    with pytest.warns(ToyFallbackWarning):
        tasks = a.load_tasks("test", None, seed=0)
    assert tasks and all(t.benchmark == "aime" for t in tasks)
    assert all(t.meta["source"] == "toy" for t in tasks)


def test_toy_answers_grade_correct(monkeypatch):
    _no_datasets(monkeypatch)
    a = get_adapter("aime")
    with pytest.warns(ToyFallbackWarning):
        tasks = a.load_tasks("test", None, seed=0)
    for t in tasks:
        assert a.score_output(rf"the answer is \boxed{{{t.answer}}}", t.answer) == 1.0


# --------------------------------------------------------------------------- #
# scoring delegates to the shared math extractor (no new scoring behaviour)
# --------------------------------------------------------------------------- #
def test_score_output_matches_the_shared_math_scorer():
    a = get_adapter("aime")
    for out, ref in ((r"\boxed{204}", "204"), ("the answer is 204", "204"),
                     (r"\boxed{7}", "204"), ("", "204")):
        assert a.score_output(out, ref) == score_text("aime", out, ref)


def test_serialize_task_shape(monkeypatch):
    rows = [{"problem": "p", "answer": 5}]
    _fake_datasets(monkeypatch, lambda path, name=None, split=None: rows)
    a = get_adapter("aime")
    (task,) = loaders._load_aime_hf("train")
    d = a.serialize_task(task)
    assert d["benchmark"] == "aime" and d["task_type"] == TaskType.MATH.value
    assert d["reference"] == "5" and d["prompt"] == "p"
    assert set(d) == {"task_id", "benchmark", "prompt", "reference", "task_type", "meta"}
