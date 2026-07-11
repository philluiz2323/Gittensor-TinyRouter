"""Branch coverage for the MMLU-Pro adapter's parsing and fallback paths.

`test_mmlu_pro_adapter.py` covers the happy path (toy load, formatting, scoring),
but several parsing branches were uncovered (mmlu_pro.py at 82%):

* `_row_get` on a non-mapping row,
* `_answer_letter`'s **answer_index fallback** (used when the explicit letter is
  absent/invalid) and its out-of-range / non-int guards,
* `_hf_mmlu_pro`'s load-error return and its per-row skips (missing
  question/options, unresolvable answer),
* the adapter's `load_tasks` / `build_prompt` delegation.

The HuggingFace path is exercised offline by injecting a fake `datasets` module
(`_hf_mmlu_pro` imports `load_dataset` lazily), exactly as the loaders tests do.
This module imports no torch.
"""
from __future__ import annotations

import sys
import types

from trinity.adapters import mmlu_pro as M
from trinity.adapters.base import TaskType


def _fake_datasets(monkeypatch, handler):
    module = types.ModuleType("datasets")
    module.load_dataset = handler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --------------------------------------------------------------------------- #
# _row_get / _answer_letter
# --------------------------------------------------------------------------- #
def test_row_get_non_mapping_row_returns_default():
    # `"k" in 123` raises TypeError -> the loop breaks -> default is returned.
    assert M._row_get(123, "k", default="d") == "d"


def test_answer_letter_prefers_explicit_letter():
    assert M._answer_letter({"answer": "c"}, 5) == "C"


def test_answer_letter_falls_back_to_index_when_letter_absent():
    assert M._answer_letter({"answer_index": 2}, 5) == "C"


def test_answer_letter_falls_back_to_index_when_letter_out_of_range():
    # 'Z' is a single letter but not among the first n_options -> use the index.
    assert M._answer_letter({"answer": "Z", "answer_index": 0}, 5) == "A"


def test_answer_letter_rejects_non_int_index():
    assert M._answer_letter({"answer_index": "not-int"}, 5) is None


def test_answer_letter_rejects_out_of_range_index():
    assert M._answer_letter({"answer_index": 9}, 5) is None


# --------------------------------------------------------------------------- #
# _hf_mmlu_pro
# --------------------------------------------------------------------------- #
def test_hf_returns_none_when_datasets_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)  # import raises
    assert M._hf_mmlu_pro("test") is None


def test_hf_returns_none_on_load_error(monkeypatch):
    def boom(path, split=None):
        raise RuntimeError("gated / offline")

    _fake_datasets(monkeypatch, boom)
    assert M._hf_mmlu_pro("test") is None


def test_hf_parses_valid_rows_and_skips_malformed(monkeypatch):
    rows = [
        {"question": "", "options": ["a", "b"], "answer": "A"},          # no question -> skip
        {"question": "q", "options": None, "answer": "A"},               # no options -> skip
        {"question": "has answer?", "options": ["x", "y", "z"], "answer": "Z"},  # bad answer -> skip
        {"question_id": "mp-7", "question": "pick", "options": ["p", "q", "r"],
         "answer_index": 1, "category": "logic"},                        # valid via index -> B
    ]

    def load_dataset(path, split=None):
        assert path == "TIGER-Lab/MMLU-Pro"
        return rows

    _fake_datasets(monkeypatch, load_dataset)
    tasks = M._hf_mmlu_pro("test")
    assert tasks is not None and len(tasks) == 1
    t = tasks[0]
    assert t.task_id == "mp-7"
    assert t.answer == "B"
    assert t.benchmark == "mmlu_pro"
    assert t.meta["n_options"] == 3
    assert t.meta["category"] == "logic"
    assert t.meta["source"] == "TIGER-Lab/MMLU-Pro"
    assert "A. p" in t.prompt and "C. r" in t.prompt


def test_hf_returns_none_when_all_rows_malformed(monkeypatch):
    _fake_datasets(monkeypatch, lambda path, split=None: [{"question": ""}])
    assert M._hf_mmlu_pro("test") is None


def test_hf_task_id_falls_back_to_index_when_no_question_id(monkeypatch):
    rows = [{"question": "q", "options": ["a", "b"], "answer": "A"}]
    _fake_datasets(monkeypatch, lambda path, split=None: rows)
    tasks = M._hf_mmlu_pro("test")
    assert tasks[0].task_id == "mmlu_pro-0"


# --------------------------------------------------------------------------- #
# adapter delegation
# --------------------------------------------------------------------------- #
def test_adapter_load_tasks_and_build_prompt_delegate():
    adapter = M.MmluProAdapter()
    # No fake datasets installed -> HF load fails -> deterministic toy fallback.
    tasks = adapter.load_tasks("test", max_items=1, seed=0)
    assert len(tasks) == 1
    task = tasks[0]
    assert adapter.build_prompt(task) == task.prompt
    assert adapter.task_type() is TaskType.MCQ


def test_adapter_score_output_scores_correct_letter():
    adapter = M.MmluProAdapter()
    assert adapter.score_output("The answer is B", "B") == 1.0
    assert adapter.score_output("The answer is A", "B") == 0.0
