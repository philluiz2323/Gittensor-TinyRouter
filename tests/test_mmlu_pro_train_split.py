"""Tests for MMLU-Pro train-split resolution (issue #50)."""
from __future__ import annotations

import sys
import types
import warnings

import pytest

from trinity.adapters.mmlu_pro import BENCHMARK, load_mmlu_pro_tasks
from trinity.adapters.split_policy import ToyFallbackWarning, resolve_split


def _install_fake_mmlu_pro_datasets() -> None:
    """Simulate TIGER-Lab/MMLU-Pro with only validation and test splits."""
    real_splits = {"TIGER-Lab/MMLU-Pro": {"validation", "test"}}

    def load_dataset(path, name=None, split=None, **kwargs):
        if split not in real_splits[path]:
            raise ValueError(
                f"Unknown split {split!r}. Should be one of {sorted(real_splits[path])}."
            )
        return [
            {
                "question": f"q-{split}-{i}",
                "options": [f"opt{j}" for j in range(10)],
                "answer": "C",
                "answer_index": 2,
                "category": "math",
                "question_id": f"{split}-{i}",
            }
            for i in range(50)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


def test_resolve_split_maps_train_to_validation():
    assert resolve_split("mmlu_pro", "train") == "validation"
    assert resolve_split("mmlu_pro", "test") == "test"


def test_train_loads_validation_not_toy_set():
    _install_fake_mmlu_pro_datasets()
    tasks = load_mmlu_pro_tasks("train", max_items=5, seed=0)
    assert len(tasks) == 5
    assert all(not t.task_id.startswith("mmlu_pro-toy-") for t in tasks)
    assert tasks[0].meta["source"] == "TIGER-Lab/MMLU-Pro"


def test_test_split_loads_real_rows():
    _install_fake_mmlu_pro_datasets()
    tasks = load_mmlu_pro_tasks("test", max_items=3, seed=0)
    assert len(tasks) == 3
    assert tasks[0].task_id.startswith("test-")


def test_toy_fallback_emits_warning_when_hf_missing(monkeypatch):
    monkeypatch.setattr("trinity.adapters.mmlu_pro._hf_mmlu_pro", lambda _split: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ToyFallbackWarning)
        tasks = load_mmlu_pro_tasks("train", max_items=None, seed=0)
    assert len(tasks) == 2
    assert any(isinstance(w.message, ToyFallbackWarning) for w in caught)
