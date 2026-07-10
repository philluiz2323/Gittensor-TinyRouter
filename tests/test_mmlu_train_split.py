"""Tests for MMLU train-split resolution (issue #35)."""
from __future__ import annotations

import sys
import types
import warnings

from trinity.adapters.loaders import load_split
from trinity.adapters.split_policy import ToyFallbackWarning, resolve_split
from trinity.orchestration.dataset import load_tasks


def _install_fake_mmlu_datasets() -> None:
    """Simulate cais/mmlu with the real upstream split set."""
    real_splits = {"auxiliary_train", "test", "validation", "dev"}

    def load_dataset(path, name=None, split=None, **kwargs):
        if path != "cais/mmlu" or split not in real_splits:
            raise ValueError(f"Unknown split {split!r}.")
        return [
            {
                "question": f"q-{split}-{i}",
                "choices": ["A", "B", "C", "D"],
                "answer": 0,
            }
            for i in range(50)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


def test_resolve_split_maps_train_to_auxiliary_train():
    assert resolve_split("mmlu", "train") == "auxiliary_train"
    assert resolve_split("mmlu", "test") == "test"


def test_train_loads_auxiliary_train_not_toy_set():
    _install_fake_mmlu_datasets()
    tasks = load_split("mmlu", "train", max_items=5, seed=0)
    assert len(tasks) == 5
    assert all(not t.task_id.startswith("mmlu-toy-") for t in tasks)
    assert tasks[0].meta["source"] == "cais/mmlu"


def test_train_and_test_load_different_upstream_splits():
    _install_fake_mmlu_datasets()
    train_prompts = {t.prompt for t in load_split("mmlu", "train", max_items=None, seed=0)}
    test_prompts = {t.prompt for t in load_split("mmlu", "test", max_items=None, seed=0)}
    assert any("auxiliary_train" in p for p in train_prompts)
    assert any("test" in p for p in test_prompts)
    assert train_prompts.isdisjoint(test_prompts)


def test_load_tasks_shim_uses_same_resolution():
    _install_fake_mmlu_datasets()
    tasks = load_tasks("mmlu", "train", max_items=3, seed=0)
    assert len(tasks) == 3
    assert tasks[0].benchmark == "mmlu"


def test_toy_fallback_emits_warning_when_hf_missing(monkeypatch):
    monkeypatch.setattr("trinity.adapters.loaders._try_load_hf", lambda *a, **k: None)
    sys.modules.pop("datasets", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ToyFallbackWarning)
        tasks = load_split("mmlu", "train", max_items=None, seed=0)
    assert len(tasks) == 2
    assert any(isinstance(w.message, ToyFallbackWarning) for w in caught)
