"""Tests for GPQA holdout-split resolution (issue #95)."""
from __future__ import annotations

import sys
import types
import warnings

from trinity.adapters.loaders import load_split
from trinity.adapters.split_policy import (
    ToyFallbackWarning,
    holdout_indices,
    resolve_split,
    select_holdout,
)
from trinity.orchestration.dataset import load_tasks

N_ROWS = 40


def _install_fake_gpqa_datasets() -> None:
    """Simulate Idavidrein/gpqa, which publishes only a ``train`` split."""

    def load_dataset(path, name=None, split=None, **kwargs):
        if path != "Idavidrein/gpqa" or name != "gpqa_diamond" or split != "train":
            raise ValueError(f"Unknown split {split!r}.")
        return [
            {
                "Question": f"q-{i}",
                "Correct Answer": f"right-{i}",
                "Incorrect Answer 1": f"wrong-{i}-a",
                "Incorrect Answer 2": f"wrong-{i}-b",
                "Incorrect Answer 3": f"wrong-{i}-c",
            }
            for i in range(N_ROWS)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


def test_resolve_split_maps_both_logical_splits_to_train():
    assert resolve_split("gpqa", "train") == "train"
    assert resolve_split("gpqa", "test") == "train"


def test_resolve_split_leaves_other_benchmarks_alone():
    assert resolve_split("math500", "test") == "test"
    assert resolve_split("mmlu", "train") == "auxiliary_train"


def test_test_split_loads_real_rows_not_toy_set():
    _install_fake_gpqa_datasets()
    tasks = load_split("gpqa", "test", max_items=None, seed=0)
    assert tasks
    assert all(not t.task_id.startswith("gpqa-toy-") for t in tasks)
    assert tasks[0].meta["source"] == "Idavidrein/gpqa"


def test_train_and_test_are_disjoint_and_cover_the_upstream_split():
    _install_fake_gpqa_datasets()
    train = {t.task_id for t in load_split("gpqa", "train", max_items=None, seed=0)}
    test = {t.task_id for t in load_split("gpqa", "test", max_items=None, seed=0)}
    assert train and test
    assert train.isdisjoint(test)
    assert len(train | test) == N_ROWS


def test_partition_is_independent_of_the_shuffle_seed():
    _install_fake_gpqa_datasets()
    a = {t.task_id for t in load_split("gpqa", "test", max_items=None, seed=0)}
    b = {t.task_id for t in load_split("gpqa", "test", max_items=None, seed=123)}
    assert a == b


def test_holdout_indices_are_deterministic_and_proportional():
    first = holdout_indices("gpqa", N_ROWS)
    assert first == holdout_indices("gpqa", N_ROWS)
    assert len(first) == round(N_ROWS * 0.25)
    assert all(0 <= i < N_ROWS for i in first)


def test_holdout_never_empties_either_side():
    for n in (2, 3, 5):
        held = holdout_indices("gpqa", n)
        assert 1 <= len(held) <= n - 1


def test_benchmarks_without_a_holdout_pass_through():
    items = list(range(10))
    assert holdout_indices("math500", 10) == frozenset()
    assert select_holdout("math500", "test", items) == items
    assert select_holdout("math500", "train", items) == items


def test_load_tasks_shim_uses_same_resolution():
    _install_fake_gpqa_datasets()
    tasks = load_tasks("gpqa", "test", max_items=3, seed=0)
    assert len(tasks) == 3
    assert tasks[0].benchmark == "gpqa"


def test_toy_fallback_still_warns_when_hf_missing(monkeypatch):
    monkeypatch.setattr("trinity.adapters.loaders._try_load_hf", lambda *a, **k: None)
    sys.modules.pop("datasets", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ToyFallbackWarning)
        tasks = load_split("gpqa", "test", max_items=None, seed=0)
    assert len(tasks) == 2
    assert any(isinstance(w.message, ToyFallbackWarning) for w in caught)
