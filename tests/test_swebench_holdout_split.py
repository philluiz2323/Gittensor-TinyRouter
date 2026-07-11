"""SWE-bench Verified holdout-split resolution.

SWE-bench Verified publishes a *single* upstream split (``test``, 500 instances).
Loading the logical ``train`` split previously asked HuggingFace for a
non-existent ``train`` split, which raised and silently fell back to the one-item
toy set (real data discarded, no warning). This mirrors the GPQA single-split
fix (issue #95): ``train`` resolves to the upstream ``test`` split and the two
logical splits are carved into disjoint subsets by ``select_holdout``.

Offline — installs a fake ``datasets`` module; no network, no GPU.
"""
from __future__ import annotations

import sys
import types
import warnings

import trinity.adapters.swebench as sb
from trinity.adapters.split_policy import ToyFallbackWarning, resolve_split, select_holdout

N_ROWS = 40


def _install_fake_swebench_datasets() -> None:
    """Simulate princeton-nlp/SWE-bench_Verified, which publishes only ``test``."""

    def load_dataset(path, name=None, split=None, **kwargs):
        if path != sb._HF_DATASET or split != "test":
            raise ValueError(f"Unknown split {split!r}.")
        return [
            {
                "instance_id": f"inst-{i}",
                "problem_statement": f"issue-{i}",
                "repo": f"org/repo-{i}",
                "base_commit": "0" * 40,
                "patch": "diff --git a/x b/x\n",
            }
            for i in range(N_ROWS)
        ]

    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


def test_resolve_split_maps_train_to_the_upstream_test_split():
    assert resolve_split("swebench_verified", "train") == "test"
    assert resolve_split("swebench_verified", "training") == "test"
    # The logical test split passes through unchanged.
    assert resolve_split("swebench_verified", "test") == "test"


def test_train_split_loads_real_rows_not_the_toy_set():
    _install_fake_swebench_datasets()
    try:
        tasks = sb.load_swebench_tasks("train", max_items=None, seed=0)
    finally:
        sys.modules.pop("datasets", None)
    assert tasks
    assert all(t.task_id != "octo__calc-1" for t in tasks), "train fell back to toy set"
    assert all(t.meta["source"] == sb._HF_DATASET for t in tasks)


def test_train_and_test_are_disjoint_and_cover_the_upstream_split():
    _install_fake_swebench_datasets()
    try:
        train = sb.load_swebench_tasks("train", max_items=None, seed=0)
        test = sb.load_swebench_tasks("test", max_items=None, seed=0)
    finally:
        sys.modules.pop("datasets", None)
    train_ids = {t.task_id for t in train}
    test_ids = {t.task_id for t in test}
    assert train_ids and test_ids
    assert train_ids.isdisjoint(test_ids), "train/test leakage"
    assert len(train_ids) + len(test_ids) == N_ROWS, "splits must cover the upstream set"
    # The holdout is the ~25% test slice; train keeps the larger remainder.
    assert len(test_ids) < len(train_ids)


def test_select_holdout_is_deterministic_across_calls():
    items = list(range(N_ROWS))
    a = select_holdout("swebench_verified", "test", items)
    b = select_holdout("swebench_verified", "test", items)
    assert a == b
    # test-half and train-half partition the input.
    train = select_holdout("swebench_verified", "train", items)
    assert sorted(a + train) == items


def test_toy_fallback_still_warns_when_hf_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tasks = sb.load_swebench_tasks("train", max_items=None, seed=0)
    assert len(tasks) == 1  # the toy task
    assert any(issubclass(w.category, ToyFallbackWarning) for w in caught)
