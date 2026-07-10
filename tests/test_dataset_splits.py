"""Logical-split resolution and the toy-set fallback contract.

Regression cover for the bug where ``load_tasks("mmlu", "train")`` silently
returned the 2-item toy set: ``cais/mmlu`` has no ``train`` split, the resulting
``load_dataset`` error was swallowed, and the toy set stood in for real data with
no diagnostic.

Every test here fakes the ``datasets`` module, so the *online* code path is
exercised without a network. ``_try_load_hf`` imports ``datasets`` lazily, so
injecting ``sys.modules["datasets"]`` is enough to intercept it.
"""
from __future__ import annotations

import contextlib
import sys
import types
import warnings

import pytest

from trinity.orchestration.dataset import (
    ToyFallbackWarning,
    _resolve_split,
    load_tasks,
)

# Split sets the real HuggingFace datasets actually expose, keyed by (path, config).
_REAL_SPLITS: dict[tuple[str, str | None], set[str]] = {
    ("cais/mmlu", "all"): {"test", "validation", "dev", "auxiliary_train"},
}


def _install_fake_datasets(monkeypatch, requested: list[dict[str, object]]):
    """Install a fake ``datasets`` module mimicking real upstream split sets.

    Records every ``load_dataset`` call into ``requested`` and raises for any
    split the real dataset does not have -- exactly as HuggingFace does.
    """

    def load_dataset(path, name=None, split=None):
        requested.append({"path": path, "name": name, "split": split})
        available = _REAL_SPLITS.get((path, name))
        if available is None:
            raise ValueError(f"Dataset {path!r} (config {name!r}) is not available.")
        if split not in available:
            raise ValueError(
                f"Unknown split {split!r}. Should be one of {sorted(available)}."
            )
        return [
            {
                "question": f"real question {i} from {split}",
                "choices": ["alpha", "beta", "gamma", "delta"],
                "answer": i % 4,
                "subject": "astronomy",
            }
            for i in range(6)
        ]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return requested


def _install_missing_datasets(monkeypatch):
    """Make ``import datasets`` fail, simulating the offline dev box."""
    monkeypatch.setitem(sys.modules, "datasets", None)


@contextlib.contextmanager
def _warnings_as_list():
    """Capture warnings without asserting on them (``pytest.warns`` requires one)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


# --------------------------------------------------------------------------- #
# _resolve_split
# --------------------------------------------------------------------------- #
def test_resolve_split_maps_mmlu_train_to_auxiliary_train():
    assert _resolve_split("mmlu", "train") == "auxiliary_train"


def test_resolve_split_leaves_mmlu_test_alone():
    assert _resolve_split("mmlu", "test") == "test"


def test_resolve_split_is_case_and_whitespace_insensitive():
    assert _resolve_split("mmlu", "  TRAIN ") == "auxiliary_train"


@pytest.mark.parametrize("benchmark", ["math500", "gpqa", "livecodebench"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_resolve_split_is_identity_for_unaliased_benchmarks(benchmark, split):
    """Only MMLU needs aliasing today; nothing else may be rewritten."""
    assert _resolve_split(benchmark, split) == split


# --------------------------------------------------------------------------- #
# The bug: mmlu/train must reach real data, not the toy set
# --------------------------------------------------------------------------- #
def test_mmlu_train_requests_auxiliary_train_split(monkeypatch):
    requested: list[dict[str, object]] = []
    _install_fake_datasets(monkeypatch, requested)

    load_tasks("mmlu", "train", max_items=None, seed=0)

    assert requested, "expected a load_dataset call"
    assert requested[0]["path"] == "cais/mmlu"
    assert requested[0]["name"] == "all"
    assert requested[0]["split"] == "auxiliary_train"


def test_mmlu_train_returns_real_tasks_not_toy(monkeypatch):
    """The regression: previously these were the two `mmlu-toy-*` tasks."""
    _install_fake_datasets(monkeypatch, [])

    tasks = load_tasks("mmlu", "train", max_items=None, seed=0)

    assert len(tasks) == 6
    assert not any(t.task_id.startswith("mmlu-toy") for t in tasks)
    assert all(t.benchmark == "mmlu" for t in tasks)
    assert all(t.answer in ("A", "B", "C", "D") for t in tasks)


def test_mmlu_train_does_not_warn_when_real_data_loads(monkeypatch):
    _install_fake_datasets(monkeypatch, [])

    with _warnings_as_list() as caught:
        load_tasks("mmlu", "train", max_items=None, seed=0)

    assert not [w for w in caught if issubclass(w.category, ToyFallbackWarning)]


def test_mmlu_test_still_uses_the_test_split(monkeypatch):
    """Eval already worked; the alias table must not disturb it."""
    requested: list[dict[str, object]] = []
    _install_fake_datasets(monkeypatch, requested)

    tasks = load_tasks("mmlu", "test", max_items=None, seed=0)

    assert requested[0]["split"] == "test"
    assert not any(t.task_id.startswith("mmlu-toy") for t in tasks)


def test_mmlu_train_and_test_draw_from_disjoint_splits(monkeypatch):
    """auxiliary_train vs test -- the held-out set is genuinely held out."""
    requested: list[dict[str, object]] = []
    _install_fake_datasets(monkeypatch, requested)

    load_tasks("mmlu", "train", max_items=None, seed=0)
    load_tasks("mmlu", "test", max_items=None, seed=0)

    splits = [call["split"] for call in requested]
    assert splits == ["auxiliary_train", "test"]


# --------------------------------------------------------------------------- #
# The toy fallback must be loud, and refusable
# --------------------------------------------------------------------------- #
def test_toy_fallback_warns_when_datasets_is_unavailable(monkeypatch):
    _install_missing_datasets(monkeypatch)

    with pytest.warns(ToyFallbackWarning, match="toy set"):
        tasks = load_tasks("mmlu", "train", max_items=None, seed=0)

    assert all(t.task_id.startswith("mmlu-toy") for t in tasks)


def test_toy_fallback_warning_names_benchmark_and_split(monkeypatch):
    _install_missing_datasets(monkeypatch)

    with pytest.warns(ToyFallbackWarning) as record:
        load_tasks("gpqa", "test", max_items=None, seed=0)

    message = str(record[0].message)
    assert "'gpqa'" in message
    assert "'test'" in message


def test_strict_mode_raises_instead_of_falling_back(monkeypatch):
    _install_missing_datasets(monkeypatch)

    with pytest.raises(RuntimeError, match="allow_toy_fallback=False"):
        load_tasks("mmlu", "train", max_items=None, seed=0, allow_toy_fallback=False)


def test_strict_mode_is_a_noop_when_real_data_loads(monkeypatch):
    _install_fake_datasets(monkeypatch, [])

    tasks = load_tasks("mmlu", "train", max_items=None, seed=0, allow_toy_fallback=False)

    assert len(tasks) == 6


def test_offline_toy_fallback_still_serves_smoke_tests(monkeypatch):
    """The fallback stays functional -- it just stopped being silent."""
    _install_missing_datasets(monkeypatch)

    with pytest.warns(ToyFallbackWarning):
        tasks = load_tasks("math500", "test", max_items=2, seed=0)

    assert len(tasks) == 2
    assert all(t.benchmark == "math500" for t in tasks)
