"""Shared split-resolution helpers for benchmark loaders."""
from __future__ import annotations

import random
import warnings
from typing import Sequence, TypeVar

__all__ = [
    "HOLDOUT_SEED",
    "ToyFallbackWarning",
    "holdout_indices",
    "resolve_split",
    "select_holdout",
    "warn_on_toy_fallback",
]

T = TypeVar("T")

#: Logical split names that do not exist upstream, mapped per benchmark.
#: MMLU-Pro publishes only ``validation`` and ``test``; training callers use
#: ``train``, which we map to ``validation`` (issue #50).
#: MMLU publishes ``auxiliary_train`` for training pool; logical ``train`` maps
#: there (issue #35).
#: GPQA-Diamond publishes only ``train``; both logical splits read it and are then
#: carved into disjoint subsets by :func:`select_holdout` (issue #95).
_SPLIT_ALIASES: dict[str, dict[str, str]] = {
    "mmlu": {
        "train": "auxiliary_train",
        "training": "auxiliary_train",
    },
    "mmlu_pro": {
        "train": "validation",
        "training": "validation",
    },
    "gpqa": {
        "train": "train",
        "training": "train",
        "test": "train",
    },
}

#: Seed for the deterministic holdout partition. Fixed: changing it silently
#: redefines the train/test boundary of every benchmark listed below.
HOLDOUT_SEED: int = 20260710

#: Fraction of upstream rows reserved for the logical ``test`` split. Only
#: benchmarks that publish a single upstream split need an entry.
_HOLDOUT_FRACTION: dict[str, float] = {
    "gpqa": 0.25,
}

#: Logical split names served the held-out rows; every other name gets the rest.
_TEST_SPLITS: frozenset[str] = frozenset({"test", "eval", "validation"})


class ToyFallbackWarning(UserWarning):
    """A loader substituted the offline toy set for real benchmark data."""


def resolve_split(benchmark: str, split: str) -> str:
    """Map a logical split to the upstream dataset split name.

    Benchmarks absent from :data:`_SPLIT_ALIASES` pass ``split`` through unchanged.
    """
    key = (benchmark or "").strip().lower()
    logical = (split or "test").strip().lower()
    return _SPLIT_ALIASES.get(key, {}).get(logical, logical)


def holdout_indices(benchmark: str, n: int) -> frozenset[int]:
    """Return the row positions reserved for the logical ``test`` split.

    The partition is a pure function of ``benchmark``, ``n``, and
    :data:`HOLDOUT_SEED`, so repeated calls — and separate processes — agree on
    which rows are held out. Positions index the upstream row order as loaded.

    Args:
        benchmark: Benchmark name; case- and whitespace-insensitive.
        n: Number of upstream rows available.

    Returns:
        The held-out positions, or an empty set for benchmarks that publish
        their own ``test`` split (and when ``n`` is too small to divide).
    """
    key = (benchmark or "").strip().lower()
    fraction = _HOLDOUT_FRACTION.get(key)
    if fraction is None or n < 2:
        return frozenset()
    k = max(1, min(n - 1, round(n * fraction)))
    rng = random.Random(f"{key}:{HOLDOUT_SEED}")
    return frozenset(rng.sample(range(n), k))


def select_holdout(benchmark: str, split: str, items: Sequence[T]) -> list[T]:
    """Carve ``items`` into the logical ``split`` half of a single-split benchmark.

    Benchmarks with no holdout configured pass ``items`` through unchanged, so
    this is safe to call for every benchmark.

    Args:
        benchmark: Benchmark name; case- and whitespace-insensitive.
        split: Logical split name, e.g. ``"train"`` or ``"test"``.
        items: Upstream rows in their loaded order.

    Returns:
        The held-out rows for a test-like ``split``, otherwise the remainder.
        The two are disjoint and together cover ``items``.
    """
    held = holdout_indices(benchmark, len(items))
    if not held:
        return list(items)
    logical = (split or "test").strip().lower()
    want_holdout = logical in _TEST_SPLITS
    return [x for i, x in enumerate(items) if (i in held) == want_holdout]


def warn_on_toy_fallback(benchmark: str, split: str, *, used_toy: bool) -> None:
    """Emit :class:`ToyFallbackWarning` when the toy set stands in for real data."""
    if not used_toy:
        return
    warnings.warn(
        (
            f"{benchmark!r} split {split!r} fell back to the offline toy set. "
            "Training or protocol builds on this split will not use real benchmark "
            "data unless HuggingFace loading succeeds."
        ),
        ToyFallbackWarning,
        stacklevel=3,
    )
