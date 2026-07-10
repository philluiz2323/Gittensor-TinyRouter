"""Shared split-resolution helpers for benchmark loaders."""
from __future__ import annotations

import warnings

__all__ = [
    "ToyFallbackWarning",
    "resolve_split",
    "warn_on_toy_fallback",
]

#: Logical split names that do not exist upstream, mapped per benchmark.
#: MMLU-Pro publishes only ``validation`` and ``test``; training callers use
#: ``train``, which we map to ``validation`` (issue #50).
#: MMLU publishes ``auxiliary_train`` for training pool; logical ``train`` maps
#: there (issue #35).
_SPLIT_ALIASES: dict[str, dict[str, str]] = {
    "mmlu": {
        "train": "auxiliary_train",
        "training": "auxiliary_train",
    },
    "mmlu_pro": {
        "train": "validation",
        "training": "validation",
    },
}


class ToyFallbackWarning(UserWarning):
    """A loader substituted the offline toy set for real benchmark data."""


def resolve_split(benchmark: str, split: str) -> str:
    """Map a logical split to the upstream dataset split name.

    Benchmarks absent from :data:`_SPLIT_ALIASES` pass ``split`` through unchanged.
    """
    key = (benchmark or "").strip().lower()
    logical = (split or "test").strip().lower()
    return _SPLIT_ALIASES.get(key, {}).get(logical, logical)


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
