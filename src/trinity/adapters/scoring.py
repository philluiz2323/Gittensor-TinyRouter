"""One top-level scoring API over cached and execution benchmarks (issue #16).

The leaderboard / evaluator should grade a submission without knowing whether a
benchmark is a cheap cached compare (MMLU, math) or an expensive live run
(SWE-bench). This module is that seam: :func:`score_item` takes any
:class:`~trinity.adapters.base.BenchmarkAdapter` and routes to the right path
based on the adapter's declared :meth:`~trinity.adapters.base.BenchmarkAdapter.scoring_modes`,
so benchmark-specific logic never leaks into the caller.

Routing rule:

* If the adapter supports :data:`ScoringMode.EXECUTION` **and** an
  ``execution_context`` is supplied, try :meth:`score_execution`; use its result
  unless it is ``None`` ("could not execute").
* Otherwise (or on an execution miss) fall back to :meth:`score_cached`, the
  cheap path every adapter supports by default.
"""
from __future__ import annotations

from typing import Any

from .base import BenchmarkAdapter, ScoringMode

__all__ = ["ScoringOutcome", "supports_execution", "requires_execution", "score_item"]


class ScoringOutcome:
    """Result of :func:`score_item`: the reward plus which path produced it.

    ``mode`` is the :class:`ScoringMode` actually used, so the caller can report
    or meter cheap vs expensive scoring without re-deriving it.
    """

    __slots__ = ("reward", "mode")

    def __init__(self, reward: float, mode: ScoringMode):
        self.reward = reward
        self.mode = mode

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ScoringOutcome(reward={self.reward}, mode={self.mode.value})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScoringOutcome)
            and other.reward == self.reward
            and other.mode == self.mode
        )


def supports_execution(adapter: BenchmarkAdapter) -> bool:
    """Whether ``adapter`` declares the expensive :data:`ScoringMode.EXECUTION` path."""
    return ScoringMode.EXECUTION in adapter.scoring_modes()


def requires_execution(adapter: BenchmarkAdapter) -> bool:
    """Whether ``adapter`` can *only* be scored by execution (no cached path).

    Lets a pipeline skip such benchmarks in a cheap-only round instead of
    silently mis-scoring them through the fallback.
    """
    modes = adapter.scoring_modes()
    return ScoringMode.EXECUTION in modes and ScoringMode.CACHED not in modes


def score_item(
    adapter: BenchmarkAdapter,
    output: str,
    reference: Any,
    *,
    execution_context: Any = None,
) -> ScoringOutcome:
    """Grade ``output`` against ``reference`` via ``adapter``'s best available path.

    The single entry point the evaluator uses for every benchmark. Prefers the
    execution path when the adapter supports it and an ``execution_context`` is
    given; otherwise (or if execution returns ``None``) uses the cached path.

    Args:
        adapter: The benchmark's adapter.
        output: The model's answer (a letter, boxed value, code, or unified diff).
        reference: The task's reference (``Task.answer``).
        execution_context: Opaque payload handed to :meth:`score_execution` (e.g.
            a prepared work-tree or an executor). ``None`` forces the cached path.

    Returns:
        A :class:`ScoringOutcome` with the binary reward and the mode used.

    Raises:
        ValueError: If the adapter declares no scoring modes at all.
    """
    modes = adapter.scoring_modes()
    if not modes:
        raise ValueError(f"adapter {adapter.name!r} declares no scoring modes")

    if execution_context is not None and ScoringMode.EXECUTION in modes:
        reward = adapter.score_execution(output, reference, context=execution_context)
        if reward is not None:
            return ScoringOutcome(float(reward), ScoringMode.EXECUTION)

    if ScoringMode.CACHED in modes:
        return ScoringOutcome(adapter.score_cached(output, reference), ScoringMode.CACHED)

    # Execution-only adapter with no usable context: fall back to its own
    # execution scorer without a context (may still return a value) rather than
    # silently reporting a cached score it does not support.
    reward = adapter.score_execution(output, reference, context=execution_context)
    return ScoringOutcome(float(reward or 0.0), ScoringMode.EXECUTION)
