"""Offline tests for honest Conductor evaluation aggregation (no network)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from trinity.fugu import eval as E
from trinity.types import Task


@dataclass
class _Run:
    """Minimal stand-in for a WorkflowRun that CostMeter.add_run can price."""

    parsed_ok: bool = True
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_tokens: dict = field(default_factory=dict)


def _evaluate_with_votes(votes: list[int]) -> E.EvalResult:
    """Run evaluate() over one task with a scripted per-rep correctness sequence.

    Stubs out propose_and_run (no network) and is_correct (returns the scripted
    votes) so only the aggregation logic under test is exercised.
    """
    seq = iter(votes)
    orig_propose, orig_is_correct = E.propose_and_run, E.is_correct

    async def _fake_propose_and_run(*_a, **_k):
        return _Run()

    def _fake_is_correct(_run, _task):
        return next(seq)

    E.propose_and_run = _fake_propose_and_run
    E.is_correct = _fake_is_correct
    try:
        task = Task(task_id="q", benchmark="math500", prompt="", answer="4")
        return asyncio.run(E.evaluate(None, [task], None, [], reps=len(votes)))
    finally:
        E.propose_and_run = orig_propose
        E.is_correct = orig_is_correct


def test_per_query_binary_is_strict_majority_not_a_tie():
    # A 50/50 split (1 of 2 correct) is NOT a majority -> the query is unsolved.
    res = _evaluate_with_votes([1, 0])
    assert res.per_query_binary["q"] == 0
    # The averaged accuracy still reflects the raw 0.5, only the binary changes.
    assert res.per_task["q"]["acc"] == 0.5


def test_per_query_binary_true_majority_and_single_rep():
    # A genuine majority (2 of 3) is solved; a minority (1 of 3) is not.
    assert _evaluate_with_votes([1, 1, 0]).per_query_binary["q"] == 1
    assert _evaluate_with_votes([1, 0, 0]).per_query_binary["q"] == 0
    # Single-rep behaviour is unchanged: correct -> 1, wrong -> 0.
    assert _evaluate_with_votes([1]).per_query_binary["q"] == 1
    assert _evaluate_with_votes([0]).per_query_binary["q"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
