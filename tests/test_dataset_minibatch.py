"""Offline unit tests for CMA-ES minibatch sampling (``dataset.sample_minibatch``).

``sample_minibatch`` is on the training hot path (``train.py``) but had no
dedicated pytest coverage. These tests pin SPEC §5.2 sampling semantics:
without-replacement when the pool is large enough, with-replacement on toy sets,
and validation of empty/invalid inputs.
"""
from __future__ import annotations

import random

import pytest

from trinity.orchestration.dataset import sample_minibatch
from trinity.types import Task


def _tasks(n: int) -> list[Task]:
    return [
        Task(task_id=f"t{i}", benchmark="math500", prompt=f"q{i}", answer=str(i))
        for i in range(n)
    ]


def test_sample_minibatch_rejects_empty_pool():
    with pytest.raises(ValueError, match="empty task list"):
        sample_minibatch([], 1, random.Random(0))


def test_sample_minibatch_rejects_non_positive_m():
    pool = _tasks(3)
    with pytest.raises(ValueError, match="must be positive"):
        sample_minibatch(pool, 0, random.Random(0))


def test_sample_minibatch_without_replacement_when_pool_large_enough():
    pool = _tasks(10)
    batch = sample_minibatch(pool, 5, random.Random(42))
    assert len(batch) == 5
    assert len({t.task_id for t in batch}) == 5


def test_sample_minibatch_with_replacement_on_toy_set():
    pool = _tasks(2)
    batch = sample_minibatch(pool, 5, random.Random(7))
    assert len(batch) == 5
    assert all(t in pool for t in batch)
    # With m > len(pool), duplicates are expected.
    assert len({t.task_id for t in batch}) < 5


def test_sample_minibatch_is_deterministic_for_fixed_rng():
    pool = _tasks(8)
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    assert sample_minibatch(pool, 4, rng_a) == sample_minibatch(pool, 4, rng_b)
