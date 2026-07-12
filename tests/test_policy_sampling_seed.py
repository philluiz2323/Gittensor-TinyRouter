"""Tests for deterministic policy sampling RNG during training."""
from __future__ import annotations

from trinity.optim.sampling import trajectory_sampling_manual_seed


def test_trajectory_sampling_manual_seed_is_deterministic():
    kwargs = dict(run_seed=7, generation=3, candidate_idx=2, task_id="q1")
    assert trajectory_sampling_manual_seed(**kwargs) == trajectory_sampling_manual_seed(**kwargs)


def test_trajectory_sampling_manual_seed_varies_by_task_id():
    base = dict(run_seed=7, generation=0, candidate_idx=0)
    assert trajectory_sampling_manual_seed(**base, task_id="q1") != trajectory_sampling_manual_seed(
        **base, task_id="q2"
    )
