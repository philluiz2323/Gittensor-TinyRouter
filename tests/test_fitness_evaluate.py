"""Offline coverage for the async fitness drivers `evaluate_candidate` /
`evaluate_population` in `trinity.optim.fitness`.

The pure shaping cores (`shaped_reward`, `variance_reweight`, `hero_*`) are covered
by `test_shaped_fitness.py` / `test_hero_dense.py`, but the two async functions that
actually turn a θ into a scalar fitness — configure the policy, run the minibatch,
assemble per-task rewards, tolerate failed trajectories, apply the HERO in-bucket
bonus, and (population-level) variance-reweight across candidates — were untested.

`evaluate_candidate` is driven with a fake policy/pool and a monkeypatched
`run_trajectory` returning hand-built trajectories that the *real* `reward` scorer
grades. `evaluate_population` is driven with a fake `evaluate_candidate` so the
population-level reweight / callback logic is isolated. No GPU, no network, no torch.
"""
from __future__ import annotations

import asyncio
import sys

import numpy as np

from trinity.optim import fitness as F
from trinity.optim.fitness import FitnessConfig
from trinity.types import Role, Task, Trajectory, TurnRecord

_BINARY = FitnessConfig(format_bonus=0.0, turn_penalty=0.0)  # plain binary, no shaping


def test_no_torch_imported():
    assert "torch" not in sys.modules, "fitness driver tests must not import torch"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _task(tid, answer):
    return Task(task_id=tid, benchmark="math500", prompt="q", answer=answer)


def _traj(task, *turn_texts, final=None):
    turns = [
        TurnRecord(turn=i + 1, agent_name="w", role=Role.WORKER,
                   raw_output=x, processed_output=x)
        for i, x in enumerate(turn_texts)
    ]
    return Trajectory(task=task, turns=turns, final_answer=final if final is not None else turn_texts[-1])


class _Policy:
    def __init__(self):
        self.configured = []

    def configure(self, theta, spec):
        self.configured.append((theta, spec))


def _install_fake_rt(monkeypatch, by_id, *, fail_ids=()):
    async def fake_run_trajectory(task, policy, pool, pool_models, *,
                                  sample=True, client=None, max_turns=5, **kw):
        if task.task_id in fail_ids:
            raise RuntimeError("retry-exhausted")
        return by_id[task.task_id]

    monkeypatch.setattr("trinity.optim.fitness.run_trajectory", fake_run_trajectory)


# --------------------------------------------------------------------------- #
# evaluate_candidate
# --------------------------------------------------------------------------- #
def test_evaluate_candidate_binary_mean_and_configures_policy(monkeypatch):
    t0, t1 = _task("0", "42"), _task("1", "5")
    _install_fake_rt(monkeypatch, {
        "0": _traj(t0, r"\boxed{42}"),   # correct
        "1": _traj(t1, r"\boxed{7}"),    # wrong
    })
    policy = _Policy()
    theta, spec = object(), object()
    fit, trajs, per_task = asyncio.run(F.evaluate_candidate(
        theta, spec, policy, object(), ["w"], [t0, t1],
        fitness_cfg=_BINARY, return_per_task=True,
    ))
    assert list(per_task) == [1.0, 0.0]
    assert fit == 0.5
    assert trajs == []                      # return_trajectories defaults False
    assert policy.configured == [(theta, spec)]


def test_evaluate_candidate_none_cfg_is_plain_binary(monkeypatch):
    # cfg=None must zero the bonuses (NOT take FitnessConfig()'s 0.05 shaping
    # defaults), so a correct multi-turn answer scores exactly 1.0, no format bonus.
    t0 = _task("0", "42")
    _install_fake_rt(monkeypatch, {"0": _traj(t0, "thinking", r"\boxed{42}")})
    _fit, _tr, per_task = asyncio.run(F.evaluate_candidate(
        object(), object(), _Policy(), object(), ["w"], [t0],
        fitness_cfg=None, return_per_task=True,
    ))
    assert list(per_task) == [1.0]


def test_evaluate_candidate_counts_failed_trajectory_as_zero(monkeypatch):
    t0, t1 = _task("0", "42"), _task("1", "9")
    _install_fake_rt(monkeypatch, {"0": _traj(t0, r"\boxed{42}")}, fail_ids={"1"})
    fit, _tr, per_task = asyncio.run(F.evaluate_candidate(
        object(), object(), _Policy(), object(), ["w"], [t0, t1],
        fitness_cfg=_BINARY, return_per_task=True,
    ))
    assert list(per_task) == [1.0, 0.0]     # failed task degrades to 0.0
    assert fit == 0.5


def test_evaluate_candidate_returns_good_trajectories_when_requested(monkeypatch):
    t0 = _task("0", "42")
    traj = _traj(t0, r"\boxed{42}")
    _install_fake_rt(monkeypatch, {"0": traj})
    _fit, trajs, _pt = asyncio.run(F.evaluate_candidate(
        object(), object(), _Policy(), object(), ["w"], [t0],
        fitness_cfg=_BINARY, return_per_task=True, return_trajectories=True,
    ))
    assert trajs == [traj]


def test_evaluate_candidate_hero_dense_adds_in_bucket_bonus(monkeypatch):
    # Two CORRECT trajectories with different self-consistency: one whose turns all
    # agree (hero_quality 1.0) and one split 1/2 (0.5). HERO min-max normalizes
    # WITHIN the correct bucket -> the consistent one gets +hero_bonus, the split
    # one +0.0, while both stay >= their binary 1.0 anchor.
    t0, t1 = _task("0", "42"), _task("1", "42")
    _install_fake_rt(monkeypatch, {
        "0": _traj(t0, r"\boxed{42}", r"\boxed{42}"),        # all agree -> q=1.0
        "1": _traj(t1, r"\boxed{42}", r"\boxed{7}", final=r"\boxed{42}"),  # split -> q=0.5
    })
    cfg = FitnessConfig(format_bonus=0.0, turn_penalty=0.0, hero_dense=True, hero_bonus=0.05)
    _fit, _tr, per_task = asyncio.run(F.evaluate_candidate(
        object(), object(), _Policy(), object(), ["w"], [t0, t1],
        fitness_cfg=cfg, return_per_task=True,
    ))
    assert per_task[0] == 1.05   # consistent correct: 1.0 + 0.05
    assert per_task[1] == 1.0    # split correct: 1.0 + 0.0


# --------------------------------------------------------------------------- #
# pure helper guards
# --------------------------------------------------------------------------- #
def test_variance_reweight_rejects_non_2d():
    import pytest

    with pytest.raises(ValueError, match="must be 2D"):
        F.variance_reweight(np.array([1.0, 0.0, 1.0]), FitnessConfig(enable_reweight=True))


def test_candidate_fitness_empty_is_zero():
    assert F._candidate_fitness(np.array([]), np.array([])) == 0.0


def test_candidate_fitness_nonpositive_weight_sum_falls_back_to_plain_mean():
    # All-zero weights -> wsum <= 0 -> plain mean rather than divide-by-zero.
    assert F._candidate_fitness(np.array([1.0, 0.0]), np.array([0.0, 0.0])) == 0.5


def test_answers_agree_unknown_benchmark_uses_stripped_text_match():
    assert F._answers_agree("no_such_benchmark", "  same ", "same") is True
    assert F._answers_agree("no_such_benchmark", "a", "b") is False


# --------------------------------------------------------------------------- #
# evaluate_population
# --------------------------------------------------------------------------- #
def _install_fake_candidate(monkeypatch, rows):
    """Fake evaluate_candidate: theta is used as an index into `rows`."""

    async def fake_evaluate_candidate(theta, spec, policy, pool, pool_models, mb, *,
                                      sample=True, client=None, return_per_task=False,
                                      fitness_cfg=None, max_turns=5, **kw):
        row = np.asarray(rows[int(theta)], dtype=float)
        return float(row.mean()), [], row

    monkeypatch.setattr("trinity.optim.fitness.evaluate_candidate", fake_evaluate_candidate)


def test_evaluate_population_returns_fits_and_calls_on_candidate(monkeypatch):
    _install_fake_candidate(monkeypatch, {0: [1.0], 1: [0.0]})
    seen = []
    fits = asyncio.run(F.evaluate_population(
        [0, 1], object(), _Policy(), object(), ["w"],
        lambda i: [object()], fitness_cfg=_BINARY,
        on_candidate=lambda i, fit, sec: seen.append((i, fit)),
    ))
    assert fits == [1.0, 0.0]
    assert seen == [(0, 1.0), (1, 0.0)]


def test_evaluate_population_variance_reweights_when_enabled(monkeypatch):
    # Rows: candidate 0 = [1,0], candidate 1 = [0,0]. Task 0 varies across
    # candidates (high sigma) -> up-weighted; task 1 is flat -> down-weighted.
    # Candidate 0's reweighted fitness therefore exceeds its plain mean 0.5.
    _install_fake_candidate(monkeypatch, {0: [1.0, 0.0], 1: [0.0, 0.0]})
    cfg = FitnessConfig(enable_reweight=True, format_bonus=0.0, turn_penalty=0.0)
    fits = asyncio.run(F.evaluate_population(
        [0, 1], object(), _Policy(), object(), ["w"],
        lambda i: [object(), object()], fitness_cfg=cfg,
    ))
    assert fits[0] > 0.5           # task-0 up-weighting lifts the [1,0] candidate
    assert fits[1] == 0.0


def test_evaluate_population_reweight_falls_back_on_width_mismatch(monkeypatch):
    # Candidates scored different minibatch sizes -> columns can't align -> the
    # reweight is skipped and the plain per-candidate means are returned.
    _install_fake_candidate(monkeypatch, {0: [1.0, 0.0], 1: [1.0]})
    cfg = FitnessConfig(enable_reweight=True, format_bonus=0.0, turn_penalty=0.0)
    widths = iter([2, 1])
    fits = asyncio.run(F.evaluate_population(
        [0, 1], object(), _Policy(), object(), ["w"],
        lambda i: [object()] * next(widths), fitness_cfg=cfg,
    ))
    assert fits == [0.5, 1.0]      # unmodified plain means
