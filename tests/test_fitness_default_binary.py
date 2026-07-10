"""Offline tests: `fitness_cfg=None` must mean plain binary reward, not shaping.

`evaluate_candidate` / `evaluate_population` document ``None`` as "plain binary,
original behavior", and the S8 smoke gate asserts ``0.0 <= fit <= 1.0``. These
tests pin that contract. No network, no GPU, no torch: `run_trajectory` is
stubbed and the policy is a no-op.
"""
from __future__ import annotations

import asyncio

import numpy as np

import trinity.optim.fitness as F
from trinity.types import Role, Task, Trajectory, TurnRecord

BENCH = "math500"


class _NoOpPolicy:
    """Stands in for CoordinatorPolicy; θ configuration is irrelevant here."""

    def configure(self, theta, spec) -> None:  # noqa: D102 - test stub
        return None


def _trajectory(task: Task, *, answer: str, n_turns: int) -> Trajectory:
    traj = Trajectory(task=task)
    traj.final_answer = answer
    traj.turns = [
        TurnRecord(turn=i + 1, agent_name="m", role=Role.WORKER,
                   raw_output="", processed_output="")
        for i in range(n_turns)
    ]
    return traj


def _run_candidate(monkeypatch, *, answer: str, n_turns: int = 1, fitness_cfg=None):
    """Score one task whose trajectory is fixed by `answer` / `n_turns`."""

    async def fake_run(task, policy, pool, pool_models, **kwargs):
        return _trajectory(task, answer=answer, n_turns=n_turns)

    monkeypatch.setattr(F, "run_trajectory", fake_run)
    tasks = [Task(task_id="0", benchmark=BENCH, prompt="q", answer="42")]
    return asyncio.run(
        F.evaluate_candidate(
            None, None, _NoOpPolicy(), None, ["m"], tasks,
            sample=False, fitness_cfg=fitness_cfg, return_per_task=True, max_turns=2,
        )
    )


def test_default_none_gives_plain_binary_reward(monkeypatch):
    # A correct answer must score exactly 1.0 -- not 1.0 + format_bonus.
    fit, _, per_task = _run_candidate(monkeypatch, answer=r"\boxed{42}")
    assert set(np.unique(per_task)) <= {0.0, 1.0}, per_task
    assert fit == 1.0


def test_default_none_stays_within_the_s8_fitness_bounds(monkeypatch):
    # tests/smoke/run_smoke.py (S8) calls evaluate_candidate with no fitness_cfg
    # and asserts 0.0 <= fit <= 1.0. Shaping on the default path breaks that.
    fit, _, _ = _run_candidate(monkeypatch, answer=r"\boxed{42}")
    assert 0.0 <= fit <= 1.0


def test_default_none_gives_no_credit_for_a_wrong_but_parseable_answer(monkeypatch):
    # Formatted-but-wrong must be 0.0 under binary reward (shaping would give 0.05).
    fit, _, per_task = _run_candidate(monkeypatch, answer=r"\boxed{7}")
    assert fit == 0.0
    assert per_task.tolist() == [0.0]


def test_explicit_shaped_config_still_shapes(monkeypatch):
    # The opt-in path is untouched: a wrong-but-parseable answer earns the bonus.
    cfg = F.FitnessConfig(format_bonus=0.05, turn_penalty=0.0)
    fit, _, _ = _run_candidate(monkeypatch, answer=r"\boxed{7}", fitness_cfg=cfg)
    assert fit > 0.0


def test_fitness_config_field_defaults_are_unchanged():
    # The configured shaping values (used via configs/trinity.yaml and from_dict)
    # keep their 0.05 defaults; only the None fallback changes.
    cfg = F.FitnessConfig()
    assert (cfg.format_bonus, cfg.turn_penalty) == (0.05, 0.05)
    assert cfg.shaping_active is True
    assert F.FitnessConfig.from_dict(None).format_bonus == 0.05


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
