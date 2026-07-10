"""The shaped-fitness format bonus is judged on the committed answer.

`_task_reward` scores correctness via `reward.score` (which selects the committed
answer — the most recent turn with an extractable answer), so the `format_bonus`
predicate must inspect that same text. Otherwise a trajectory scored correct via an
answer boxed in an earlier turn, whose final turn re-phrased without re-boxing, is
denied the bonus it earned. Offline / numpy-only: no torch, no API. (Follow-up to
the shaped-fitness work; see reward.has_answer's consistency contract.)
"""
from __future__ import annotations

import pytest

from trinity.optim import fitness as F
from trinity.optim.fitness import FitnessConfig
from trinity.orchestration import reward as R
from trinity.types import Role, Task, Trajectory, TurnRecord


def _worker(turn: int, text: str) -> TurnRecord:
    return TurnRecord(turn, "worker-a", Role.WORKER, text, text)


def _verifier(turn: int, text: str) -> TurnRecord:
    return TurnRecord(turn, "verifier-a", Role.VERIFIER, text, text)


def _math_traj(turns: list[TurnRecord], final_answer: str) -> Trajectory:
    task = Task(task_id="m1", benchmark="math500", prompt="2+2?", answer="4")
    return Trajectory(task=task, turns=turns, final_answer=final_answer)


# The bug scenario: correct answer committed in turn 1, final turn has no answer.
_EARLY_ANSWER = _math_traj(
    turns=[_worker(1, r"So it is \boxed{4}."), _verifier(2, "Looks right to me.")],
    final_answer="Looks right to me.",
)


def test_committed_answer_and_final_answer_disagree_on_format():
    # Establishes the precondition: this trajectory is exactly the case where
    # scoring the final output alone would miss the parseable answer.
    assert R.score(_EARLY_ANSWER) == 1.0
    assert R.has_answer("math500", _EARLY_ANSWER.final_answer) is False
    assert R.has_answer("math500", R.committed_answer("math500", _EARLY_ANSWER)) is True


def test_format_bonus_awarded_for_early_committed_answer():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.0)
    # correct=1, has_answer via committed=True, turn_penalty off -> 1 + 0.05 = 1.05
    assert F._task_reward(_EARLY_ANSWER, cfg, max_turns=5) == pytest.approx(1.05)


def test_binary_correctness_is_unchanged_and_stored():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.0)
    F._task_reward(_EARLY_ANSWER, cfg, max_turns=5)
    assert _EARLY_ANSWER.reward == 1.0  # eval-facing binary is untouched


def test_no_bonus_when_no_turn_has_an_answer():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.0)
    traj = _math_traj(
        turns=[_worker(1, "let me think"), _verifier(2, "I am unsure")],
        final_answer="I am unsure",
    )
    # Nothing parseable anywhere -> wrong and no format bonus.
    assert R.score(traj) == 0.0
    assert F._task_reward(traj, cfg, max_turns=5) == pytest.approx(0.0)


def test_final_answer_with_box_is_unaffected():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.0)
    traj = _math_traj(
        turns=[_worker(1, "working..."), _worker(2, r"final: \boxed{4}")],
        final_answer=r"final: \boxed{4}",
    )
    # committed == final here, so behavior is identical to before the fix.
    assert F._task_reward(traj, cfg, max_turns=5) == pytest.approx(1.05)


def test_shaping_off_returns_plain_binary_regardless():
    cfg = FitnessConfig(format_bonus=0.0, turn_penalty=0.0, hero_dense=False)
    assert F._task_reward(_EARLY_ANSWER, cfg, max_turns=5) == pytest.approx(1.0)
