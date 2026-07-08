"""Tests that pr_eval live scoring uses committed answers, not final_answer only."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.orchestration.reward import score, score_text
from trinity.types import Role, Task, Trajectory, TurnRecord


def _verifier_accept_trajectory() -> Trajectory:
    """Worker boxes the answer; Verifier ACCEPTs without re-boxing."""
    task = Task(task_id="q1", benchmark="math500", prompt="2+2", answer="4")
    return Trajectory(
        task=task,
        turns=[
            TurnRecord(
                turn=1,
                agent_name="glm-5p2",
                role=Role.WORKER,
                raw_output="\\boxed{4}",
                processed_output="\\boxed{4}",
            ),
            TurnRecord(
                turn=2,
                agent_name="deepseek-v4-pro",
                role=Role.VERIFIER,
                raw_output="Looks good. VERDICT: ACCEPT",
                processed_output="Looks good. VERDICT: ACCEPT",
                verdict="ACCEPT",
            ),
        ],
        final_answer="Looks good. VERDICT: ACCEPT",
        terminated_by="accept",
    )


def test_score_recovers_answer_missed_by_final_answer_only():
    traj = _verifier_accept_trajectory()
    assert score_text("math500", traj.final_answer, "4") == 0.0
    assert score(traj) == 1.0


def test_live_eval_scoring_uses_score_not_final_answer():
    """Mirror pr_eval._evaluate_live grading: use score(traj), not final_answer."""
    traj = _verifier_accept_trajectory()
    graded_correct = score(traj) > 0.0
    legacy_wrong = score_text(traj.task.benchmark, traj.final_answer, traj.task.answer) > 0.0
    assert graded_correct
    assert not legacy_wrong
