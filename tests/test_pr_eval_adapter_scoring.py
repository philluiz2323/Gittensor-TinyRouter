"""pr_eval scores through benchmark adapters, not hardcoded reward dispatch (#154).

`_evaluate_cached` / `_evaluate_live` used to call `reward.score_text` / `reward.score`
directly, so an execution-aware benchmark like SWE-bench Verified was not scoreable
through the maintainer scorer -- `score_text("swebench_verified", ...)` raises
`ValueError: Unknown benchmark`. Routing through `get_adapter(benchmark)` fixes that
and is behaviour-preserving for math/mmlu/... (their delegating adapters forward to
the same `score_text` / committed-answer logic).

Offline: fakes only, no network.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import pr_eval  # noqa: E402
from trinity.types import Trajectory, TurnRecord  # noqa: E402

_POOL = ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"]
_GOLD = "diff --git a/f b/f\n@@ -1 +1 @@\n-x\n+y\n"
_PATCH_REF = {
    "repo": "octo/calc", "base_commit": "c0", "gold_patch": _GOLD,
    "fail_to_pass": ["t::a"],
}


class _FakePolicy:
    """Routes every item to pool index 0."""

    def decide(self, prompt, *, sample=False):
        return (0, None)


# --------------------------------------------------------------------------- #
# cached path
# --------------------------------------------------------------------------- #
def test_cached_scores_swebench_through_its_adapter():
    # The routed model (index 0) returns the gold patch; the SWE-bench adapter
    # grades it by exact normalized match -> 1.0.
    item = {
        "question_text": "resolve the issue",
        "benchmark": "swebench_verified",
        "correct_answer": _PATCH_REF,
        "model_answers": {"deepseek-v4-pro": _GOLD},
    }
    acc = pr_eval._evaluate_cached(_FakePolicy(), [item], _POOL)
    assert acc == 1.0


def test_cached_swebench_was_unscoreable_via_the_old_direct_dispatch():
    # Proof the routing change matters: the old `score_text(benchmark, ...)` path
    # cannot grade a SWE-bench item at all.
    from trinity.orchestration.reward import score_text

    with pytest.raises(ValueError, match="Unknown benchmark"):
        score_text("swebench_verified", _GOLD, _PATCH_REF)


def test_cached_math_is_behaviour_preserved():
    from trinity.orchestration.reward import score_text

    item = {
        "question_text": "2+2",
        "benchmark": "math500",
        "correct_answer": "4",
        "model_answers": {"deepseek-v4-pro": "\\boxed{4}"},
    }
    acc = pr_eval._evaluate_cached(_FakePolicy(), [item], _POOL)
    assert acc == 1.0
    # Identical to the old direct dispatch for a delegating adapter.
    assert score_text("math500", "\\boxed{4}", "4") == 1.0


def test_cached_math_wrong_answer_scores_zero():
    item = {
        "question_text": "2+2",
        "benchmark": "math500",
        "correct_answer": "4",
        "model_answers": {"deepseek-v4-pro": "\\boxed{5}"},
    }
    assert pr_eval._evaluate_cached(_FakePolicy(), [item], _POOL) == 0.0


# --------------------------------------------------------------------------- #
# live path
# --------------------------------------------------------------------------- #
def _install_fake_httpx(monkeypatch):
    class _AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    module = types.ModuleType("httpx")
    module.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)


def test_live_scores_swebench_through_its_adapter(monkeypatch):
    _install_fake_httpx(monkeypatch)

    async def fake_run_trajectory(task, *args, **kwargs):
        # The model's final answer is the gold patch.
        return Trajectory(task=task, turns=[], final_answer=_GOLD, terminated_by="done")

    monkeypatch.setattr(
        "trinity.orchestration.session.run_trajectory", fake_run_trajectory
    )

    item = {
        "question_text": "resolve the issue",
        "benchmark": "swebench_verified",
        "correct_answer": _PATCH_REF,
        "model_answers": {},
    }
    acc, _turns = asyncio.run(
        pr_eval._evaluate_live(_FakePolicy(), object(), _POOL, [item])
    )
    # SweBenchAdapter.score_trajectory grades the final patch -> 1.0.
    assert acc == 1.0


def test_live_math_is_behaviour_preserved(monkeypatch):
    _install_fake_httpx(monkeypatch)

    def _math_traj(task):
        # Final turn is a verbose verifier line with no boxed answer; an earlier
        # Worker turn boxed the answer. committed-answer scoring (which the adapter
        # uses, matching reward.score) must still recover it.
        return Trajectory(
            task=task,
            turns=[
                TurnRecord(turn=1, agent_name="glm-5p2", role=_worker_role(),
                           raw_output="\\boxed{4}", processed_output="\\boxed{4}"),
            ],
            final_answer="Looks good.",
            terminated_by="done",
        )

    async def fake_run_trajectory(task, *args, **kwargs):
        return _math_traj(task)

    monkeypatch.setattr(
        "trinity.orchestration.session.run_trajectory", fake_run_trajectory
    )

    item = {"question_text": "2+2", "benchmark": "math500", "correct_answer": "4",
            "model_answers": {}}
    acc, _turns = asyncio.run(
        pr_eval._evaluate_live(_FakePolicy(), object(), _POOL, [item])
    )
    assert acc == 1.0  # committed answer recovered, exactly as reward.score would


def _worker_role():
    from trinity.types import Role

    return Role.WORKER
