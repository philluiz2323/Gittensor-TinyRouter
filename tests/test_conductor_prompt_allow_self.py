"""Offline tests: the Conductor prompt must agree with the parse-gate on the self index.

`propose_and_run` parses with `allow_self=(max_depth > 0)`. If the prompt advertises
the self index while the gate rejects it, a prompt-obeying model produces a workflow
that cannot pass — a parse-gate false reject. No network, no GPU.
"""
from __future__ import annotations

from trinity.fugu.conductor import PromptedConductor, build_prompt
from trinity.fugu.workflow import parse_workflow
from trinity.types import Task

WORKERS = ["m-a", "m-b", "m-c"]
SELF_INDEX = len(WORKERS)
TASK = Task(task_id="t", benchmark="math500", prompt="2+2?", answer="4")


def _system(**kw) -> str:
    return build_prompt(TASK, WORKERS, **kw)[0]["content"]


# ---------------------------------------------------------------------------
# The prompt tracks allow_self
# ---------------------------------------------------------------------------
def test_self_index_is_advertised_when_recursion_is_allowed():
    system = _system(allow_self=True)
    assert f"  {SELF_INDEX}: yourself" in system
    assert f"Worker index {SELF_INDEX} means call yourself recursively" in system


def test_self_index_is_not_advertised_when_recursion_is_disabled():
    system = _system(allow_self=False)
    assert "yourself (recursive sub-workflow)" not in system
    assert "call yourself recursively" not in system
    # The real workers are still listed, in index order.
    for i, name in enumerate(WORKERS):
        assert f"  {i}: {name}" in system


def test_allow_self_defaults_to_true_so_existing_callers_are_unchanged():
    assert _system() == _system(allow_self=True)


# ---------------------------------------------------------------------------
# The regression: prompt and parse-gate must not disagree
# ---------------------------------------------------------------------------
def _routes_to_self() -> str:
    return f"model_id=[{SELF_INDEX}]\nsubtasks=['solve']\naccess_list=[[]]"


def test_a_prompt_obeying_model_is_never_rejected_by_the_gate():
    # For each parse-gate setting, a model that only uses indices the prompt
    # advertised must pass the gate.
    for allow_self in (False, True):
        system = _system(allow_self=allow_self)
        advertised_self = f"  {SELF_INDEX}: yourself" in system
        assert advertised_self == allow_self

        if advertised_self:
            # It was offered, so it must parse.
            assert parse_workflow(_routes_to_self(), len(WORKERS), allow_self=allow_self)[1]


def test_self_route_still_rejected_by_the_gate_when_disallowed():
    # Unchanged gate behaviour: the fix is to stop advertising it, not to accept it.
    assert parse_workflow(_routes_to_self(), len(WORKERS), allow_self=False)[1] is False
    assert parse_workflow(_routes_to_self(), len(WORKERS), allow_self=True)[1] is True


def test_a_plain_worker_workflow_parses_under_both_settings():
    proposal = "model_id=[0,1]\nsubtasks=['solve','answer']\naccess_list=[[],[0]]"
    assert parse_workflow(proposal, len(WORKERS), allow_self=False)[1] is True
    assert parse_workflow(proposal, len(WORKERS), allow_self=True)[1] is True


# ---------------------------------------------------------------------------
# PromptedConductor forwards the flag
# ---------------------------------------------------------------------------
class _CapturingPool:
    """Records the messages it is handed; returns a parseable proposal."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def chat(self, model, messages, **kwargs):
        self.messages = messages

        class _R:
            text = "model_id=[0]\nsubtasks=['solve']\naccess_list=[[]]"
            prompt_tokens = 1
            completion_tokens = 1

        return _R()


def _propose_system(allow_self: bool) -> str:
    import asyncio

    pool = _CapturingPool()
    conductor = PromptedConductor(pool, "m-a", allow_self=allow_self)
    asyncio.run(conductor.propose(TASK, WORKERS))
    return pool.messages[0]["content"]


def test_prompted_conductor_forwards_allow_self_into_the_prompt():
    assert "yourself (recursive sub-workflow)" in _propose_system(True)
    assert "yourself (recursive sub-workflow)" not in _propose_system(False)


def test_prompted_conductor_defaults_to_allowing_self():
    import asyncio

    pool = _CapturingPool()
    asyncio.run(PromptedConductor(pool, "m-a").propose(TASK, WORKERS))
    assert "yourself (recursive sub-workflow)" in pool.messages[0]["content"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
