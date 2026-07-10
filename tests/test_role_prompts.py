"""Offline unit tests for role prompt assembly (``prompts.render_transcript`` / ``build_messages``).

These pure helpers turn the running transcript into OpenAI-style chat messages on
the orchestration hot path but had no dedicated pytest coverage.
"""
from __future__ import annotations

from trinity.roles.prompts import (
    THINKER_SYSTEM,
    VERIFIER_SYSTEM,
    WORKER_SYSTEM,
    build_messages,
    render_transcript,
)
from trinity.types import Role, TurnRecord


def _turn(
    *,
    turn: int = 1,
    role: Role = Role.WORKER,
    processed: str = "step done",
    verdict: str | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        agent_name="qwen3.5-35b-a3b",
        role=role,
        raw_output=processed,
        processed_output=processed,
        verdict=verdict,
    )


def test_render_transcript_empty_sentinel():
    assert render_transcript([]) == "(no prior turns yet)"


def test_render_transcript_includes_turn_header_and_body():
    text = render_transcript([_turn(turn=2, processed="derive x=3")])
    assert "--- Turn 2 | agent=qwen3.5-35b-a3b | role=WORKER ---" in text
    assert "derive x=3" in text


def test_render_transcript_surfaces_verifier_parsed_verdict():
    text = render_transcript([
        _turn(role=Role.VERIFIER, processed="Looks good.", verdict="ACCEPT"),
    ])
    assert "[parsed verdict: ACCEPT]" in text


def test_build_messages_layout_and_role_system_prompt():
    msgs = build_messages(Role.THINKER, "What is 2+2?", [])
    assert msgs[0] == {"role": "system", "content": THINKER_SYSTEM}
    assert "QUERY:\nWhat is 2+2?" in msgs[1]["content"]
    assert "TRANSCRIPT SO FAR:\n(no prior turns yet)" in msgs[1]["content"]


def test_build_messages_embeds_rendered_transcript():
    transcript = [_turn(turn=1, role=Role.WORKER, processed="try 4")]
    msgs = build_messages(Role.VERIFIER, "Solve.", transcript)
    assert msgs[0]["content"] == VERIFIER_SYSTEM
    assert "TRANSCRIPT SO FAR:\n--- Turn 1" in msgs[1]["content"]
    assert "try 4" in msgs[1]["content"]
    assert msgs[0]["content"] != WORKER_SYSTEM
