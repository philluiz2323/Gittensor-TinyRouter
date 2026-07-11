"""Tests that cached pr_eval routing uses the same turn-1 transcript as training."""
from __future__ import annotations

from trinity.orchestration.session import _transcript_text, routing_transcript


def test_routing_transcript_matches_run_trajectory_turn_one():
    prompt = "What is 2+2?"
    assert routing_transcript(prompt) == _transcript_text(prompt, [])
    assert routing_transcript(prompt).startswith("QUERY:\n")
