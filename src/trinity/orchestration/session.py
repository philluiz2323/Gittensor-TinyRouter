"""Run one query through the multi-turn coordination loop.

Each turn: encode current state -> head/policy picks (model, role) or stop ->
call the selected Fireworks model with the role prompt -> append output ->
repeat until stop or max_turns. Returns the final answer + trace for scoring.

TODO(SPEC §2/§4): implement the loop, context-passing, and final-answer
extraction per docs/SPEC.md.
"""
from __future__ import annotations
# Placeholder — implemented once docs/SPEC.md pins the protocol.
