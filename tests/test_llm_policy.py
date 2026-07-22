"""Offline tests for the LLM-as-coordinator baseline (SPEC R11, Table 8).

Pure stdlib over a stub chat callable — no network, no GPU, no torch.
"""
from __future__ import annotations

import asyncio

import pytest

from trinity.coordinator.llm_policy import (
    DEFAULT_FALLBACK_ROLE,
    LLM_COORDINATOR_SYSTEM,
    MAX_FALLBACK_RATE,
    DecisionStats,
    LLMCoordinatorPolicy,
    StubDecider,
    build_decision_prompt,
    parse_decision,
)
from trinity.types import ROLE_ORDER, Role

POOL = ["gpt-5", "gemini-3.1-flash", "deepseek-v4"]


def _policy(reply, **kw) -> LLMCoordinatorPolicy:
    return LLMCoordinatorPolicy(StubDecider(reply), POOL, **kw)


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------


def test_prompt_lists_every_pool_model_by_name():
    msgs = build_decision_prompt("QUERY:\n2+2?", POOL)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == LLM_COORDINATOR_SYSTEM
    for m in POOL:
        assert m in msgs[1]["content"]


def test_prompt_carries_the_transcript_the_trained_head_would_see():
    """Both coordinators must route on identical context or R11 is not a fair test."""
    transcript = "QUERY:\nsolve x\n\n[Turn 1 | worker | gpt-5]\npartial"
    msgs = build_decision_prompt(transcript, POOL)
    assert transcript in msgs[1]["content"]


def test_prompt_states_remaining_budget_only_when_both_bounds_are_known():
    with_budget = build_decision_prompt("q", POOL, turn=2, max_turns=5)[1]["content"]
    assert "turn 2 of at most 5" in with_budget
    assert "4 remaining" in with_budget

    without = build_decision_prompt("q", POOL, turn=2)[1]["content"]
    assert "remaining" not in without


def test_prompt_requires_a_non_empty_pool():
    with pytest.raises(ValueError, match="pool_models must be non-empty"):
        build_decision_prompt("q", [])


def test_system_prompt_names_all_three_roles_and_the_output_format():
    for r in ROLE_ORDER:
        assert r.name in LLM_COORDINATOR_SYSTEM
    assert "AGENT:" in LLM_COORDINATOR_SYSTEM
    assert "ROLE:" in LLM_COORDINATOR_SYSTEM


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def test_parses_the_documented_format():
    assert parse_decision("AGENT: gemini-3.1-flash\nROLE: THINKER", POOL) == (1, Role.THINKER)


@pytest.mark.parametrize(
    "text",
    [
        "agent: gpt-5\nrole: worker",                       # lowercase
        "  AGENT :  gpt-5 \n  ROLE :  Worker ",             # padded
        "AGENT: **gpt-5**\nROLE: WORKER",                   # markdown emphasis
        "Sure!\nAGENT: gpt-5\nROLE: WORKER\nHope that helps.",  # surrounding prose
    ],
)
def test_parsing_tolerates_harmless_formatting_variation(text):
    assert parse_decision(text, POOL) == (0, Role.WORKER)


def test_the_last_occurrence_wins_so_reasoning_aloud_is_read_correctly():
    text = (
        "First I considered AGENT: gpt-5\nROLE: THINKER\n"
        "but on reflection:\nAGENT: deepseek-v4\nROLE: VERIFIER"
    )
    assert parse_decision(text, POOL) == (2, Role.VERIFIER)


def test_accepts_the_one_based_menu_index_it_advertises():
    assert parse_decision("AGENT: 2\nROLE: WORKER", POOL) == (1, Role.WORKER)
    assert parse_decision("AGENT: 2.\nROLE: WORKER", POOL) == (1, Role.WORKER)
    assert parse_decision("AGENT: 3. deepseek-v4\nROLE: WORKER", POOL) == (2, Role.WORKER)


def test_a_zero_index_is_rejected_not_read_as_the_first_model():
    """The menu is 1-based; reading '0' as index 0 would invent a decision."""
    assert parse_decision("AGENT: 0\nROLE: WORKER", POOL) is None


def test_an_out_of_range_index_is_a_parse_failure():
    assert parse_decision("AGENT: 9\nROLE: WORKER", POOL) is None


def test_an_agent_outside_the_pool_is_a_parse_failure_not_a_guess():
    """Coercing an unknown name to index 0 would fabricate a routing decision."""
    assert parse_decision("AGENT: claude-opus\nROLE: WORKER", POOL) is None


def test_an_unknown_role_is_a_parse_failure():
    assert parse_decision("AGENT: gpt-5\nROLE: SOLVER", POOL) is None


@pytest.mark.parametrize("text", ["", "no fields here", "AGENT: gpt-5", "ROLE: WORKER"])
def test_missing_or_empty_fields_parse_to_none(text):
    assert parse_decision(text, POOL) is None


# --------------------------------------------------------------------------------------
# The policy itself
# --------------------------------------------------------------------------------------


def test_decide_returns_the_policy_protocol_shape():
    idx, role = _policy("AGENT: deepseek-v4\nROLE: VERIFIER").decide("q")
    assert (idx, role) == (2, Role.VERIFIER)


def test_decide_accepts_and_ignores_sample_and_rng():
    """Protocol compatibility: the baseline's randomness lives in the model's decoding."""
    p = _policy("AGENT: gpt-5\nROLE: WORKER")
    assert p.decide("q", sample=True, rng=object()) == (0, Role.WORKER)


def test_unparseable_replies_fall_back_deterministically_and_are_counted():
    p = _policy("I'd rather not say")
    for _ in range(3):
        assert p.decide("q") == (0, DEFAULT_FALLBACK_ROLE)
    assert p.stats.parse_failures == 3
    assert p.stats.fallback_rate == pytest.approx(1.0)


def test_a_raising_client_is_recorded_not_propagated():
    """One dead call must not abort a whole baseline run."""
    def boom(_messages):
        raise RuntimeError("gateway down")

    p = LLMCoordinatorPolicy(boom, POOL)
    assert p.decide("q") == (0, DEFAULT_FALLBACK_ROLE)
    assert p.stats.client_errors == 1
    assert p.stats.parse_failures == 0


def test_the_fallback_is_configurable_and_validated():
    p = _policy("junk", fallback_agent_idx=2, fallback_role=Role.THINKER)
    assert p.decide("q") == (2, Role.THINKER)

    with pytest.raises(ValueError, match="out of range"):
        _policy("junk", fallback_agent_idx=9)
    with pytest.raises(ValueError, match="pool_models must be non-empty"):
        LLMCoordinatorPolicy(StubDecider("x"), [])


def test_turn_context_advances_and_resets_per_trajectory():
    stub = StubDecider("AGENT: gpt-5\nROLE: WORKER")
    p = LLMCoordinatorPolicy(stub, POOL, max_turns=5)
    p.decide("q")
    p.decide("q")
    assert "turn 2 of at most 5" in stub.calls[-1][1]["content"]

    p.reset()
    p.decide("q")
    assert "turn 1 of at most 5" in stub.calls[-1][1]["content"]


def test_history_records_every_decision_with_its_raw_reply():
    p = _policy("AGENT: gpt-5\nROLE: WORKER")
    p.decide("q")
    assert p.history[-1].parsed is True
    assert p.history[-1].raw == "AGENT: gpt-5\nROLE: WORKER"


# --------------------------------------------------------------------------------------
# Accounting — the two integrity properties R11 depends on
# --------------------------------------------------------------------------------------


def test_a_fallback_dominated_run_is_not_representative():
    """A constant policy wearing the LLM's name must not be compared as one."""
    p = _policy("junk")
    for _ in range(10):
        p.decide("q")
    assert p.stats.fallback_rate == 1.0
    assert p.stats.is_representative() is False


def test_a_clean_run_is_representative():
    p = _policy("AGENT: gpt-5\nROLE: WORKER")
    for _ in range(10):
        p.decide("q")
    assert p.stats.fallbacks == 0
    assert p.stats.is_representative() is True


def test_representativeness_threshold_is_the_documented_bar():
    stats = DecisionStats(calls=100, parse_failures=int(MAX_FALLBACK_RATE * 100))
    assert stats.is_representative() is True
    stats_worse = DecisionStats(calls=100, parse_failures=int(MAX_FALLBACK_RATE * 100) + 1)
    assert stats_worse.is_representative() is False


def test_an_empty_run_is_never_representative():
    assert DecisionStats().is_representative() is False
    assert DecisionStats().fallback_rate == 0.0


def test_coordinator_token_spend_is_accounted_so_r11_can_be_budget_matched():
    """The trained head costs ~0 tokens per decision; this baseline does not."""
    p = _policy(
        "AGENT: gpt-5\nROLE: WORKER",
        token_counter=lambda messages, reply: (120, 8),
    )
    for _ in range(4):
        p.decide("q")
    assert p.stats.prompt_tokens == 480
    assert p.stats.completion_tokens == 32


def test_stats_track_the_routing_distribution():
    replies = iter(
        [
            "AGENT: gpt-5\nROLE: WORKER",
            "AGENT: gpt-5\nROLE: THINKER",
            "AGENT: deepseek-v4\nROLE: WORKER",
        ]
    )
    p = _policy(lambda _m: next(replies))
    for _ in range(3):
        p.decide("q")
    assert p.stats.agent_counts == {0: 2, 2: 1}
    assert p.stats.role_counts == {"worker": 2, "thinker": 1}


def test_stats_to_dict_is_json_serializable():
    import json

    p = _policy("AGENT: gpt-5\nROLE: WORKER", token_counter=lambda m, r: (10, 2))
    p.decide("q")
    d = p.stats.to_dict()
    json.dumps(d)
    assert d["calls"] == 1
    assert d["is_representative"] is True


# --------------------------------------------------------------------------------------
# It really is a drop-in for the session loop
# --------------------------------------------------------------------------------------


def test_it_satisfies_the_session_policy_protocol_end_to_end():
    """Runs through the REAL run_trajectory with no change to the session loop."""
    from trinity.orchestration.session import run_trajectory
    from trinity.types import Task

    class _Pool:
        async def chat(self, model, messages, **kw):
            class _R:
                text = "VERDICT: ACCEPT\nThe answer is 4."
                completion_tokens = 5
                prompt_tokens = 10
                cost_usd = 0.0
            return _R()

    policy = _policy("AGENT: gpt-5\nROLE: WORKER")
    traj = asyncio.run(
        run_trajectory(
            Task(task_id="t1", benchmark="math500", prompt="2+2?", answer="4"),
            policy,
            _Pool(),
            POOL,
            max_turns=2,
        )
    )

    assert traj.turns, "the trajectory should have run at least one turn"
    assert traj.turns[0].agent_name == "gpt-5"
    assert traj.turns[0].role is Role.WORKER
    assert policy.stats.calls >= 1
