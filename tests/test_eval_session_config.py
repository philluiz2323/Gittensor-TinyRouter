"""Eval must run the trained protocol: read the config ``session:`` block.

``train.py`` derives the turn budget ``K`` from ``session.max_turns`` and threads
``verifier_requires_prior_worker`` from the same block. ``eval.resolve_session_run_kwargs``
mirrors that so a coordinator trained with a non-default ``session:`` is evaluated
under the same protocol (and the budget-matched single-model baselines are sized
against the right ``K``), instead of silently falling back to CLI defaults.

Pure / offline — exercises only the config-resolution contract.
"""
from __future__ import annotations

from types import SimpleNamespace

from trinity.eval import resolve_session_run_kwargs, single_model_budget


def _args(max_turns: int = 0, max_tokens: int = 4096, reasoning: str = "minimal"):
    return SimpleNamespace(max_turns=max_turns, max_tokens=max_tokens, reasoning=reasoning)


def test_defaults_when_session_absent():
    rk = resolve_session_run_kwargs(_args(), {})
    assert rk["max_turns"] == 5
    assert rk["verifier_requires_prior_worker"] is True
    assert rk["max_tokens"] == 4096
    assert rk["reasoning"] == "minimal"


def test_session_max_turns_is_honoured():
    # No CLI override -> the config's K wins (was ignored before the fix).
    rk = resolve_session_run_kwargs(_args(max_turns=0), {"max_turns": 8})
    assert rk["max_turns"] == 8


def test_cli_max_turns_overrides_config():
    rk = resolve_session_run_kwargs(_args(max_turns=3), {"max_turns": 8})
    assert rk["max_turns"] == 3


def test_verifier_gate_flows_from_session():
    rk = resolve_session_run_kwargs(_args(), {"verifier_requires_prior_worker": False})
    assert rk["verifier_requires_prior_worker"] is False


def test_budget_matches_the_resolved_session_turns():
    # The single-model baseline budget must track the SAME K eval runs TRINITY at.
    rk = resolve_session_run_kwargs(_args(max_turns=0), {"max_turns": 8})
    assert single_model_budget(rk["max_tokens"], rk["max_turns"]) == 8 * 4096


def test_matches_train_resolution_rule():
    # Parity with train.py: `args.max_turns or session.get("max_turns", 5)`.
    for cli, sess, expected in [(0, {}, 5), (0, {"max_turns": 6}, 6), (2, {"max_turns": 6}, 2)]:
        rk = resolve_session_run_kwargs(_args(max_turns=cli), sess)
        assert rk["max_turns"] == expected
