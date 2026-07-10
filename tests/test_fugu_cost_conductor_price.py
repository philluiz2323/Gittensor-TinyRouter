"""Conductor pricing in the pre-launch cost projections.

Regression cover for the bug where passing an explicit ``prices=`` table silently
billed the Conductor $0 -- even with ``conductor_local=False`` and a
``conductor_model``. ``conductor_local`` / ``conductor_model`` were consumed only
inside ``price_table``, and that branch was skipped whenever ``prices`` was given.
A worker table carries no ``CONDUCTOR_KEY``, so the lookup fell back to ``(0, 0)``
while the returned ``assumptions`` still reported ``conductor_local: False``.

These projections gate whether a paid GRPO run is affordable, so an estimate that
under-states spend is worse than no estimate.
"""
from __future__ import annotations

import pytest

from trinity.fugu.cost import (
    CONDUCTOR_KEY,
    PRICES,
    _conductor_price,
    estimate_eval_cost,
    estimate_grpo_cost,
    price_table,
)

_WORKERS = ["qwen3.5-35b-a3b", "deepseek-v4-flash"]
_HOSTED = "minimax-m3"  # (0.30, 1.20) $/1M tokens

# One rollout, 1M prompt + 1M completion tokens of Conductor generation.
# => 0.30 + 1.20 = $1.50 of Conductor spend, exactly.
_ONE_ROLLOUT = dict(
    worker_names=_WORKERS,
    group_size=1,
    iterations=1,
    questions_per_iter=1,
    avg_conductor_prompt_tokens=1_000_000,
    avg_conductor_completion_tokens=1_000_000,
)


# --------------------------------------------------------------------------- #
# The bug: an explicit prices table must not zero the Conductor
# --------------------------------------------------------------------------- #
def test_explicit_prices_still_bills_a_hosted_conductor():
    """The regression: previously conductor_api_usd was 0.0 here."""
    est = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=False, conductor_model=_HOSTED,
        **_ONE_ROLLOUT,
    )
    assert est["conductor_api_usd"] == pytest.approx(1.50)


def test_explicit_prices_matches_the_default_table():
    """Passing PRICES explicitly must equal letting price_table build it."""
    implicit = estimate_grpo_cost(
        conductor_local=False, conductor_model=_HOSTED, **_ONE_ROLLOUT
    )
    explicit = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=False, conductor_model=_HOSTED,
        **_ONE_ROLLOUT,
    )
    assert explicit["conductor_api_usd"] == implicit["conductor_api_usd"]
    assert explicit["total_usd"] == implicit["total_usd"]


def test_reported_cost_and_assumptions_agree():
    """`conductor_local: False` with `$0` conductor spend is self-contradictory."""
    est = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=False, conductor_model=_HOSTED,
        **_ONE_ROLLOUT,
    )
    assert est["assumptions"]["conductor_local"] is False
    assert est["conductor_api_usd"] > 0.0


def test_conductor_spend_reaches_the_total():
    est = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=False, conductor_model=_HOSTED,
        **_ONE_ROLLOUT,
    )
    assert est["total_usd"] == pytest.approx(est["worker_usd"] + est["conductor_api_usd"])


def test_estimate_eval_cost_inherits_the_fix():
    """`estimate_eval_cost` forwards to `estimate_grpo_cost`."""
    est = estimate_eval_cost(
        worker_names=_WORKERS, n_tasks=1, reps=1,
        prices=dict(PRICES), conductor_local=False, conductor_model=_HOSTED,
        avg_conductor_prompt_tokens=1_000_000,
        avg_conductor_completion_tokens=1_000_000,
    )
    assert est["conductor_api_usd"] == pytest.approx(1.50)


# --------------------------------------------------------------------------- #
# Behaviour that must not change
# --------------------------------------------------------------------------- #
def test_local_conductor_is_free_with_an_explicit_table():
    est = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=True, conductor_model=_HOSTED,
        **_ONE_ROLLOUT,
    )
    assert est["conductor_api_usd"] == 0.0


def test_no_conductor_model_is_free():
    est = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=False, conductor_model=None,
        **_ONE_ROLLOUT,
    )
    assert est["conductor_api_usd"] == 0.0


def test_unknown_conductor_model_prices_at_zero():
    """An unpriced model must not raise; it prices at zero, as price_table does."""
    est = estimate_grpo_cost(
        prices=dict(PRICES), conductor_local=False, conductor_model="not-a-model",
        **_ONE_ROLLOUT,
    )
    assert est["conductor_api_usd"] == 0.0


def test_caller_supplied_conductor_key_is_respected():
    """An explicit CONDUCTOR_KEY wins over the derived one."""
    table = dict(PRICES)
    table[CONDUCTOR_KEY] = (1.0, 1.0)  # $1/1M in, $1/1M out => $2.00 per rollout
    est = estimate_grpo_cost(
        prices=table, conductor_local=False, conductor_model=_HOSTED, **_ONE_ROLLOUT
    )
    assert est["conductor_api_usd"] == pytest.approx(2.00)


def test_caller_table_is_not_mutated():
    """The estimator must not write CONDUCTOR_KEY back into the caller's dict."""
    table = dict(PRICES)
    estimate_grpo_cost(
        prices=table, conductor_local=False, conductor_model=_HOSTED, **_ONE_ROLLOUT
    )
    assert CONDUCTOR_KEY not in table


def test_worker_prices_from_the_explicit_table_win():
    """`prices` stays authoritative for worker models."""
    cheap = {name: (0.0, 0.0) for name in _WORKERS}
    est = estimate_grpo_cost(
        prices=cheap, conductor_local=True, avg_steps=1.0, **_ONE_ROLLOUT
    )
    assert est["worker_usd"] == 0.0


def test_conductor_model_resolves_against_prices_when_absent_from_the_table():
    """A restricted worker table still resolves the Conductor's rate from PRICES."""
    workers_only = {name: PRICES[name] for name in _WORKERS}
    est = estimate_grpo_cost(
        prices=workers_only, conductor_local=False, conductor_model=_HOSTED,
        **_ONE_ROLLOUT,
    )
    assert est["conductor_api_usd"] == pytest.approx(1.50)


# --------------------------------------------------------------------------- #
# The extracted rule
# --------------------------------------------------------------------------- #
def test_conductor_price_helper_is_zero_when_local():
    assert _conductor_price(PRICES, _HOSTED, True) == (0.0, 0.0)


def test_conductor_price_helper_is_zero_without_a_model():
    assert _conductor_price(PRICES, None, False) == (0.0, 0.0)


def test_conductor_price_helper_resolves_a_hosted_model():
    assert _conductor_price(PRICES, _HOSTED, False) == PRICES[_HOSTED]


def test_price_table_still_agrees_with_the_helper():
    """price_table and the estimators must not drift apart again."""
    table = price_table(_HOSTED, conductor_local=False)
    assert table[CONDUCTOR_KEY] == _conductor_price(PRICES, _HOSTED, False)
