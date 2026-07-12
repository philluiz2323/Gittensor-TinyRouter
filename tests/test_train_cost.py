"""Offline tests for the sep-CMA-ES training-cost estimate. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.optim.budget import cmaes_budget
from trinity.train_cost import estimate_cmaes_cost

PRICES = {"a": (1.0, 2.0), "b": (1.0, 2.0)}


def _est(**over):
    kw = dict(population_size=33, m_cma=16, generations=60,
              worker_names=["a", "b"], prices=PRICES,
              avg_turns=2.5, avg_prompt_tokens=1200, avg_completion_tokens=800)
    kw.update(over)
    return estimate_cmaes_cost(**kw)


# ---------------------------------------------------------------------------
# counts reuse cmaes_budget
# ---------------------------------------------------------------------------
def test_atomic_evals_matches_cmaes_budget():
    e = _est()
    assert e.atomic_evals == cmaes_budget(33, 16, 60)          # 33*16*60
    assert e.worker_calls == round(e.atomic_evals * 2.5)


def test_cost_scales_with_the_run_size():
    small = _est(generations=10).total_usd
    big = _est(generations=60).total_usd
    assert big > small
    assert big == pytest.approx(small * 6, rel=1e-6)           # linear in generations


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def test_total_equals_hand_computed_cost():
    e = _est(population_size=2, m_cma=1, generations=1, avg_turns=1,
             avg_prompt_tokens=1_000_000, avg_completion_tokens=1_000_000)
    # atomic = 2, worker_calls = 2, split over 2 models -> 1 call each.
    # per call at (1,2)/1M tokens over 1M+1M tokens = 1 + 2 = $3. Two calls = $6.
    assert e.atomic_evals == 2 and e.worker_calls == 2
    assert e.total_usd == pytest.approx(6.0)
    assert e.per_model_usd == {"a": pytest.approx(3.0), "b": pytest.approx(3.0)}


def test_unknown_worker_prices_at_zero_and_is_surfaced():
    e = _est(worker_names=["a", "unknown"])
    assert e.per_model_usd["unknown"] == 0.0
    assert "unknown" in e.per_model_usd


# ---------------------------------------------------------------------------
# receipt-floor flag
# ---------------------------------------------------------------------------
def test_below_receipt_floor_is_flagged_for_a_tiny_run():
    e = _est(generations=1, m_cma=1, population_size=2)
    assert e.total_usd < 15.0 and e.below_receipt_floor


def test_a_large_run_clears_the_floor():
    e = _est()  # 33*16*60 atomic evals -> well above $15
    assert not e.below_receipt_floor
    assert e.total_usd >= 15.0


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_negative_avg_turns_raises():
    with pytest.raises(ValueError, match="avg_turns"):
        _est(avg_turns=-1)


def test_negative_generations_raises_via_budget():
    with pytest.raises(ValueError):
        _est(generations=-1)


def test_zero_generations_is_zero_cost():
    e = _est(generations=0)
    assert e.atomic_evals == 0 and e.worker_calls == 0 and e.total_usd == 0.0
    assert e.below_receipt_floor  # $0 < $15


def test_estimate_roundtrips_to_dict():
    d = _est().to_dict()
    assert d["min_receipt_usd"] == 15.0
    assert "per_model_usd" in d and d["assumptions"]["generations"] == 60


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
