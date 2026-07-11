"""Offline tests for the atomic-eval (B_env) budget planner (docs/SPEC.md §5.2).

Pure arithmetic — no torch, no numpy, no network. Pins the SPEC formula
``T = floor(B_env / (m_cma·λ))``, the forward/inverse round-trip, and the live
consumption tracker used by ``trinity.train`` under ``--budget``.
"""
import sys

import pytest

from trinity.optim import AtomicEvalBudget as _reexport  # re-export check
from trinity.optim.budget import AtomicEvalBudget, cmaes_budget, plan_generations


def test_no_torch_imported():
    assert "torch" not in sys.modules


def test_reexported_from_package():
    assert _reexport is AtomicEvalBudget


# --------------------------------------------------------------------------- #
# plan_generations / cmaes_budget
# --------------------------------------------------------------------------- #
def test_plan_generations_spec_numbers():
    # SPEC: B_env=31,680, m_cma=16, λ=33 -> T=60.
    assert plan_generations(31680, 16, 33) == 60


def test_plan_generations_floors():
    assert plan_generations(100, 2, 3) == 16   # 100 // 6
    assert plan_generations(5, 2, 3) == 0       # can't fund one generation


def test_cmaes_budget_and_roundtrip():
    assert cmaes_budget(33, 16, 60) == 31680
    for popsize, m_cma, T in [(33, 16, 60), (5, 2, 7), (10, 1, 3)]:
        assert plan_generations(cmaes_budget(popsize, m_cma, T), m_cma, popsize) == T


@pytest.mark.parametrize("b,m,p", [(-1, 16, 33), (100, 0, 3), (100, 2, 0)])
def test_plan_generations_validation(b, m, p):
    with pytest.raises(ValueError):
        plan_generations(b, m, p)


# --------------------------------------------------------------------------- #
# AtomicEvalBudget
# --------------------------------------------------------------------------- #
def test_budget_basic_properties():
    b = AtomicEvalBudget(b_env=31680, m_cma=16, popsize=33)
    assert b.cost_per_generation == 528
    assert b.max_generations == 60
    assert b.remaining == 31680 and b.consumed == 0
    assert b.fraction_used == 0.0
    assert b.can_afford_generation is True and b.exhausted is False


def test_budget_exact_spend_roundtrip():
    b = AtomicEvalBudget(b_env=cmaes_budget(33, 16, 60), m_cma=16, popsize=33)
    for _ in range(b.max_generations):
        assert b.can_afford_generation
        b.record_generation()
    assert b.consumed == 31680 and b.remaining == 0
    assert b.fraction_used == pytest.approx(1.0)
    assert b.exhausted is True


def test_budget_partial_leaves_unaffordable_remainder():
    b = AtomicEvalBudget(b_env=1000, m_cma=2, popsize=3)   # cost/gen = 6
    assert b.max_generations == 166
    for _ in range(166):
        b.record_generation()
    assert b.consumed == 996 and b.remaining == 4          # 4 < 6 -> can't fund another
    assert b.exhausted is True


def test_budget_record_accumulates_and_reports():
    b = AtomicEvalBudget(b_env=100, m_cma=2, popsize=5)    # cost/gen = 10
    assert b.record_generation() == 10
    assert b.record_generation(2) == 30
    r = b.report()
    assert r["consumed"] == 30 and r["remaining"] == 70 and r["cost_per_generation"] == 10
    assert r["max_generations"] == 10 and r["exhausted"] is False
    assert set(r) == {"b_env", "consumed", "remaining", "cost_per_generation",
                      "max_generations", "fraction_used", "exhausted"}


def test_budget_zero_is_immediately_exhausted():
    b = AtomicEvalBudget(b_env=0, m_cma=16, popsize=33)
    assert b.max_generations == 0 and b.exhausted is True and b.fraction_used == 1.0


@pytest.mark.parametrize("kw", [
    {"b_env": -1, "m_cma": 16, "popsize": 33},
    {"b_env": 100, "m_cma": 0, "popsize": 33},
    {"b_env": 100, "m_cma": 16, "popsize": 0},
    {"b_env": 100, "m_cma": 16, "popsize": 33, "consumed": -5},
])
def test_budget_validation(kw):
    with pytest.raises(ValueError):
        AtomicEvalBudget(**kw)


def test_budget_record_negative_raises():
    with pytest.raises(ValueError):
        AtomicEvalBudget(b_env=100, m_cma=2, popsize=5).record_generation(-1)
