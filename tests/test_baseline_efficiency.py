"""Offline tests for the SPEC R12 token-efficiency verifier (issue #364).

R12: TRINITY is *far* more token-efficient than the multi-agent baselines. The headline
property is that efficiency is measured per CORRECT answer, so a baseline cannot look
efficient by answering cheaply and wrongly. Pure stdlib — no torch/network/GPU.
"""
from __future__ import annotations

import json
import math

import pytest

from trinity.analysis.baseline_efficiency import (
    analyze,
    render,
    tokens_per_correct,
)

_TRINITY = {"accuracy": 0.80, "tokens_per_query": 1000.0}      # tpc = 1250


def _systems(**baselines) -> dict:
    return {"TRINITY": dict(_TRINITY), **baselines}


# --------------------------------------------------------------------------- #
# tokens_per_correct
# --------------------------------------------------------------------------- #
def test_tokens_per_correct_divides_by_accuracy():
    assert tokens_per_correct(0.5, 1000.0) == 2000.0
    assert tokens_per_correct(1.0, 1000.0) == 1000.0


def test_tokens_per_correct_of_a_never_correct_system_is_infinite():
    assert tokens_per_correct(0.0, 1000.0) == math.inf
    assert tokens_per_correct(-0.1, 1000.0) == math.inf


# --------------------------------------------------------------------------- #
# the headline: cheap-but-wrong must not look efficient
# --------------------------------------------------------------------------- #
def test_cheap_but_wrong_baseline_is_not_efficient():
    # MoA spends HALF of TRINITY's tokens per query, but is only a quarter as accurate:
    # 500/0.20 = 2500 tokens/correct vs TRINITY's 1000/0.80 = 1250 -> TRINITY is 2x better.
    r = analyze(_systems(MoA={"accuracy": 0.20, "tokens_per_query": 500.0}))
    moa = r["baselines"][0]
    assert moa["tokens_per_correct"] == 2500.0
    assert moa["speedup"] == pytest.approx(2.0) and moa["far_more_efficient"] is True
    assert r["holds"] is True


def test_r12_holds_when_every_baseline_is_beaten_by_the_factor():
    r = analyze(_systems(
        MoA={"accuracy": 0.77, "tokens_per_query": 8400.0},        # tpc ~10909 -> 8.7x
        Smoothie={"accuracy": 0.74, "tokens_per_query": 5200.0},   # tpc ~7027  -> 5.6x
        MasRouter={"accuracy": 0.78, "tokens_per_query": 6100.0},  # tpc ~7821  -> 6.3x
    ))
    assert r["holds"] is True and r["n_baselines"] == 3
    assert all(b["far_more_efficient"] for b in r["baselines"])
    assert r["min_speedup"] > 2.0


def test_r12_fails_when_any_single_baseline_is_within_the_factor():
    # Smoothie is only 1.5x worse -> "far more efficient" does not hold for EVERY baseline.
    r = analyze(_systems(
        MoA={"accuracy": 0.77, "tokens_per_query": 8400.0},
        Smoothie={"accuracy": 0.80, "tokens_per_query": 1500.0},   # tpc 1875 -> 1.5x
    ))
    assert r["holds"] is False
    assert r["worst_baseline"] == "Smoothie" and r["min_speedup"] == pytest.approx(1.5)


def test_factor_is_configurable():
    sys_ = _systems(Smoothie={"accuracy": 0.80, "tokens_per_query": 1500.0})   # 1.5x
    assert analyze(sys_, factor=1.5)["holds"] is True      # 1.5x clears a 1.5x bar
    assert analyze(sys_, factor=2.0)["holds"] is False


def test_equal_efficiency_is_not_far_more_efficient():
    r = analyze(_systems(Twin=dict(_TRINITY)))
    assert r["baselines"][0]["speedup"] == pytest.approx(1.0)
    assert r["holds"] is False


# --------------------------------------------------------------------------- #
# degenerate systems
# --------------------------------------------------------------------------- #
def test_never_correct_baseline_is_infinitely_worse():
    r = analyze(_systems(Broken={"accuracy": 0.0, "tokens_per_query": 900.0}))
    b = r["baselines"][0]
    assert b["tokens_per_correct"] is None            # infinite -> JSON-safe None
    assert b["speedup"] is None and b["far_more_efficient"] is True
    assert r["holds"] is True


def test_never_correct_trinity_fails_r12():
    systems = {"TRINITY": {"accuracy": 0.0, "tokens_per_query": 10.0},
               "MoA": {"accuracy": 0.7, "tokens_per_query": 8000.0}}
    r = analyze(systems)
    assert r["baselines"][0]["speedup"] == 0.0 and r["holds"] is False


def test_missing_or_malformed_trinity_raises():
    with pytest.raises(ValueError):
        analyze({"MoA": {"accuracy": 0.7, "tokens_per_query": 8000.0}})
    with pytest.raises(ValueError):
        analyze({"TRINITY": {"accuracy": 0.8, "tokens_per_query": 0.0}})   # non-positive


@pytest.mark.parametrize("bad", [
    {"accuracy": "high", "tokens_per_query": 100.0},
    {"accuracy": 0.5},                                   # missing tokens
    {"accuracy": 0.5, "tokens_per_query": -5.0},         # non-positive tokens
    "not-a-mapping",
])
def test_malformed_baselines_are_dropped_not_silently_passed(bad):
    r = analyze(_systems(Junk=bad))
    assert r["n_baselines"] == 0 and r["holds"] is False   # no baseline -> claim unproven


def test_no_baselines_means_r12_is_unproven():
    assert analyze({"TRINITY": dict(_TRINITY)})["holds"] is False


# --------------------------------------------------------------------------- #
# report shape / render
# --------------------------------------------------------------------------- #
def test_report_is_json_serializable_even_with_infinities():
    r = analyze(_systems(Broken={"accuracy": 0.0, "tokens_per_query": 900.0}))
    assert json.loads(json.dumps(r))["invariant"] == "R12"


def test_render_holds_and_fails():
    md = render(_systems(MoA={"accuracy": 0.77, "tokens_per_query": 8400.0}))
    assert "R12 HOLDS" in md and "tokens per correct" in md and "| MoA |" in md
    bad = render(_systems(Smoothie={"accuracy": 0.80, "tokens_per_query": 1500.0}))
    assert "R12 does NOT hold" in bad and "Smoothie" in bad


def test_render_without_baselines():
    assert "no baselines" in render({"TRINITY": dict(_TRINITY)})
