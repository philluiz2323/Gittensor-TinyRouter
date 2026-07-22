"""Offline tests for the R3 (TRINITY > best multi-agent baseline) verifier.

No network, no GPU, no torch — plain numbers only.
"""
from __future__ import annotations

import pytest

from trinity.analysis.multi_agent_baseline import (
    SPEC_BASELINES,
    analyze_benchmark,
    analyze_benchmarks,
    render,
)


def _bl(moa=None, masrouter=None, routerdc=None, smoothie=None):
    d = {"MoA": moa, "MasRouter": masrouter, "RouterDC": routerdc, "Smoothie": smoothie}
    return {k: v for k, v in d.items() if v is not None}


# --------------------------------------------------------------------------- #
# analyze_benchmark
# --------------------------------------------------------------------------- #
def test_spec_names_the_four_baselines():
    assert SPEC_BASELINES == ("MoA", "MasRouter", "RouterDC", "Smoothie")


def test_trinity_beats_best_baseline_holds():
    c = analyze_benchmark(0.72, _bl(moa=0.66, masrouter=0.68, routerdc=0.60, smoothie=0.64),
                          benchmark="math500")
    assert c.comparable and c.holds
    assert c.best_baseline == "MasRouter" and c.best_baseline_score == 0.68
    assert c.margin == pytest.approx(0.04)


def test_trinity_below_best_baseline_fails():
    c = analyze_benchmark(0.65, _bl(moa=0.66, masrouter=0.68))
    assert c.comparable and not c.holds
    assert c.best_baseline == "MasRouter"
    assert c.margin == pytest.approx(-0.03)


def test_tie_with_best_baseline_is_not_a_win():
    c = analyze_benchmark(0.68, _bl(masrouter=0.68, moa=0.60))
    assert c.comparable and not c.holds
    assert c.margin == 0.0


def test_best_of_present_baselines_is_the_bar():
    # Only two baselines present -> the stronger of them is the bar.
    c = analyze_benchmark(0.70, _bl(moa=0.55, smoothie=0.69))
    assert c.best_baseline == "Smoothie" and c.holds


def test_non_numeric_baselines_ignored_and_missing_trinity_incomparable():
    c1 = analyze_benchmark(0.7, {"MoA": "n/a", "MasRouter": 0.6})
    assert c1.comparable and c1.best_baseline == "MasRouter"
    c2 = analyze_benchmark(None, _bl(moa=0.6))
    assert not c2.comparable and not c2.holds and c2.margin is None
    c3 = analyze_benchmark(0.7, {"MoA": "n/a"})   # no numeric baseline
    assert not c3.comparable and not c3.holds


# --------------------------------------------------------------------------- #
# analyze_benchmarks: union verdict
# --------------------------------------------------------------------------- #
def test_union_holds_when_every_benchmark_beats_its_best_baseline():
    report = analyze_benchmarks({
        "math500": (0.72, _bl(moa=0.66, masrouter=0.68)),
        "mmlu": (0.70, _bl(routerdc=0.60, smoothie=0.65)),
    })
    assert report["r3_holds"] is True
    assert report["violations"] == []
    assert report["union_trinity"] == pytest.approx((0.72 + 0.70) / 2)
    assert report["union_best_baseline"] == pytest.approx((0.68 + 0.65) / 2)
    assert report["union_margin"] > 0


def test_union_violated_when_any_benchmark_loses():
    report = analyze_benchmarks({
        "math500": (0.72, _bl(moa=0.66)),
        "livecodebench": (0.60, _bl(masrouter=0.70)),   # loses
    })
    assert report["r3_holds"] is False
    assert report["violations"] == ["livecodebench"]


def test_incomparable_rows_excluded_from_union():
    report = analyze_benchmarks({
        "math500": (0.72, _bl(moa=0.66)),
        "gpqa": (0.4, {}),                              # no baselines -> incomparable
    })
    assert report["r3_holds"] is True
    assert report["union_trinity"] == pytest.approx(0.72)
    assert "gpqa" not in report["violations"]


def test_all_incomparable_does_not_hold():
    report = analyze_benchmarks({"math500": (0.72, {})})
    assert report["r3_holds"] is False
    assert report["union_trinity"] == 0.0


def test_accepts_mapping_form():
    report = analyze_benchmarks({
        "math500": {"trinity": 0.72, "baselines": _bl(moa=0.66, masrouter=0.68)},
    })
    assert report["r3_holds"] is True


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_reports_holds_and_names_the_best_baseline():
    md = render({"math500": (0.72, _bl(moa=0.66, masrouter=0.68))})
    assert "R3 (TRINITY > best multi-agent baseline): HOLDS" in md
    assert "MasRouter 0.680" in md
    assert "| benchmark | trinity | best baseline | margin | R3 |" in md


def test_render_flags_violation_and_incomparable():
    md = render({"lcb": (0.60, _bl(masrouter=0.70)), "gpqa": (0.4, {})})
    assert "VIOLATED" in md and "violations: lcb" in md
    assert "| gpqa | - | - | - | n/a |" in md
