"""Offline tests for the R11 (trained coordinator > LLM-as-coordinator) verifier.

No network, no GPU, no torch — plain numbers only.
"""
from __future__ import annotations

import pytest

from trinity.analysis.coordinator_vs_llm import (
    analyze_benchmarks,
    analyze_pair,
    render,
)


# --------------------------------------------------------------------------- #
# analyze_pair
# --------------------------------------------------------------------------- #
def test_trained_above_baseline_holds():
    c = analyze_pair(0.615, 0.5376, benchmark="livecodebench")
    assert c.comparable and c.holds
    assert c.margin == pytest.approx(0.615 - 0.5376)


def test_trained_below_or_equal_does_not_hold():
    below = analyze_pair(0.50, 0.5376, benchmark="mmlu")
    assert below.comparable and not below.holds
    assert below.margin == pytest.approx(-0.0376)

    tie = analyze_pair(0.60, 0.60)
    assert tie.comparable and not tie.holds       # strict: a tie is not a win
    assert tie.margin == 0.0


def test_missing_or_non_numeric_is_not_comparable():
    for c in (
        analyze_pair(None, 0.5376),
        analyze_pair(0.6, None),
        analyze_pair("n/a", 0.5),
        analyze_pair(True, 0.5),                   # bool is not a valid accuracy
    ):
        assert not c.comparable and not c.holds and c.margin is None


# --------------------------------------------------------------------------- #
# analyze_benchmarks: union verdict
# --------------------------------------------------------------------------- #
def test_union_holds_when_every_comparable_benchmark_wins():
    report = analyze_benchmarks({
        "math500": (0.72, 0.60),
        "mmlu": (0.68, 0.55),
        "livecodebench": (0.615, 0.5376),
    })
    assert report["r11_holds"] is True
    assert report["violations"] == []
    assert report["union_margin"] > 0
    # Equal-weight union means over the three benchmarks.
    assert report["union_trained"] == pytest.approx((0.72 + 0.68 + 0.615) / 3)
    assert report["union_llm_as_coordinator"] == pytest.approx((0.60 + 0.55 + 0.5376) / 3)


def test_union_violated_when_any_benchmark_loses():
    report = analyze_benchmarks({
        "math500": (0.72, 0.60),
        "mmlu": (0.50, 0.55),          # trained loses here
    })
    assert report["r11_holds"] is False
    assert report["violations"] == ["mmlu"]


def test_incomparable_rows_excluded_from_union_but_do_not_fail_it():
    report = analyze_benchmarks({
        "math500": (0.72, 0.60),       # comparable win
        "gpqa": (0.4, None),           # incomparable -> excluded
    })
    assert report["r11_holds"] is True
    assert report["union_trained"] == pytest.approx(0.72)   # only the comparable row
    assert "gpqa" not in report["violations"]


def test_all_incomparable_does_not_hold():
    report = analyze_benchmarks({"math500": (None, None)})
    assert report["r11_holds"] is False
    assert report["union_trained"] == 0.0


def test_accepts_mapping_and_alias_keys():
    report = analyze_benchmarks({
        "math500": {"trained": 0.72, "llm_as_coordinator": 0.60},
        "mmlu": {"trinity": 0.68, "llm": 0.55},     # aliases
    })
    assert report["r11_holds"] is True


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_reports_holds_and_the_union_line():
    md = render({"math500": (0.72, 0.60), "livecodebench": (0.615, 0.5376)})
    assert "R11 (trained coordinator > LLM-as-coordinator): HOLDS" in md
    assert "livecodebench" in md and "math500" in md
    assert "| benchmark | trained | llm-as-coord | margin | R11 |" in md


def test_render_flags_violation_and_incomparable_rows():
    md = render({"mmlu": (0.50, 0.55), "gpqa": ("?", 0.5)})
    assert "VIOLATED" in md
    assert "violations: mmlu" in md
    assert "| gpqa | - | - | - | n/a |" in md
