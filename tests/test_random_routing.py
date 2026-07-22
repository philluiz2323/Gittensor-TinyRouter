"""Offline tests for the R4 (TRINITY > random routing) verifier.

No network, no GPU, no torch — plain numbers only.
"""
from __future__ import annotations

import pytest

from trinity.analysis.random_routing import (
    analyze_benchmark,
    analyze_benchmarks,
    render,
)


# --------------------------------------------------------------------------- #
# analyze_benchmark
# --------------------------------------------------------------------------- #
def test_trinity_above_random_holds():
    c = analyze_benchmark(0.41, 0.32, benchmark="rlpr")   # the SPEC example
    assert c.comparable and c.holds
    assert c.margin == pytest.approx(0.09)


def test_trinity_below_or_equal_does_not_hold():
    below = analyze_benchmark(0.30, 0.32, benchmark="mmlu")
    assert below.comparable and not below.holds
    assert below.margin == pytest.approx(-0.02)

    tie = analyze_benchmark(0.50, 0.50)
    assert tie.comparable and not tie.holds        # a tie is not a win
    assert tie.margin == 0.0


def test_missing_or_non_numeric_is_not_comparable():
    for c in (
        analyze_benchmark(None, 0.32),
        analyze_benchmark(0.41, None),
        analyze_benchmark("n/a", 0.32),
        analyze_benchmark(True, 0.32),             # bool is not an accuracy
    ):
        assert not c.comparable and not c.holds and c.margin is None


# --------------------------------------------------------------------------- #
# analyze_benchmarks: union verdict
# --------------------------------------------------------------------------- #
def test_union_holds_when_every_benchmark_beats_random():
    report = analyze_benchmarks({
        "math500": (0.88, 0.30),
        "mmlu": (0.91, 0.28),
        "rlpr": (0.41, 0.32),
    })
    assert report["r4_holds"] is True
    assert report["violations"] == []
    assert report["union_trinity"] == pytest.approx((0.88 + 0.91 + 0.41) / 3)
    assert report["union_random_routing"] == pytest.approx((0.30 + 0.28 + 0.32) / 3)
    assert report["union_margin"] > 0


def test_union_violated_when_any_benchmark_loses():
    report = analyze_benchmarks({
        "math500": (0.88, 0.30),
        "mmlu": (0.27, 0.28),          # loses to random here
    })
    assert report["r4_holds"] is False
    assert report["violations"] == ["mmlu"]


def test_incomparable_rows_excluded_from_union():
    report = analyze_benchmarks({
        "math500": (0.88, 0.30),
        "gpqa": (0.4, None),           # incomparable -> excluded
    })
    assert report["r4_holds"] is True
    assert report["union_trinity"] == pytest.approx(0.88)
    assert "gpqa" not in report["violations"]


def test_all_incomparable_does_not_hold():
    report = analyze_benchmarks({"math500": (None, None)})
    assert report["r4_holds"] is False
    assert report["union_trinity"] == 0.0


def test_accepts_mapping_and_random_alias():
    report = analyze_benchmarks({
        "math500": {"trinity": 0.88, "random_routing": 0.30},
        "rlpr": {"trinity": 0.41, "random": 0.32},     # alias
    })
    assert report["r4_holds"] is True


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_reports_holds_and_union_line():
    md = render({"rlpr": (0.41, 0.32), "math500": (0.88, 0.30)})
    assert "R4 (TRINITY > random routing): HOLDS" in md
    assert "rlpr" in md and "math500" in md
    assert "| benchmark | trinity | random routing | margin | R4 |" in md


def test_render_flags_violation_and_incomparable():
    md = render({"mmlu": (0.27, 0.28), "gpqa": ("?", 0.3)})
    assert "VIOLATED" in md and "violations: mmlu" in md
    assert "| gpqa | - | - | - | n/a |" in md
