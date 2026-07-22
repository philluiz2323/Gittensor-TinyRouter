"""Offline tests for the R13 (relative-error-reduction vs best single agent) verifier.

No network, no GPU, no torch — plain numbers only.
"""
from __future__ import annotations

import pytest

from trinity.analysis.relative_error_reduction import (
    analyze_task,
    analyze_tasks,
    relative_error_reduction,
    render,
)


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #
def test_rer_formula():
    # (0.88 - 0.80) / (1 - 0.80) = 0.4
    assert relative_error_reduction(0.88, 0.80) == pytest.approx(0.4)
    # matching the best single -> 0 reduction
    assert relative_error_reduction(0.80, 0.80) == pytest.approx(0.0)
    # doing worse -> negative
    assert relative_error_reduction(0.75, 0.80) == pytest.approx(-0.25)


# --------------------------------------------------------------------------- #
# analyze_task
# --------------------------------------------------------------------------- #
def test_positive_reduction_holds():
    r = analyze_task(0.88, 0.80, task="math500")
    assert r.comparable and r.holds
    assert r.rer == pytest.approx(0.4)
    assert r.best_single == 0.80


def test_best_single_from_a_model_map():
    r = analyze_task(0.72, {"gpt5": 0.65, "gemini": 0.60}, task="mmlu")
    assert r.best_single == 0.65
    assert r.rer == pytest.approx((0.72 - 0.65) / (1 - 0.65))
    assert r.holds


def test_matching_or_worse_than_best_single_does_not_hold():
    assert analyze_task(0.80, 0.80).holds is False        # tie -> 0 RER
    below = analyze_task(0.70, 0.80)
    assert below.comparable and not below.holds
    assert below.rer == pytest.approx(-0.5)


def test_perfect_single_model_is_incomparable():
    # 1 - best_single == 0 -> no error to reduce; guarded, not a divide-by-zero.
    r = analyze_task(1.0, 1.0)
    assert not r.comparable and r.rer is None


def test_missing_or_non_numeric_is_incomparable():
    assert not analyze_task(None, 0.8).comparable
    assert not analyze_task(0.8, "n/a").comparable
    assert not analyze_task(0.8, {}).comparable           # empty model map


# --------------------------------------------------------------------------- #
# analyze_tasks: mean + verdict
# --------------------------------------------------------------------------- #
def test_mean_rer_and_holds_when_every_task_reduces_error():
    report = analyze_tasks({
        "math500": (0.88, 0.80),      # RER 0.40
        "mmlu": (0.92, 0.90),         # RER 0.20
    })
    assert report["r13_holds"] is True
    assert report["n_tasks_scored"] == 2
    assert report["mean_rer"] == pytest.approx((0.4 + 0.2) / 2)
    assert report["violations"] == []


def test_one_non_reducing_task_violates():
    report = analyze_tasks({
        "math500": (0.88, 0.80),
        "rlpr": (0.40, 0.45),         # worse than best single -> violation
    })
    assert report["r13_holds"] is False
    assert report["violations"] == ["rlpr"]


def test_incomparable_tasks_excluded_from_mean():
    report = analyze_tasks({
        "math500": (0.88, 0.80),
        "perfect": (1.0, 1.0),        # incomparable, excluded
    })
    assert report["r13_holds"] is True
    assert report["mean_rer"] == pytest.approx(0.4)


def test_all_incomparable_does_not_hold():
    report = analyze_tasks({"perfect": (1.0, 1.0)})
    assert report["r13_holds"] is False
    assert report["mean_rer"] == 0.0


def test_accepts_mapping_with_singles_alias():
    report = analyze_tasks({
        "mmlu": {"trinity": 0.72, "singles": {"gpt5": 0.65, "gemini": 0.60}},
    })
    assert report["r13_holds"] is True


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_reports_percent_and_verdict():
    md = render({"math500": (0.88, 0.80)})
    assert "R13 (relative-error-reduction > 0 vs best single): HOLDS" in md
    assert "+40.0%" in md
    assert "ballpark ~21.9%" in md


def test_render_flags_violation_and_incomparable():
    md = render({"rlpr": (0.40, 0.45), "perfect": (1.0, 1.0)})
    assert "VIOLATED" in md and "violations: rlpr" in md
    assert "| perfect | - | - | - | n/a |" in md
