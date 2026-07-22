"""Offline tests for the R1/R2 (TRINITY > single models) verifier.

No network, no GPU, no torch — plain numbers only.
"""
from __future__ import annotations

import pytest

from trinity.analysis.single_model_dominance import (
    analyze_task,
    analyze_tasks,
    render,
)


# --------------------------------------------------------------------------- #
# analyze_task (R2, per task)
# --------------------------------------------------------------------------- #
def test_trinity_beats_every_model_dominates():
    r = analyze_task(0.90, {"gpt5": 0.85, "gemini": 0.80}, task="math500")
    assert r.comparable and r.dominates
    assert r.best_model == "gpt5" and r.best_model_score == 0.85
    assert r.margin == pytest.approx(0.05)


def test_losing_to_one_model_is_not_domination():
    r = analyze_task(0.83, {"gpt5": 0.85, "gemini": 0.80})
    assert r.comparable and not r.dominates
    assert r.best_model == "gpt5"
    assert r.margin == pytest.approx(-0.02)


def test_tie_with_best_model_is_not_a_win():
    r = analyze_task(0.85, {"gpt5": 0.85, "gemini": 0.80})
    assert r.comparable and not r.dominates
    assert r.margin == 0.0


def test_non_numeric_models_ignored_and_missing_trinity_incomparable():
    r1 = analyze_task(0.9, {"gpt5": "n/a", "gemini": 0.8})
    assert r1.comparable and r1.best_model == "gemini"
    assert not analyze_task(None, {"gpt5": 0.8}).comparable
    assert not analyze_task(0.9, {}).comparable


# --------------------------------------------------------------------------- #
# analyze_tasks: R2 union + R1 average
# --------------------------------------------------------------------------- #
def test_r2_and_r1_both_hold():
    report = analyze_tasks({
        "math500": (0.90, {"gpt5": 0.85, "gemini": 0.80}),
        "mmlu": (0.92, {"gpt5": 0.88, "gemini": 0.86}),
    })
    assert report["r2_holds"] is True and report["r2_violations"] == []
    # best single model by average: gpt5 = (0.85+0.88)/2 = 0.865
    assert report["best_single_model"] == "gpt5"
    assert report["best_single_avg"] == pytest.approx(0.865)
    assert report["trinity_avg"] == pytest.approx((0.90 + 0.92) / 2)
    assert report["r1_holds"] is True
    assert report["r1r2_holds"] is True


def test_r2_can_fail_while_r1_still_holds():
    # TRINITY loses to a model on livecodebench (R2 violated) but still wins on average (R1).
    report = analyze_tasks({
        "math500": (0.95, {"gpt5": 0.70}),
        "livecodebench": (0.60, {"gpt5": 0.65}),   # R2 violation here
    })
    assert report["r2_holds"] is False
    assert report["r2_violations"] == ["livecodebench"]
    # trinity avg 0.775 > gpt5 avg 0.675 -> R1 holds
    assert report["r1_holds"] is True
    assert report["r1r2_holds"] is False           # combined requires both


def test_r1_fails_when_a_model_wins_on_average():
    report = analyze_tasks({
        "math500": (0.60, {"gpt5": 0.90}),
        "mmlu": (0.62, {"gpt5": 0.88}),
    })
    assert report["r1_holds"] is False
    assert report["best_single_model"] == "gpt5"


def test_incomparable_tasks_excluded():
    report = analyze_tasks({
        "math500": (0.90, {"gpt5": 0.85}),
        "gpqa": (0.5, {}),                         # no models -> excluded
    })
    assert report["r2_holds"] is True
    assert report["trinity_avg"] == pytest.approx(0.90)
    assert "gpqa" not in report["r2_violations"]


def test_all_incomparable_holds_nothing():
    report = analyze_tasks({"math500": (0.9, {})})
    assert report["r1_holds"] is False and report["r2_holds"] is False
    assert report["best_single_model"] is None


def test_accepts_mapping_form():
    report = analyze_tasks({
        "math500": {"trinity": 0.90, "singles": {"gpt5": 0.85}},
    })
    assert report["r1r2_holds"] is True


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_reports_both_invariants():
    md = render({"math500": (0.90, {"gpt5": 0.85, "gemini": 0.80})})
    assert "R2 (TRINITY > every single model on every task): HOLDS" in md
    assert "R1 (TRINITY avg > best single model avg): HOLDS" in md
    assert "gpt5 0.850" in md


def test_render_flags_r2_violation():
    md = render({"lcb": (0.60, {"gpt5": 0.65}), "gpqa": (0.5, {})})
    assert "R2 (TRINITY > every single model on every task): VIOLATED" in md
    assert "violations: lcb" in md
    assert "| gpqa | - | - | - | n/a |" in md
