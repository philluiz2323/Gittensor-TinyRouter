"""Offline tests for the R11 (trained coordinator > LLM-as-coordinator) verifier."""
from __future__ import annotations

import pytest

from trinity.analysis.coordinator_vs_llm import (
    analyze_benchmarks,
    analyze_task,
    render,
)


# ---------------------------------------------------------------------------
# analyze_task
# ---------------------------------------------------------------------------
def test_trinity_wins_on_a_benchmark():
    m = analyze_task("math500", 0.88, 0.70)
    assert m.margin == pytest.approx(0.18)
    assert m.trinity_wins is True


def test_trinity_loses_on_a_benchmark():
    m = analyze_task("mmlu", 0.60, 0.65)
    assert m.margin == pytest.approx(-0.05)
    assert m.trinity_wins is False


def test_a_tie_is_not_a_win():
    m = analyze_task("x", 0.70, 0.70)
    assert m.trinity_wins is False


# ---------------------------------------------------------------------------
# analyze_benchmarks + the R11 verdict
# ---------------------------------------------------------------------------
def _spec_shape():
    # SPEC §6: LLM-as-coordinator avg 53.76 vs TRINITY 70.44 -> TRINITY wins everywhere.
    return {
        "livecodebench": {"trinity": 0.615, "llm_coordinator": 0.52},
        "math500": {"trinity": 0.88, "llm_coordinator": 0.70},
        "mmlu": {"trinity": 0.916, "llm_coordinator": 0.60},
    }


def test_r11_holds_when_trinity_wins_everywhere():
    report = analyze_benchmarks(_spec_shape())
    assert report["n_scored"] == 3 and report["n_wins"] == 3
    assert report["r11_holds"] is True
    assert report["losses"] == []
    assert report["union_margin"] == pytest.approx(((0.615 - 0.52) + 0.18 + 0.316) / 3)


def test_r11_violated_when_trinity_loses_any_benchmark_under_require_all():
    tasks = {
        "a": {"trinity": 0.88, "llm_coordinator": 0.70},   # win
        "b": {"trinity": 0.55, "llm_coordinator": 0.60},   # loss
    }
    report = analyze_benchmarks(tasks)  # require_all default
    assert report["r11_holds"] is False
    assert report["losses"] == ["b"]


def test_union_mode_holds_on_average_despite_one_loss():
    tasks = {
        "a": {"trinity": 0.88, "llm_coordinator": 0.70},   # +0.18
        "b": {"trinity": 0.55, "llm_coordinator": 0.60},   # -0.05
    }
    # require_all -> False; union mean margin = +0.065 > 0 -> holds
    report = analyze_benchmarks(tasks, require_all=False)
    assert report["r11_holds"] is True
    assert report["union_margin"] == pytest.approx(0.065)


def test_llm_key_aliases_and_skipping():
    tasks = {
        "ok": {"trinity_accuracy": 0.7, "llm": 0.6},
        "also_ok": {"trinity": 0.7, "llm_coordinator_accuracy": 0.6},
        "missing": {"trinity": 0.7},
        "bad": {"trinity": "x", "llm": 0.5},
    }
    report = analyze_benchmarks(tasks)
    assert [r["benchmark"] for r in report["per_benchmark"]] == ["also_ok", "ok"]
    assert report["r11_holds"] is True


def test_empty_input_does_not_hold():
    assert analyze_benchmarks({})["r11_holds"] is False


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def test_render_reports_verdict_and_losses():
    tasks = {
        "a": {"trinity": 0.88, "llm_coordinator": 0.70},
        "b": {"trinity": 0.55, "llm_coordinator": 0.60},
    }
    md = render(tasks)  # require_all -> VIOLATED
    assert "R11 (trained coordinator > LLM-as-coordinator): VIOLATED" in md
    assert "won 1/2 benchmarks" in md
    assert "did not beat the LLM coordinator on: b" in md


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
