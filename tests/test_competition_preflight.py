"""Offline tests for the submission composite preflight go/no-go. No network, no GPU."""
from __future__ import annotations

import json

import pytest

from trinity.competition_preflight import (
    load_and_preflight,
    preflight_submission,
    render,
)


def _lb(best_composite=0.80, win_margin=0.02, king="alice", best_pb=None):
    return {"benchmarks": {}, "competition": {
        "benchmarks": ["math500", "mmlu", "livecodebench"], "win_margin": win_margin,
        "best_composite_score": best_composite, "best_miner": king,
        "best_per_benchmark": best_pb or {"math500": 0.85, "mmlu": 0.85, "livecodebench": 0.70}}}


def test_a_submission_that_clears_king_plus_margin_would_win():
    r = preflight_submission(_lb(best_composite=0.80, win_margin=0.02),
                             {"math500": 0.90, "mmlu": 0.90, "livecodebench": 0.75})
    assert r is not None
    assert r.composite == pytest.approx(0.85)          # mean of the three
    assert r.score_to_beat == pytest.approx(0.82)      # 0.80 + 0.02
    assert r.would_win and r.gap == pytest.approx(-0.03)


def test_a_submission_below_the_threshold_would_not_win_and_reports_the_gap():
    r = preflight_submission(_lb(best_composite=0.90, win_margin=0.02),
                             {"math500": 0.90, "mmlu": 0.90, "livecodebench": 0.75})
    assert r is not None and not r.would_win
    assert r.composite == pytest.approx(0.85) and r.score_to_beat == pytest.approx(0.92)
    assert r.gap == pytest.approx(0.07)                # 0.92 - 0.85 to close


def test_exact_threshold_wins():
    # composite exactly == king + margin clears it (pr_eval uses >=).
    r = preflight_submission(_lb(best_composite=0.80, win_margin=0.02),
                             {"math500": 0.82, "mmlu": 0.82, "livecodebench": 0.82})
    assert r is not None and r.composite == pytest.approx(0.82) and r.would_win


def test_missing_benchmark_counts_as_zero_and_is_flagged():
    r = preflight_submission(_lb(), {"math500": 0.9, "mmlu": 0.9})   # livecodebench absent
    assert r is not None
    assert r.my_scores["livecodebench"] == 0.0
    assert r.missing_benchmarks == ["livecodebench"]
    assert r.composite == pytest.approx((0.9 + 0.9 + 0.0) / 3)


def test_per_benchmark_deltas_vs_king_and_weakest_board():
    r = preflight_submission(
        _lb(best_pb={"math500": 0.85, "mmlu": 0.85, "livecodebench": 0.70}),
        {"math500": 0.90, "mmlu": 0.80, "livecodebench": 0.60})
    assert r is not None
    assert r.vs_king == {"math500": pytest.approx(0.05), "mmlu": pytest.approx(-0.05),
                         "livecodebench": pytest.approx(-0.10)}
    assert r.weakest_benchmark == "livecodebench"      # my lowest board


def test_no_competition_record_returns_none():
    assert preflight_submission({"benchmarks": {}}, {"math500": 0.9}) is None
    assert preflight_submission({}, {"math500": 0.9}) is None


def test_non_numeric_scores_are_dropped():
    r = preflight_submission(_lb(), {"math500": 0.9, "mmlu": "oops", "livecodebench": 0.7})
    assert r is not None and r.my_scores["mmlu"] == 0.0 and "mmlu" in r.missing_benchmarks


def test_falls_back_to_own_boards_when_competition_declares_none():
    lb = {"benchmarks": {}, "competition": {
        "best_composite_score": 0.5, "win_margin": 0.02, "best_miner": None,
        "best_per_benchmark": {}}}                      # no declared benchmarks
    r = preflight_submission(lb, {"math500": 0.8, "mmlu": 0.6})
    assert r is not None and r.benchmarks == ["math500", "mmlu"]
    assert r.composite == pytest.approx(0.70) and r.would_win


def test_render_and_to_dict():
    r = preflight_submission(_lb(best_composite=0.90),
                             {"math500": 0.90, "mmlu": 0.90, "livecodebench": 0.75})
    md = render(r)
    assert "preflight" in md.lower() and "would NOT win" in md and "shortfall" in md
    assert "weakest board" in md
    d = r.to_dict()
    assert json.loads(json.dumps(d))["would_win"] is False
    assert render(None).strip().endswith("(no competition record to compare against)_")


def test_load_and_preflight_from_file(tmp_path):
    p = tmp_path / "leaderboard.json"
    p.write_text(json.dumps(_lb(best_composite=0.80, win_margin=0.02)))
    r = load_and_preflight(p, {"math500": 0.9, "mmlu": 0.9, "livecodebench": 0.9})
    assert r is not None and r.would_win
    assert load_and_preflight(tmp_path / "nope.json", {"math500": 0.9}) is None  # no competition


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
