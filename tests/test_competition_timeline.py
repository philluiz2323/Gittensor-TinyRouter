"""Offline tests for the competition king-progression timeline. No network, no GPU."""
from __future__ import annotations

import json

import pytest

from trinity.competition_timeline import build_timeline, load_timeline, render

_T0, _T1, _T2 = ("2026-07-05T00:00:00Z", "2026-07-05T01:00:00Z", "2026-07-05T02:00:00Z")


def _reign(miner, composite, pr, per_benchmark=None, gen=1, ts=_T0):
    return {"miner": miner, "generation": gen, "score": composite,
            "per_benchmark": per_benchmark or {}, "pr": pr, "merged": True, "timestamp": ts}


def _lb(history, **comp_over):
    comp = {"benchmarks": ["math500", "mmlu"], "win_margin": 0.02, "history": history}
    comp.update(comp_over)
    # load_leaderboard keeps the record only when a top-level benchmarks dict exists (the
    # real file always has both trees), so include it for the load-from-file path.
    return {"benchmarks": {}, "competition": comp}


def test_progression_gains_are_measured_over_the_previous_king():
    t = build_timeline(_lb([
        _reign("alice", 0.60, 1, {"math500": 0.6, "mmlu": 0.6}, ts=_T0),
        _reign("bob", 0.85, 2, {"math500": 0.9, "mmlu": 0.8}, ts=_T1),
        _reign("carol", 0.90, 3, {"math500": 0.95, "mmlu": 0.85}, ts=_T2),
    ]))
    assert t.n_crownings == 3
    assert [r.miner for r in t.reigns] == ["alice", "bob", "carol"]
    # first crown measured against the 0.0 seed floor; then over the previous king.
    assert [round(r.gain_over_prev, 4) for r in t.reigns] == [0.60, 0.25, 0.05]
    assert t.current_king == "carol" and t.current_composite == 0.90
    assert t.total_gain == pytest.approx(0.30)          # 0.90 - 0.60 (first -> now)


def test_biggest_leap_is_the_largest_gain():
    # A later king can hold the biggest single advance: bob's +0.48 beats alice's opening
    # +0.30 (over the 0.0 seed) and carol's +0.02.
    t = build_timeline(_lb([
        _reign("alice", 0.30, 1), _reign("bob", 0.78, 2), _reign("carol", 0.80, 3),
    ]))
    assert t.biggest_leap is not None
    assert t.biggest_leap.miner == "bob" and t.biggest_leap.order == 2
    assert t.biggest_leap.gain_over_prev == pytest.approx(0.48)


def test_per_benchmark_breakdown_is_kept():
    t = build_timeline(_lb([_reign("alice", 0.85, 1, {"math500": 0.9, "mmlu": 0.8})]))
    assert t.reigns[0].per_benchmark == {"math500": 0.9, "mmlu": 0.8}


def test_empty_history_is_no_crownings():
    t = build_timeline(_lb([]))
    assert t.n_crownings == 0 and t.current_king is None and t.reigns == []
    assert t.total_gain == 0.0 and t.biggest_leap is None


def test_no_competition_object_is_empty():
    assert build_timeline({"benchmarks": {}}).n_crownings == 0
    assert build_timeline({}).n_crownings == 0
    assert build_timeline({"competition": 7}).n_crownings == 0


def test_entries_missing_miner_or_score_are_skipped():
    t = build_timeline(_lb([
        {"generation": 1, "score": 0.5, "pr": 1},          # no miner
        {"miner": "x", "pr": 2},                           # no score
        _reign("y", 0.7, 3),
    ]))
    assert [r.miner for r in t.reigns] == ["y"] and t.reigns[0].order == 1


def test_non_list_history_and_non_dict_entries_do_not_crash():
    assert build_timeline(_lb(5)).n_crownings == 0                    # non-list history
    t = build_timeline(_lb([7, _reign("z", 0.6, 1)]))                # non-dict entry mixed in
    assert [r.miner for r in t.reigns] == ["z"]


def test_string_score_is_coerced():
    t = build_timeline(_lb([{"miner": "a", "score": "0.7", "pr": 1, "generation": 1}]))
    assert t.reigns[0].composite == pytest.approx(0.7)


def test_render_and_to_dict():
    lb = _lb([_reign("alice", 0.6, 1), _reign("bob", 0.85, 2)])
    t = build_timeline(lb)
    md = render(t)
    assert "king progression" in md.lower()
    assert "| 1 | alice" in md and "| 2 | bob" in md and "current king:** bob" in md
    d = t.to_dict()
    assert json.loads(json.dumps(d))["current_king"] == "bob"
    assert d["reigns"][1]["gain_over_prev"] == pytest.approx(0.25)
    # empty render
    assert render(build_timeline(_lb([]))).strip().endswith("(no one has been crowned yet)_")


def test_load_timeline_from_file(tmp_path):
    p = tmp_path / "leaderboard.json"
    p.write_text(json.dumps(_lb([_reign("alice", 0.8, 1)])))
    assert load_timeline(p).current_king == "alice"
    assert load_timeline(tmp_path / "nope.json").n_crownings == 0   # missing file, no raise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
