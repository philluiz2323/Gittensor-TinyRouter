"""Offline tests for the leaderboard target reader. No network, no GPU."""
from __future__ import annotations

import json

import pytest

from trinity.leaderboard import (
    load_leaderboard,
    summarize_competition,
    summarize_targets,
)

_REPO_LB = __import__("pathlib").Path(__file__).resolve().parents[1] / "leaderboard.json"


def _lb(**benches):
    return {"benchmarks": benches}


def _entry(**over):
    e = {"best_score": 0.82, "best_single_model": 0.80, "oracle_ceiling": 0.86,
         "baseline_random": 0.79, "best_miner": None, "best_generation": 0, "best_pr": None}
    e.update(over)
    return e


# ---------------------------------------------------------------------------
# targets derived from an entry
# ---------------------------------------------------------------------------
def test_score_to_beat_and_headroom():
    t = summarize_targets(_lb(math500=_entry()))[0]
    assert t.benchmark == "math500"
    assert t.score_to_beat == 0.82
    assert t.headroom == 0.86 - 0.80          # oracle - best_single
    assert t.captured == 0.82 - 0.80          # best - best_single
    assert t.remaining == 0.86 - 0.82         # oracle - best
    assert t.contested                        # remaining > 0


def test_no_king_when_best_miner_is_null():
    t = summarize_targets(_lb(math500=_entry()))[0]
    assert not t.has_king and t.king_miner is None


def test_king_is_reported():
    t = summarize_targets(_lb(mmlu=_entry(best_miner="alice", best_generation=3, best_pr=42)))[0]
    assert t.has_king and t.king_miner == "alice"
    assert t.king_generation == 3 and t.king_pr == 42


def test_no_remaining_headroom_is_uncontested():
    # best already at the oracle -> nothing left to win.
    t = summarize_targets(_lb(x=_entry(best_score=0.86, oracle_ceiling=0.86)))[0]
    assert t.remaining == 0.0 and not t.contested


def test_negative_differences_are_clamped_to_zero():
    # A degenerate entry where best < single or oracle < best must not go negative.
    t = summarize_targets(_lb(x=_entry(best_score=0.70, best_single_model=0.80,
                                       oracle_ceiling=0.60)))[0]
    assert t.captured == 0.0 and t.remaining == 0.0 and t.headroom == 0.0


def test_remaining_is_floored_at_best_single_model():
    # Seed / early state: best_score is still below the strongest single model, so no
    # routing headroom has been claimed. A trivial router already reaches
    # best_single_model, so the headroom still on the table is oracle - best_single,
    # NOT oracle - best_score. captured + remaining must equal headroom.
    t = summarize_targets(_lb(x=_entry(best_score=0.0, best_single_model=0.817,
                                       oracle_ceiling=0.856)))[0]
    assert t.captured == 0.0
    assert t.headroom == 0.856 - 0.817
    assert t.remaining == 0.856 - 0.817          # was 0.856 (oracle - best_score) below the floor
    assert t.captured + t.remaining == t.headroom


def test_no_routing_headroom_seed_is_uncontested():
    # oracle == best_single: a perfect router cannot beat the best single model, so
    # there is nothing left to win even though best_score is still 0 (seed).
    t = summarize_targets(_lb(x=_entry(best_score=0.0, best_single_model=0.92,
                                       oracle_ceiling=0.92)))[0]
    assert t.remaining == 0.0 and not t.contested   # was remaining 0.92, wrongly "contested"


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------
def test_string_numbers_are_coerced():
    t = summarize_targets(_lb(x=_entry(best_score="0.82")))[0]
    assert t.score_to_beat == 0.82


def test_benchmarks_are_sorted():
    names = [t.benchmark for t in summarize_targets(_lb(mmlu=_entry(), aime=_entry()))]
    assert names == ["aime", "mmlu"]


def test_malformed_entry_is_skipped():
    targets = summarize_targets(_lb(good=_entry(), bad="not-a-dict"))
    assert [t.benchmark for t in targets] == ["good"]


def test_empty_leaderboard_yields_no_targets():
    assert summarize_targets({"benchmarks": {}}) == []
    assert summarize_targets({}) == []


# ---------------------------------------------------------------------------
# load_leaderboard
# ---------------------------------------------------------------------------
def test_load_missing_file_returns_empty_shape(tmp_path):
    lb = load_leaderboard(tmp_path / "nope.json")
    assert lb == {"benchmarks": {}}


def test_load_unparseable_file_returns_empty_shape(tmp_path):
    p = tmp_path / "lb.json"
    p.write_text("{not json")
    assert load_leaderboard(p) == {"benchmarks": {}}


def test_load_wrong_shape_returns_empty(tmp_path):
    p = tmp_path / "lb.json"
    p.write_text(json.dumps({"benchmarks": [1, 2, 3]}))
    assert load_leaderboard(p) == {"benchmarks": {}}


# ---------------------------------------------------------------------------
# the real repo leaderboard parses
# ---------------------------------------------------------------------------
def test_the_repo_leaderboard_parses():
    targets = summarize_targets(load_leaderboard(_REPO_LB))
    assert targets, "expected at least one benchmark in the repo leaderboard"
    for t in targets:
        assert t.score_to_beat >= 0.0
        assert t.oracle_ceiling >= t.best_single_model  # by construction
        d = t.to_dict()
        assert d["benchmark"] == t.benchmark and "remaining" in d


# ---------------------------------------------------------------------------
# composite competition target (the score every submission is actually judged on)
# ---------------------------------------------------------------------------
def _comp(**over):
    c = {"benchmarks": ["math500", "mmlu", "livecodebench"], "win_margin": 0.02,
         "best_composite_score": 0.0, "best_miner": None, "best_generation": 0,
         "best_pr": None, "best_per_benchmark": {}, "baseline_best_single": None,
         "baseline_random": None}
    c.update(over)
    return {"benchmarks": {}, "competition": c}


def test_composite_score_to_beat_is_king_plus_margin():
    ct = summarize_competition(_comp(
        best_composite_score=0.85, win_margin=0.02, best_miner="alice",
        best_generation=3, best_pr=42,
        best_per_benchmark={"math500": 0.90, "mmlu": 0.90, "livecodebench": 0.75}))
    assert ct is not None
    assert ct.current_best == 0.85 and ct.win_margin == 0.02
    assert ct.score_to_beat == pytest.approx(0.87)          # 0.85 + 0.02 (the APPROVE threshold)
    assert ct.has_king and ct.king_miner == "alice" and ct.king_pr == 42
    assert ct.best_per_benchmark["livecodebench"] == 0.75
    assert ct.reachable


def test_seed_competition_has_no_king_and_low_target():
    ct = summarize_competition(_comp())                     # seed: best 0.0, no king
    assert ct is not None and not ct.has_king
    assert ct.score_to_beat == pytest.approx(0.02) and ct.king_miner is None


def test_unbeatable_when_target_exceeds_one():
    ct = summarize_competition(_comp(best_composite_score=0.99, win_margin=0.02))
    assert ct is not None and ct.score_to_beat == pytest.approx(1.01)
    assert not ct.reachable                                 # a perfect 1.0 still wouldn't clear it


def test_summarize_competition_none_without_competition():
    assert summarize_competition({"benchmarks": {}}) is None
    assert summarize_competition({}) is None
    assert summarize_competition({"competition": 7}) is None


def test_competition_tolerates_missing_and_string_fields():
    ct = summarize_competition({"competition": {
        "best_composite_score": "0.5", "win_margin": "0.02",
        "best_per_benchmark": {"math500": "0.6", "mmlu": None}}})
    assert ct is not None
    assert ct.score_to_beat == pytest.approx(0.52)
    assert ct.best_per_benchmark == {"math500": 0.6}        # null dropped, string coerced
    assert ct.benchmarks == [] and not ct.has_king


def test_competition_to_dict_roundtrips_json():
    ct = summarize_competition(_comp(best_composite_score=0.7, best_miner="bob"))
    d = ct.to_dict()
    assert json.loads(json.dumps(d))["score_to_beat"] == pytest.approx(0.72)
    assert d["has_king"] is True and d["reachable"] is True


def test_the_repo_leaderboard_composite_target_parses():
    ct = summarize_competition(load_leaderboard(_REPO_LB))
    assert ct is not None                                   # the committed file has a competition
    assert ct.score_to_beat == pytest.approx(ct.current_best + ct.win_margin)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
