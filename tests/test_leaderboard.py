"""Offline tests for the leaderboard target reader. No network, no GPU."""
from __future__ import annotations

import json

from trinity.leaderboard import load_leaderboard, summarize_targets

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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
