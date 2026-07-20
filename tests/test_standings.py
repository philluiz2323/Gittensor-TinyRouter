"""Offline tests for the cross-benchmark competition standings.

The competition is composite: each merged win lives in ``competition.history`` with a
``per_benchmark`` score breakdown. The headline case is that a miner strong on every
benchmark outranks one who is lopsided or skips a board. stdlib only, no torch/network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.standings import compute_standings, load_standings, render

_REPO = Path(__file__).resolve().parents[1]


def _win(miner, per_benchmark, pr, merged=True, ts="2026-07-13T00:00:00Z"):
    """A composite win entry (score = mean of the per-benchmark breakdown)."""
    score = round(sum(per_benchmark.values()) / len(per_benchmark), 4) if per_benchmark else 0.0
    return {"miner": miner, "generation": 1, "score": score,
            "per_benchmark": dict(per_benchmark), "pr": pr, "merged": merged, "timestamp": ts}


def _lb(history, benchmarks=("math500", "mmlu")):
    """A leaderboard whose composite competition carries ``history``."""
    return {"updated_at": "2026-07-13T00:00:00Z", "benchmarks": {}, "competition": {
        "benchmarks": list(benchmarks), "win_margin": 0.02, "best_composite_score": 0.0,
        "best_miner": None, "history": history}}


def test_module_imports_without_torch():
    # A global sys.modules check is unreliable where torch IS installed (CI): another
    # test imports it into the shared process before this one runs. Verify in a CLEAN
    # subprocess that importing this module alone never pulls in torch.
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.standings; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# reads the composite competition history (regression: it used to read the
# always-empty benchmarks.*.history subtree and rank nobody forever)
# --------------------------------------------------------------------------- #
def test_ranks_miners_from_competition_history():
    lb = _lb([
        _win("alice", {"math500": 0.80, "mmlu": 0.90}, 1),   # mean 0.85
        _win("bob", {"math500": 0.85}, 2),                   # only math500 -> mmlu counts 0
        _win("carol", {"math500": 0.60, "mmlu": 0.60}, 3),   # mean 0.60
    ])
    s = compute_standings(lb)
    assert [m.miner for m in s.miners] == ["alice", "carol", "bob"]
    assert s.leader == "alice"
    assert [m.overall for m in s.miners] == pytest.approx([0.85, 0.60, 0.425])
    # bob tops math500 (0.85); alice tops mmlu (0.90); carol leads nothing.
    assert s.miners[0].benchmarks_led == 1               # alice leads mmlu


def test_a_populated_competition_no_longer_ranks_nobody():
    # The exact production shape after a composite win: competition.history populated,
    # benchmarks.*.history empty. Pre-fix this returned leader=None.
    lb = {
        "competition": {"benchmarks": ["math500", "mmlu", "livecodebench"],
                        "best_miner": "alice", "best_composite_score": 0.85,
                        "history": [_win("alice", {"math500": 0.9, "mmlu": 0.9,
                                                   "livecodebench": 0.75}, 42)]},
        "benchmarks": {"math500": {"best_score": 0.0, "history": []},
                       "mmlu": {"best_score": 0.0, "history": []},
                       "livecodebench": {"best_score": 0.0, "history": []}},
    }
    s = compute_standings(lb)
    assert s.leader == "alice" and s.miners[0].overall == pytest.approx((0.9 + 0.9 + 0.75) / 3)


def test_only_merged_wins_count():
    lb = _lb([_win("bob", {"math500": 0.99, "mmlu": 0.99}, 1, merged=False),
              _win("bob", {"math500": 0.60, "mmlu": 0.60}, 2)])
    s = compute_standings(lb)
    assert s.miners[0].per_benchmark == {"math500": 0.60, "mmlu": 0.60}  # unmerged ignored


def test_best_of_multiple_wins_per_miner():
    lb = _lb([_win("bob", {"math500": 0.60, "mmlu": 0.50}, 1),
              _win("bob", {"math500": 0.75, "mmlu": 0.40}, 2)])
    s = compute_standings(lb)
    assert s.miners[0].per_benchmark == {"math500": 0.75, "mmlu": 0.50}  # max per board


def test_missing_benchmark_counts_as_zero_and_reports_competed():
    lb = _lb([_win("solo", {"math500": 0.80}, 1)])          # only scored math500
    m = compute_standings(lb).miners[0]
    assert m.per_benchmark == {"math500": 0.80}
    assert m.overall == pytest.approx(0.40) and m.n_competed == 1   # 0.80 on 1 of 2 boards


def test_ranking_tiebreak_by_benchmarks_led():
    # all three tie on overall (0.50); the two who UNIQUELY lead a board rank above the
    # one who leads none.
    lb = _lb([_win("x", {"math500": 0.60, "mmlu": 0.40}, 1),
              _win("y", {"math500": 0.40, "mmlu": 0.60}, 2),
              _win("z", {"math500": 0.50, "mmlu": 0.50}, 3)])
    s = compute_standings(lb)
    assert [m.overall for m in s.miners] == pytest.approx([0.50, 0.50, 0.50])
    assert [m.miner for m in s.miners] == ["x", "y", "z"]
    assert s.miners[0].benchmarks_led == 1 and s.miners[2].benchmarks_led == 0


def test_a_tie_for_a_board_top_credits_no_leader():
    lb = _lb([_win("x", {"math500": 0.50, "mmlu": 0.30}, 1),
              _win("y", {"math500": 0.50, "mmlu": 0.20}, 2)])
    s = compute_standings(lb)                              # both top math500 at 0.50
    assert all(m.benchmarks_led == 0 for m in s.miners if m.miner in {"x", "y"}) or \
        s.miners[0].benchmarks_led == 1                    # only the unique mmlu top (x) can lead
    # x uniquely tops mmlu (0.30 > 0.20) so x leads exactly one board; math500 tie -> no leader.
    x = next(m for m in s.miners if m.miner == "x")
    y = next(m for m in s.miners if m.miner == "y")
    assert x.benchmarks_led == 1 and y.benchmarks_led == 0


# --------------------------------------------------------------------------- #
# seed / robustness
# --------------------------------------------------------------------------- #
def test_seed_competition_has_no_miners():
    s = compute_standings(_lb([]))
    assert s.benchmarks == ["math500", "mmlu"] and s.miners == [] and s.leader is None


def test_missing_or_non_dict_competition_is_empty():
    assert compute_standings({"competition": 7}).miners == []
    assert compute_standings({}).benchmarks == [] and compute_standings({}).miners == []
    # a benchmarks-only leaderboard with no competition ranks nobody (no crash).
    assert compute_standings({"benchmarks": {"math500": {"history": []}}}).miners == []


def test_tampered_history_does_not_crash():
    assert compute_standings(_lb(5)).miners == []           # non-list history
    lb = _lb([7, _win("z", {"mmlu": 0.5}, 1)])              # non-dict entry mixed in
    s = compute_standings(lb)                               # must not raise
    assert s.miners[0].miner == "z" and s.miners[0].per_benchmark == {"mmlu": 0.5}


def test_non_dict_per_benchmark_is_skipped():
    lb = _lb([{"miner": "q", "merged": True, "per_benchmark": "oops", "pr": 1,
               "score": 0.5, "generation": 1, "timestamp": "2026-07-13T00:00:00Z"},
              _win("r", {"math500": 0.7}, 2)])
    s = compute_standings(lb)                               # q dropped, r kept
    assert [m.miner for m in s.miners] == ["r"]


# --------------------------------------------------------------------------- #
# load + render
# --------------------------------------------------------------------------- #
def test_load_standings_from_file(tmp_path):
    p = tmp_path / "leaderboard.json"
    p.write_text(json.dumps(_lb([_win("alice", {"math500": 0.8, "mmlu": 0.9}, 1)])))
    s = load_standings(p)
    assert s.leader == "alice" and s.miners[0].overall == pytest.approx(0.85)
    # missing file -> empty, never raises
    assert load_standings(tmp_path / "nope.json").miners == []


def test_render_table_and_empty():
    lb = _lb([_win("alice", {"math500": 0.8, "mmlu": 0.9}, 1)])
    md = render(compute_standings(lb))
    assert "competition standings" in md.lower() and "equal-weighted" in md
    assert "| 1 | alice |" in md and "overall leader:** alice" in md
    empty = render(compute_standings(_lb([])))
    assert empty.strip().endswith("(no miners have won a benchmark yet)_")


def test_to_dict_roundtrips_json():
    lb = _lb([_win("alice", {"math500": 0.8}, 1)])
    d = compute_standings(lb).to_dict()
    assert json.loads(json.dumps(d))["leader"] == "alice"
    assert d["miners"][0]["per_benchmark"] == {"math500": 0.8}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
