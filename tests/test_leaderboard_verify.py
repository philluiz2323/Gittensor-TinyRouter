"""Offline tests for the leaderboard integrity verifier (scripts/verify_leaderboard.py
core in trinity.submission.leaderboard).

Synthetic leaderboards, no network/torch. Covers the tamper-detection invariants (best_*
vs winning history, score-above-oracle, truncated attempts ledger, duplicate/mis-owned
PRs, non-monotone timestamps, stale updated_at) and the report renderer.
"""
import copy
import sys

import pytest

from trinity.submission.leaderboard import (
    headroom_captured,
    leaderboard_report,
    verify_leaderboard,
)

_T0 = "2026-07-05T00:00:00Z"
_T1 = "2026-07-05T01:00:00Z"


def _clean() -> dict:
    """A well-formed single-benchmark leaderboard with one recorded win."""
    return {
        "updated_at": "2026-07-05T02:00:00Z",
        "benchmarks": {
            "math500": {
                "best_score": 0.85,
                "best_miner": "alice",
                "best_generation": 3,
                "best_pr": 42,
                "baseline_random": 0.79,
                "best_single_model": 0.82,
                "oracle_ceiling": 0.90,
                "history": [
                    {"miner": "alice", "generation": 3, "score": 0.85, "pr": 42,
                     "merged": True, "timestamp": _T0},
                ],
                "attempts": [
                    {"miner": "alice", "generation": 3, "pr": 42, "timestamp": _T0},
                ],
            }
        },
    }


def test_no_torch_imported():
    assert "torch" not in sys.modules


def test_clean_leaderboard_has_no_problems():
    assert verify_leaderboard(_clean()) == []


def test_seed_state_empty_history_is_clean():
    # The committed seed: null leader, empty history, no attempts ledger.
    lb = {"updated_at": _T0, "benchmarks": {"mmlu": {
        "best_score": 0.922, "best_miner": None, "best_generation": 0, "best_pr": None,
        "baseline_random": 0.875, "best_single_model": 0.922, "oracle_ceiling": 0.939,
        "history": []}}}
    assert verify_leaderboard(lb) == []


# --------------------------------------------------------------------------- #
# Tamper detection
# --------------------------------------------------------------------------- #
def test_best_score_bumped_without_matching_win():
    lb = _clean()
    lb["benchmarks"]["math500"]["best_score"] = 0.88   # history max is still 0.85
    assert any("!= max history score" in p for p in verify_leaderboard(lb))


def test_best_score_above_oracle_is_impossible():
    lb = _clean()
    m = lb["benchmarks"]["math500"]
    m["best_score"] = 0.95
    m["history"][0]["score"] = 0.95   # isolate the oracle-impossibility check
    assert any("exceeds oracle_ceiling" in p for p in verify_leaderboard(lb))


def test_best_pointer_must_match_winning_history_entry():
    lb = _clean()
    lb["benchmarks"]["math500"]["best_miner"] = "bob"
    assert any("best_miner" in p and "winning history" in p for p in verify_leaderboard(lb))


def test_history_entry_must_be_merged():
    lb = _clean()
    lb["benchmarks"]["math500"]["history"][0]["merged"] = False
    assert any("not merged=true" in p for p in verify_leaderboard(lb))


def test_truncated_attempts_ledger_flags_rate_limit_bypass():
    lb = _clean()
    lb["benchmarks"]["math500"]["attempts"] = []   # win no longer has a recorded attempt
    assert any("missing from attempts ledger" in p for p in verify_leaderboard(lb))


def test_duplicate_attempt_pair():
    lb = _clean()
    att = lb["benchmarks"]["math500"]["attempts"]
    att.append(dict(att[0]))   # same (miner, pr) twice
    assert any("duplicate (miner, pr)" in p for p in verify_leaderboard(lb))


def test_pr_owned_by_two_miners():
    lb = _clean()
    lb["benchmarks"]["math500"]["attempts"].append(
        {"miner": "bob", "generation": 1, "pr": 42, "timestamp": _T1})   # pr 42 is alice's
    assert any("claimed by both" in p for p in verify_leaderboard(lb))


def test_backwards_history_timestamp():
    lb = _clean()
    h = lb["benchmarks"]["math500"]["history"]
    h.append({"miner": "alice", "generation": 4, "score": 0.80, "pr": 43,
              "merged": True, "timestamp": "2026-07-01T00:00:00Z"})   # earlier than _T0
    lb["benchmarks"]["math500"]["attempts"].append(
        {"miner": "alice", "generation": 4, "pr": 43, "timestamp": "2026-07-01T00:00:00Z"})
    assert any("goes backwards" in p for p in verify_leaderboard(lb))


def test_stale_updated_at():
    lb = _clean()
    lb["updated_at"] = "2026-07-01T00:00:00Z"   # older than the _T0 entries
    assert any("older than the newest entry" in p for p in verify_leaderboard(lb))


def test_score_out_of_unit_range():
    lb = _clean()
    lb["benchmarks"]["math500"]["best_score"] = 1.5
    assert any("not a number in [0, 1]" in p for p in verify_leaderboard(lb))


def test_non_object_leaderboard():
    assert verify_leaderboard([]) == ["leaderboard is not a JSON object"]
    assert verify_leaderboard({}) == ["missing or invalid 'benchmarks' object"]


# --------------------------------------------------------------------------- #
# headroom_captured + report
# --------------------------------------------------------------------------- #
def test_headroom_captured_math():
    e = _clean()["benchmarks"]["math500"]
    assert headroom_captured(e) == pytest.approx((0.85 - 0.82) / (0.90 - 0.82))
    assert headroom_captured({"best_single_model": 0.9, "oracle_ceiling": 0.9, "best_score": 0.9}) is None
    assert headroom_captured({"best_score": 0.5}) is None


def test_report_renders_frontier_and_rate_status():
    lb = _clean()
    md = leaderboard_report(lb)
    assert "Leaderboard status" in md and "math500" in md and "alice (#42)" in md
    assert "Rate-limit status" not in md            # omitted without now=
    now = 1783382400.0 + 3 * 86400                  # a few days after _T0
    md2 = leaderboard_report(lb, now=now)
    assert "Rate-limit status" in md2 and "alice: 1" in md2


def test_verify_does_not_mutate_input():
    lb = _clean()
    before = copy.deepcopy(lb)
    verify_leaderboard(lb)
    assert lb == before   # read-only


# --------------------------------------------------------------------------- #
# Robustness: a tampered/corrupt record must be REPORTED, never crash the
# verifier or the live rate-limit gate that share rate_limit_entries. Same
# "report, don't crash" contract as the #201 verify_benchmark fix.
# --------------------------------------------------------------------------- #
from trinity.submission.gates import (  # noqa: E402  (grouped with the robustness tests)
    check_rate_limit,
    rate_limit_entries,
)


@pytest.mark.parametrize("field", ["attempts", "history"])
@pytest.mark.parametrize("bad", [5, True, "x", {"not": "a list"}])
def test_non_list_attempts_or_history_is_reported_not_crash(field, bad):
    lb = _clean()
    lb["benchmarks"]["math500"][field] = bad
    problems = verify_leaderboard(lb)                       # must not raise
    assert any(f"{field} is not a JSON array" in p for p in problems)


def test_non_dict_benchmark_entry_is_reported_not_crash():
    lb = _clean()
    lb["benchmarks"]["math500"] = 7                          # scalar where an object belongs
    assert any("not a JSON object" in p for p in verify_leaderboard(lb))


def test_rate_limit_entries_tolerates_malformed_ledgers():
    assert rate_limit_entries({"attempts": 5}) == []        # non-list attempts
    assert rate_limit_entries({"history": True}) == []      # non-list history fallback
    assert rate_limit_entries(9) == []                      # non-dict bench entry
    # well-formed still passes through unchanged
    row = {"miner": "a", "pr": 1, "timestamp": _T0}
    assert rate_limit_entries({"attempts": [row]}) == [row]
    assert rate_limit_entries({"history": [row]}) == [row]  # falls back to history


@pytest.mark.parametrize("lb", [
    {"benchmarks": {"math500": {"attempts": True}}},        # non-list attempts
    {"benchmarks": {"math500": {"history": 3}}},            # non-list history
    {"benchmarks": {"math500": 9}},                         # non-dict entry
    {"benchmarks": 7},                                      # non-dict benchmarks map
])
def test_check_rate_limit_never_crashes_on_tampered_leaderboard(lb):
    # A corrupt ledger must not crash the anti-cheat gate; it degrades to "no prior
    # attempts" (the integrity verifier is the layer that flags the tampering).
    assert check_rate_limit("miner", "math500", lb) is None
