"""Offline tests for the ``competition`` king-record integrity checks in
``trinity.submission.leaderboard.verify_leaderboard``.

The ``competition`` object holds ``best_composite_score`` — the single number every
``pr_eval`` APPROVE/REJECT is compared against — plus the ``best_*`` king pointers and a
composite ``history``. These invariants are grounded in ``pr_eval._update_leaderboard``
and its APPROVE rule (composite = mean of per-benchmark scores; a new king clears the
previous by ``win_margin``). Synthetic dicts only — no network / torch.
"""
import copy

import pytest

from trinity.submission.leaderboard import verify_leaderboard

_T0 = "2026-07-05T00:00:00Z"
_T1 = "2026-07-05T01:00:00Z"
_T2 = "2026-07-05T02:00:00Z"


def _clean() -> dict:
    """A well-formed leaderboard with a populated competition king + benchmarks scaffold.

    Two crowned kings: alice 0.60 then bob 0.85 (clears the 0.02 win_margin). The composite
    equals the mean of the per-benchmark scores, and each win has a recorded attempt under
    ``benchmarks.composite``.
    """
    return {
        "updated_at": "2026-07-05T03:00:00Z",
        "competition": {
            "benchmarks": ["math500", "mmlu", "livecodebench"],
            "win_margin": 0.02,
            "best_composite_score": 0.85,
            "best_miner": "bob",
            "best_generation": 4,
            "best_pr": 43,
            "best_per_benchmark": {"math500": 0.90, "mmlu": 0.90, "livecodebench": 0.75},
            "baseline_best_single": 0.70,
            "baseline_random": 0.50,
            "history": [
                {"miner": "alice", "generation": 2, "score": 0.60,
                 "per_benchmark": {"math500": 0.60, "mmlu": 0.60, "livecodebench": 0.60},
                 "pr": 42, "merged": True, "timestamp": _T0},
                {"miner": "bob", "generation": 4, "score": 0.85,
                 "per_benchmark": {"math500": 0.90, "mmlu": 0.90, "livecodebench": 0.75},
                 "pr": 43, "merged": True, "timestamp": _T1},
            ],
        },
        # benchmarks.composite is the rate-limit ledger pr_eval._record_attempt writes
        # (shape of _empty_bench_entry() + the attempts), where the composite attempts live.
        "benchmarks": {
            "composite": {
                "best_score": 0.0, "best_miner": None, "best_generation": 0,
                "best_pr": None, "baseline_random": None, "best_single_model": None,
                "oracle_ceiling": None, "history": [],
                "attempts": [
                    {"miner": "alice", "generation": 2, "pr": 42, "timestamp": _T0},
                    {"miner": "bob", "generation": 4, "pr": 43, "timestamp": _T1},
                ],
            },
        },
    }


def test_clean_competition_has_no_problems():
    assert verify_leaderboard(_clean()) == []


def test_seed_competition_unclaimed_is_clean():
    # The committed seed: composite 0.0, null king, empty history, empty per-benchmark.
    lb = {"updated_at": _T0, "benchmarks": {}, "competition": {
        "benchmarks": ["math500", "mmlu", "livecodebench"], "win_margin": 0.02,
        "best_composite_score": 0.0, "best_miner": None, "best_generation": 0,
        "best_pr": None, "best_per_benchmark": {}, "baseline_best_single": None,
        "baseline_random": None, "history": []}}
    assert verify_leaderboard(lb) == []


def test_a_leaderboard_without_competition_is_unaffected():
    # Backward-compat: benchmarks-only leaderboards must still verify clean.
    lb = {"updated_at": _T0, "benchmarks": {"math500": {
        "best_score": 0.85, "best_miner": None, "best_generation": 0, "best_pr": None,
        "history": []}}}
    assert verify_leaderboard(lb) == []


# --------------------------------------------------------------------------- #
# The headline anti-cheat: an inflated best_composite_score
# --------------------------------------------------------------------------- #
def test_inflated_composite_that_beats_its_own_per_benchmark_mean():
    lb = _clean()
    # Set the king score to 0.99 while the per-benchmark row (and history) stay at 0.85.
    lb["competition"]["best_composite_score"] = 0.99
    probs = verify_leaderboard(lb)
    assert any("best_composite_score" in p and "mean(best_per_benchmark)" in p for p in probs)


def test_composite_not_matching_the_winning_history_score():
    lb = _clean()
    # Make per-benchmark mean agree with 0.99 but leave history max at 0.85.
    lb["competition"]["best_composite_score"] = 0.99
    lb["competition"]["best_per_benchmark"] = {"math500": 0.99, "mmlu": 0.99, "livecodebench": 0.99}
    assert any("!= max history score" in p for p in verify_leaderboard(lb))


def test_composite_out_of_unit_range():
    lb = _clean()
    lb["competition"]["best_composite_score"] = 1.5
    assert any("best_composite_score" in p and "not a number in [0, 1]" in p
               for p in verify_leaderboard(lb))


# --------------------------------------------------------------------------- #
# King-pointer consistency
# --------------------------------------------------------------------------- #
def test_best_pointer_must_match_winning_history_entry():
    lb = _clean()
    lb["competition"]["best_miner"] = "mallory"
    assert any("competition.best_miner" in p and "winning history" in p
               for p in verify_leaderboard(lb))


def test_best_king_set_but_history_empty():
    lb = _clean()
    lb["competition"]["history"] = []
    assert any("set but history is empty" in p for p in verify_leaderboard(lb))


# --------------------------------------------------------------------------- #
# win_margin: a king that didn't clear the margin never legitimately crowned
# --------------------------------------------------------------------------- #
def test_king_that_does_not_clear_win_margin_is_flagged():
    lb = _clean()
    h = lb["competition"]["history"]
    # bob only reaches 0.61 (gain 0.01 < 0.02 margin over alice's 0.60); realign pointers
    # so ONLY the margin rule can fire.
    h[1]["score"] = 0.61
    h[1]["per_benchmark"] = {"math500": 0.61, "mmlu": 0.61, "livecodebench": 0.61}
    lb["competition"]["best_composite_score"] = 0.61
    lb["competition"]["best_per_benchmark"] = {"math500": 0.61, "mmlu": 0.61, "livecodebench": 0.61}
    assert any("does not clear the previous king" in p for p in verify_leaderboard(lb))


def test_history_entry_must_be_merged():
    lb = _clean()
    lb["competition"]["history"][1]["merged"] = False
    assert any("competition.history[1] is not merged=true" in p for p in verify_leaderboard(lb))


def test_backwards_history_timestamp():
    lb = _clean()
    lb["competition"]["history"][1]["timestamp"] = "2026-07-01T00:00:00Z"  # before alice
    assert any("competition.history[1] timestamp goes backwards" in p
               for p in verify_leaderboard(lb))


# --------------------------------------------------------------------------- #
# Cross-subtree: a crowned win with no recorded rate-limit attempt
# --------------------------------------------------------------------------- #
def test_crowned_win_missing_from_attempts_ledger_is_rate_limit_bypass():
    lb = _clean()
    # Drop bob's attempt: his crowning has no recorded submission.
    lb["benchmarks"]["composite"]["attempts"] = [
        {"miner": "alice", "generation": 2, "pr": 42, "timestamp": _T0},
    ]
    assert any("missing" in p and "attempts ledger" in p for p in verify_leaderboard(lb))


# --------------------------------------------------------------------------- #
# Robustness + read-only contract
# --------------------------------------------------------------------------- #
def test_non_dict_competition_is_reported_not_crash():
    lb = _clean()
    lb["competition"] = 7
    assert any("competition: not a JSON object" in p for p in verify_leaderboard(lb))


@pytest.mark.parametrize("bad", [5, True, "x", {"not": "a list"}])
def test_non_list_competition_history_is_reported_not_crash(bad):
    lb = _clean()
    lb["competition"]["history"] = bad
    problems = verify_leaderboard(lb)                     # must not raise
    assert any("competition.history is not a JSON array" in p for p in problems)


def test_stale_updated_at_accounts_for_competition_history():
    lb = _clean()
    lb["updated_at"] = "2026-07-01T00:00:00Z"            # older than the _T0/_T1 wins
    assert any("older than the newest entry" in p for p in verify_leaderboard(lb))


def test_verify_does_not_mutate_competition():
    lb = _clean()
    before = copy.deepcopy(lb)
    verify_leaderboard(lb)
    assert lb == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
