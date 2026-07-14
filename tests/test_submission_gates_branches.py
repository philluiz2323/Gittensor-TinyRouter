"""Offline coverage for the anti-cheat submission gates (`trinity.submission.gates`).

`test_submission_preflight.py` exercises the happy path and a few failures, but
most of the individual **rejection branches** — the ones that actually stop a
cheating or malformed submission — were uncovered (gates.py sat at 82%). Each of
these is an independent anti-cheat rule; an untested branch that silently stops
rejecting is exactly the kind of regression a coverage gap hides.

These tests drive the pure gate functions directly (no GPU, no OpenRouter, no
torch): from a known-VALID weight pair / receipt they perturb one thing at a time
and assert the specific rejection reason, plus the duplicate-scan filesystem
branches and the per-gate `receipt_missing` wrappers.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.submission import gates as G
from trinity.submission.constants import (
    DUPLICATE_HEAD_COSINE_THRESHOLD,
    EXPECTED_HEAD_SHAPE,
    MAX_WEIGHT_MAGNITUDE,
    MIN_TRAINING_COST_USD,
    N_HEAD_MODELS,
    RATE_LIMIT_MAX_SUBMISSIONS,
)
from trinity.submission.pack import SubmissionPack


# --------------------------------------------------------------------------- #
# valid baselines
# --------------------------------------------------------------------------- #
def _valid_weights():
    head = np.full(EXPECTED_HEAD_SHAPE, 0.01, dtype=np.float64)
    svf = np.full(7 * 1024, 1.0, dtype=np.float64)
    return head, svf


def _distinct_head(seed):
    # Row-varying head so routing_invariant_head (which mean-centers each row group)
    # does NOT collapse it to zero — required for a meaningful cosine comparison.
    return np.random.default_rng(seed).standard_normal(EXPECTED_HEAD_SHAPE)


def _valid_receipt():
    return {
        "total_cost_usd": 20.0,
        "fitness_history": [
            {"gen_mean_fitness": 0.10},
            {"gen_mean_fitness": 0.30},
            {"gen_mean_fitness": 0.25},
            {"gen_mean_fitness": 0.40},
        ],
        "generations": 4,
        "best_fitness": 0.0,
    }


def test_baselines_are_valid():
    assert G.validate_weights(*_valid_weights()) is None
    assert G.validate_receipt(_valid_receipt()) is None


# --------------------------------------------------------------------------- #
# validate_weights — one rejection branch per test
# --------------------------------------------------------------------------- #
def test_weights_reject_wrong_param_count():
    head, svf = _valid_weights()
    assert "param_count" in G.validate_weights(head[:, :-1], svf)


def test_weights_reject_wrong_head_shape():
    # Same total param count, wrong 2-D shape (3x2048 instead of 6x1024).
    _, svf = _valid_weights()
    head = np.full((3, 2048), 0.01)
    assert "head_shape" in G.validate_weights(head, svf)


def test_weights_reject_inf():
    head, svf = _valid_weights()
    head[0, 0] = np.inf
    assert G.validate_weights(head, svf) == "weights_contain_Inf"


def test_weights_reject_head_over_max_magnitude():
    head, svf = _valid_weights()
    head[0, 0] = MAX_WEIGHT_MAGNITUDE * 2
    assert "head_weights_exceed_max" in G.validate_weights(head, svf)


def test_weights_reject_svf_over_max_magnitude():
    head, svf = _valid_weights()
    svf[0] = MAX_WEIGHT_MAGNITUDE * 2
    assert "svf_scales_exceed_max" in G.validate_weights(head, svf)


def test_weights_reject_svf_all_zeros():
    head, svf = _valid_weights()
    assert G.validate_weights(head, np.zeros_like(svf)) == "svf_scales_all_zeros"


def test_weights_reject_head_norm_too_small():
    _, svf = _valid_weights()
    head = np.full(EXPECTED_HEAD_SHAPE, 1e-5)  # nonzero but norm < 0.001
    assert "head_weight_norm_too_small" in G.validate_weights(head, svf)


# --------------------------------------------------------------------------- #
# cosine_similarity / routing_invariant_head edges
# --------------------------------------------------------------------------- #
def test_cosine_similarity_zero_norm_vectors():
    z = np.zeros(4)
    assert G.cosine_similarity(z, z) == 1.0        # both zero -> identical
    assert G.cosine_similarity(z, np.ones(4)) == 0.0  # one zero -> orthogonal


def test_routing_invariant_head_none_for_non_head_shapes():
    assert G.routing_invariant_head(np.ones(10)) is None            # 1-D
    assert G.routing_invariant_head(np.ones((N_HEAD_MODELS, 4))) is None  # too few rows


# --------------------------------------------------------------------------- #
# check_rate_limit — the same-PR re-eval skip
# --------------------------------------------------------------------------- #
def _lb_with_attempt(miner, benchmark, pr, *, age_days=0.0):
    import time

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_days * 86400))
    return {"benchmarks": {benchmark: {"attempts": [
        {"miner": miner, "timestamp": ts, "pr": pr},
    ]}}}


def test_rate_limit_ignores_reeval_of_same_pr():
    lb = _lb_with_attempt("alice", "math500", pr=7)
    # Re-evaluating PR #7 must not count against alice (CI retry / transient fail).
    assert G.check_rate_limit("alice", "math500", lb, current_pr=7) is None


def test_rate_limit_counts_a_distinct_pr():
    lb = _lb_with_attempt("alice", "math500", pr=7)
    reason = G.check_rate_limit("alice", "math500", lb, current_pr=8)
    assert reason is not None and "rate_limited" in reason
    assert RATE_LIMIT_MAX_SUBMISSIONS == 1


def test_rate_limit_ignores_attempts_outside_the_window():
    # An attempt older than the window (10 > 7 days) must not count.
    lb = _lb_with_attempt("alice", "math500", pr=1, age_days=10.0)
    assert G.check_rate_limit("alice", "math500", lb, current_pr=2) is None


def test_rate_limit_skips_entry_with_unparseable_timestamp():
    lb = {"benchmarks": {"math500": {"attempts": [
        {"miner": "alice", "timestamp": "not-a-timestamp", "pr": 1},
    ]}}}
    assert G.check_rate_limit("alice", "math500", lb, current_pr=2) is None


def test_parse_utc_timestamp_edges():
    assert G.parse_utc_timestamp("") is None
    assert G.parse_utc_timestamp("garbage") is None
    assert G.parse_utc_timestamp("2026-01-02T03:04:05Z") == pytest.approx(1767322745.0)


# --------------------------------------------------------------------------- #
# check_duplicate — filesystem scan branches
# --------------------------------------------------------------------------- #
def _write_sub(root, miner, gen, head, svf):
    d = root / miner / str(gen)
    d.mkdir(parents=True, exist_ok=True)
    np.save(str(d / "head_weights.npy"), head)
    np.save(str(d / "svf_scales.npy"), svf)
    return d


def test_duplicate_detects_local_copy(tmp_path):
    head, svf = _distinct_head(1), np.full(7 * 1024, 1.0)
    _write_sub(tmp_path, "bob", 1, head.copy(), svf)  # identical -> cosine 1.0
    reason = G.check_duplicate(head, svf, tmp_path, "alice", 1)
    assert reason is not None and reason.startswith("duplicate_of_bob_gen_1")
    assert DUPLICATE_HEAD_COSINE_THRESHOLD == 0.999


def test_duplicate_skips_dirs_missing_arrays(tmp_path):
    (tmp_path / "bob" / "1").mkdir(parents=True)  # no .npy files
    head, svf = _distinct_head(1), np.full(7 * 1024, 1.0)
    assert G.check_duplicate(head, svf, tmp_path, "alice", 1) is None


def test_duplicate_skips_corrupt_arrays(tmp_path):
    d = tmp_path / "bob" / "1"
    d.mkdir(parents=True)
    (d / "head_weights.npy").write_bytes(b"not a real npy file")
    (d / "svf_scales.npy").write_bytes(b"garbage")
    head, svf = _distinct_head(1), np.full(7 * 1024, 1.0)
    assert G.check_duplicate(head, svf, tmp_path, "alice", 1) is None


def test_duplicate_scans_leaderboard_king_without_false_hit(tmp_path):
    # A reigning king that does NOT match must exercise the king-scan branch and
    # still return None (no duplicate).
    head, svf = _distinct_head(1), np.full(7 * 1024, 1.0)
    _write_sub(tmp_path, "king", 2, _distinct_head(2), svf)  # different head
    lb = {"benchmarks": {"math500": {"best_miner": "king", "best_generation": 2}}}
    assert G.check_duplicate(head, svf, tmp_path, "alice", 1, leaderboard=lb) is None


def test_duplicate_king_loaded_via_load_leaderboard_callback(tmp_path):
    head, svf = _distinct_head(1), np.full(7 * 1024, 1.0)
    _write_sub(tmp_path, "king", 3, _distinct_head(2), svf)
    lb = {"benchmarks": {"math500": {"best_miner": "king", "best_generation": 3}}}
    assert G.check_duplicate(
        head, svf, tmp_path, "alice", 1, load_leaderboard=lambda: lb
    ) is None


def test_duplicate_king_scan_skips_empty_and_missing_and_corrupt(tmp_path):
    head, svf = _distinct_head(1), np.full(7 * 1024, 1.0)
    # A corrupt king dir (bad .npy) is skipped; an empty best_miner is skipped;
    # a king pointing at a non-existent dir is skipped -> overall None.
    bad = tmp_path / "corruptking" / "9"
    bad.mkdir(parents=True)
    (bad / "head_weights.npy").write_bytes(b"nope")
    (bad / "svf_scales.npy").write_bytes(b"nope")
    lb = {"benchmarks": {
        "a": {"best_miner": "", "best_generation": 0},              # empty -> skipped
        "b": {"best_miner": "ghost", "best_generation": 1},         # dir missing -> skipped
        "c": {"best_miner": "corruptking", "best_generation": 9},   # corrupt -> skipped
    }}
    assert G.check_duplicate(head, svf, tmp_path, "alice", 1, leaderboard=lb) is None


# --------------------------------------------------------------------------- #
# validate_receipt — anti-cheat heuristics
# --------------------------------------------------------------------------- #
def test_receipt_reject_zero_cost():
    r = _valid_receipt()
    r["total_cost_usd"] = 0.0
    assert G.validate_receipt(r) == "receipt_cost_zero_or_missing"


def test_receipt_reject_cost_below_minimum():
    r = _valid_receipt()
    r["total_cost_usd"] = MIN_TRAINING_COST_USD - 1.0
    assert "receipt_cost_too_low" in G.validate_receipt(r)


def test_receipt_reject_history_too_short():
    r = _valid_receipt()
    r["fitness_history"] = [{"gen_mean_fitness": 0.1}, {"gen_mean_fitness": 0.2}]
    assert "receipt_fitness_history_too_short" in G.validate_receipt(r)


def test_receipt_reject_no_valid_fitness_values():
    r = _valid_receipt()
    # Three entries, none carrying a usable fitness key -> no numeric values.
    r["fitness_history"] = [{"note": "x"}, {"note": "y"}, {"note": "z"}]
    assert G.validate_receipt(r) == "receipt_fitness_history_no_valid_values"


def test_receipt_reject_flat_line():
    r = _valid_receipt()
    r["fitness_history"] = [{"gen_mean_fitness": 0.5}] * 4
    assert G.validate_receipt(r) == "receipt_fitness_flat_line"


def test_receipt_reject_starts_too_high():
    r = _valid_receipt()
    r["fitness_history"] = [
        {"gen_mean_fitness": 0.99},
        {"gen_mean_fitness": 0.30},
        {"gen_mean_fitness": 0.40},
    ]
    assert "receipt_fitness_starts_too_high" in G.validate_receipt(r)


def test_receipt_reject_too_perfect_monotonic():
    r = _valid_receipt()
    # >3 diffs, all non-negative -> "too perfect" (bare floats hit the numeric path).
    r["fitness_history"] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    r["generations"] = 6
    assert "receipt_fitness_too_perfect" in G.validate_receipt(r)


def test_receipt_reject_generations_mismatch():
    r = _valid_receipt()  # history of length 4, non-monotonic
    r["generations"] = 100
    assert "receipt_generations_mismatch" in G.validate_receipt(r)


def test_receipt_reject_best_fitness_mismatch():
    r = _valid_receipt()
    r["fitness_history"] = [
        {"gen_mean_fitness": 0.10, "gen_max_fitness": 0.15},
        {"gen_mean_fitness": 0.30, "gen_max_fitness": 0.35},
        {"gen_mean_fitness": 0.25, "gen_max_fitness": 0.28},
    ]
    r["generations"] = 3
    r["best_fitness"] = 0.95  # far above the 0.35 history peak
    assert "receipt_best_fitness_mismatch" in G.validate_receipt(r)


def test_receipt_history_tolerates_mixed_numeric_and_junk_entries():
    # Bare floats take the numeric path; a non-dict/non-number entry is skipped in
    # BOTH the values loop and the peaks loop. best_fitness > 0 drives the peak scan.
    r = _valid_receipt()
    r["fitness_history"] = [0.10, 0.50, 0.20, "junk"]
    r["generations"] = 4
    r["best_fitness"] = 0.95  # peaks are the bare floats -> max 0.50, mismatch
    assert "receipt_best_fitness_mismatch" in G.validate_receipt(r)


# --------------------------------------------------------------------------- #
# validate_ledger_receipt_cost / per-gate wrappers
# --------------------------------------------------------------------------- #
def test_ledger_cost_noop_when_receipt_cost_nonpositive():
    # cost <= 0 short-circuits to None before any ledger read.
    assert G.validate_ledger_receipt_cost({"total_cost_usd": 0.0}, "some/path") is None


def test_gate_ledger_cost_wrapper_skips_when_no_receipt(tmp_path):
    # The per-gate wrapper returns None (no-op) when the pack carries no receipt.
    assert G._gate_ledger_cost(_pack(tmp_path, {}), _ctx(tmp_path)) is None


def test_gate_pack_schema_wrapper_flags_missing_receipt(tmp_path):
    assert G._gate_pack_schema(_pack(tmp_path, {}), _ctx(tmp_path)) == "receipt_missing"


def _pack(tmp_path, receipt, *, head=None, svf=None):
    if head is None:
        head, svf = _valid_weights()
    return SubmissionPack(path=tmp_path, miner="alice", generation=1,
                          head_weights=head, svf_scales=svf, receipt=receipt)


def _ctx(tmp_path):
    return G.PreflightContext(benchmark="math500", leaderboard={},
                              submissions_root=tmp_path)


def test_run_offline_gates_receipt_gate_flags_missing_receipt(tmp_path):
    results = G.run_offline_gates(_pack(tmp_path, {}), _ctx(tmp_path))
    # Chain stops at the first failure; the receipt gate reports the missing pack.
    assert results[-1].failed
    assert results[-1].reason == "receipt_missing"


def test_run_offline_gates_stops_at_first_failure(tmp_path):
    # An all-zero head fails the weights gate; later gates never run.
    pack = _pack(tmp_path, _valid_receipt(),
                 head=np.zeros(EXPECTED_HEAD_SHAPE), svf=np.full(7 * 1024, 1.0))
    results = G.run_offline_gates(pack, _ctx(tmp_path))
    assert results[-1].failed
    assert [r.gate for r in results] == ["rate_limit", "weights"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])


def test_run_offline_gates_collect_all_runs_every_gate(tmp_path):
    # An all-zero head fails the weights gate; with collect_all every later gate
    # still runs, so a miner sees all problems at once (docstring's promise).
    pack = _pack(tmp_path, _valid_receipt(),
                 head=np.zeros(EXPECTED_HEAD_SHAPE), svf=np.full(7 * 1024, 1.0))
    results = G.run_offline_gates(pack, _ctx(tmp_path), collect_all=True)
    assert [r.gate for r in results] == [g.name for g in G.OFFLINE_GATES]
    assert any(r.gate == "weights" and r.failed for r in results)


def test_run_offline_gates_default_is_still_fail_fast(tmp_path):
    pack = _pack(tmp_path, _valid_receipt(),
                 head=np.zeros(EXPECTED_HEAD_SHAPE), svf=np.full(7 * 1024, 1.0))
    results = G.run_offline_gates(pack, _ctx(tmp_path))
    assert [r.gate for r in results] == ["rate_limit", "weights"]


def test_run_offline_gates_collect_all_captures_a_raising_gate(tmp_path):
    def _boom(pack, ctx):
        raise RuntimeError("kaboom")

    gates = (G.SubmissionGate("boom", _boom), G.SubmissionGate("after", lambda p, c: None))
    results = G.run_offline_gates(_pack(tmp_path, _valid_receipt()), _ctx(tmp_path),
                                  gates=gates, collect_all=True)
    assert [r.gate for r in results] == ["boom", "after"]
    assert results[0].failed and "gate_error" in (results[0].reason or "")
    assert results[1].ok  # the raising gate did not abort the collection
