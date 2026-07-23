"""Offline tests for the termination-profile diagnostic. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.analysis.termination_profile import (
    analyze,
    analyze_benchmarks,
    render,
)


# ---------------------------------------------------------------------------
# analyze — accept/exhausted rates, turn stats, never_accepts
# ---------------------------------------------------------------------------
def test_accept_and_exhausted_rates():
    recs = [(2, "accept"), (3, "accept"), (5, "max_turns"), (5, None)]
    r = analyze(recs, benchmark="math500")
    assert r.n_trajectories == 4
    assert r.accept_rate == pytest.approx(0.5)
    assert r.exhausted_rate == pytest.approx(0.5)
    assert r.mean_turns == pytest.approx(3.75)
    assert r.median_turns == pytest.approx(4.0)
    assert r.max_turns_observed == 5
    assert r.turn_histogram == {2: 1, 3: 1, 5: 2}


def test_never_accepts_flag_when_no_trajectory_accepts():
    r = analyze([(5, "max_turns"), (5, None), (5, "max_turns")])
    assert r.accept_rate == 0.0 and r.exhausted_rate == 1.0
    assert r.never_accepts is True


def test_all_accept_is_not_flagged():
    r = analyze([(1, "accept"), (2, "accept")])
    assert r.accept_rate == 1.0 and r.never_accepts is False


def test_bool_and_string_accepted_shapes():
    # accepted may be a bool, or a terminated_by string; only "accept"-like => accepted.
    recs = [
        {"turns": 2, "accepted": True},
        {"turns": 3, "terminated_by": "accept"},
        {"turns": 4, "terminated_by": "max_turns"},
        {"turns": 5, "accepted": False},
    ]
    r = analyze(recs)
    assert r.accept_rate == pytest.approx(0.5)


def test_unusable_records_are_skipped():
    # non-positive / non-int turn counts and junk are dropped.
    r = analyze([(0, "accept"), (-1, "accept"), ("x", True), None, (3, "accept")])
    assert r.n_trajectories == 1 and r.accept_rate == 1.0


def test_empty_input_is_zeroed_and_not_flagged():
    r = analyze([])
    assert r.n_trajectories == 0 and r.never_accepts is False
    assert r.accept_rate == 0.0 and r.turn_histogram == {}


# ---------------------------------------------------------------------------
# analyze_benchmarks — per-benchmark + pooled union
# ---------------------------------------------------------------------------
def test_union_pools_all_and_flags_inert_benchmarks():
    per = {
        "math500": [(2, "accept"), (3, "accept")],       # healthy loop
        "mmlu": [(5, "max_turns"), (5, None)],           # never accepts
    }
    report = analyze_benchmarks(per)
    assert report["union"]["n_trajectories"] == 4
    assert report["union"]["accept_rate"] == pytest.approx(0.5)
    assert report["any_never_accepts"] is True
    assert report["never_accepts_benchmarks"] == ["mmlu"]


def test_histogram_serializes_with_string_keys():
    report = analyze_benchmarks({"a": [(2, "accept"), (2, None), (4, "accept")]})
    hist = report["per_benchmark"][0]["turn_histogram"]
    assert hist == {"2": 2, "4": 1}          # JSON object keys are strings, sorted


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def test_render_reports_inert_loop():
    per = {"mmlu": [(5, "max_turns"), (5, None)]}
    md = render(per)
    assert "Verifier NEVER accepts" in md and "mmlu" in md
    assert "| union |" in md
    assert "| benchmark | n | accept rate | exhausted | mean turns | histogram |" in md


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
