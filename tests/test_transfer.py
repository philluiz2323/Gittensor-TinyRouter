"""Offline tests for the zero-shot transfer report (SPEC §6.1 vs §6.2).

The SPEC's strongest generalization claim is that the coordinator still beats the best
single model on benchmarks it never trained on (§6.2 held-out; Table 1). Nothing encoded
that split, so these tests pin the three things that make the verdict trustworthy:

* benchmarks land in the cohort the SPEC assigns them (and an unlisted one is EXCLUDED,
  never guessed into a cohort — misfiling one moves a task across the boundary being
  measured);
* a cohort mean can never hide a held-out benchmark that lost;
* "transfer holds" requires a POSITIVE held-out margin, not merely a positive average
  across both cohorts.

Synthetic rows, stdlib only — no torch, no network.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis import transfer as transfer_pkg  # re-export check
from trinity.analysis.transfer import (
    HELD_OUT,
    IN_DISTRIBUTION,
    assess,
    classify,
    render,
)

_REPO = Path(__file__).resolve().parents[1]


def _row(bench, tri, single, model="glm"):
    return {"benchmark": bench, "trinity": tri, "best_single": single, "best_model": model}


# --------------------------------------------------------------------------- #
# cohort classification
# --------------------------------------------------------------------------- #
def test_spec_cohorts_are_as_documented():
    assert {"math500", "mmlu", "livecodebench"} <= IN_DISTRIBUTION
    assert {"aime", "aime2025", "gpqa", "bigcodebench"} <= HELD_OUT
    assert IN_DISTRIBUTION.isdisjoint(HELD_OUT)


def test_classify_resolves_aliases_through_the_grader():
    assert classify("livecodebench_v6") == "in_distribution"   # alias of livecodebench
    assert classify("MATH500") == "in_distribution"            # case-insensitive
    assert classify("gpqa") == "held_out"


def test_unlisted_benchmark_is_unknown_not_guessed():
    for bench in ("swebench_verified", "drop", "bbh", "", "made-up"):
        assert classify(bench) == "unknown"


def test_unknown_benchmarks_are_excluded_and_reported():
    s = assess([_row("math500", 0.9, 0.8), _row("gpqa", 0.7, 0.6),
                _row("swebench_verified", 0.3, 0.9)])
    assert s.unknown == ["swebench_verified"]
    assert s.in_distribution.n == 1 and s.held_out.n == 1     # not folded into a cohort


# --------------------------------------------------------------------------- #
# margins + the verdict
# --------------------------------------------------------------------------- #
def test_transfer_holds_when_the_held_out_margin_is_at_least_the_in_dist_one():
    s = assess([_row("math500", 0.90, 0.85), _row("gpqa", 0.80, 0.70)])
    assert s.in_distribution.margin == pytest.approx(0.05)
    assert s.held_out.margin == pytest.approx(0.10)
    assert s.transfer_gap > 0
    assert s.verdict == "holds"
    assert "HOLDS" in render(s)


def test_shrinking_but_positive_margin_is_degraded_not_holds():
    s = assess([_row("math500", 0.90, 0.70), _row("gpqa", 0.71, 0.70)])
    assert s.held_out.margin > 0 and s.transfer_gap < 0
    assert s.verdict == "degraded"
    assert "DEGRADED" in render(s)


def test_negative_held_out_margin_fails_even_with_a_big_in_dist_win():
    # The case the old all-benchmark average hid: a huge in-dist win masking a held-out loss.
    s = assess([_row("math500", 0.99, 0.50), _row("gpqa", 0.60, 0.70)])
    assert s.verdict == "failed"
    assert "FAILED" in render(s)


def test_zero_held_out_margin_is_not_a_pass():
    s = assess([_row("math500", 0.9, 0.8), _row("gpqa", 0.7, 0.7)])
    assert s.held_out.margin == 0.0
    assert s.verdict == "failed"      # tying the best single is not beating it


def test_no_held_out_evidence_is_insufficient_not_a_pass():
    s = assess([_row("math500", 0.9, 0.5)])
    assert s.held_out.n == 0
    assert s.transfer_gap is None
    assert s.verdict == "insufficient_evidence"
    assert "untested" in render(s)


def test_cohort_mean_cannot_hide_a_held_out_loss():
    # aime wins big, gpqa loses; the cohort mean stays positive but the loss is surfaced.
    s = assess([_row("math500", 0.9, 0.8), _row("aime", 0.9, 0.5), _row("gpqa", 0.60, 0.70)])
    assert s.held_out.margin > 0
    assert s.held_out.losses == ["gpqa"]
    assert "does NOT win: gpqa" in render(s)


# --------------------------------------------------------------------------- #
# row handling
# --------------------------------------------------------------------------- #
def test_repeated_benchmark_keeps_the_best_trinity():
    s = assess([_row("gpqa", 0.60, 0.70), _row("gpqa", 0.80, 0.70)])
    assert s.held_out.n == 1
    assert s.held_out.benchmarks[0].trinity == 0.80

def test_rows_missing_a_score_are_skipped_not_zeroed():
    s = assess([_row("math500", None, 0.8), _row("gpqa", 0.7, None),
                {"benchmark": "aime"}, _row("mmlu", 0.9, 0.8)])
    assert [b.benchmark for b in s.in_distribution.benchmarks] == ["mmlu"]
    assert s.held_out.n == 0          # a missing score must not fabricate a 0.0 loss


def test_summary_is_json_serializable():
    json.dumps(assess([_row("math500", 0.9, 0.8), _row("gpqa", 0.7, 0.6)]).to_dict())


def test_empty_input_renders_gracefully():
    s = assess([])
    assert s.verdict == "insufficient_evidence"
    assert "no benchmarks classified" in render(s)


def test_module_is_reachable_through_the_analysis_package():
    assert hasattr(transfer_pkg, "assess") and hasattr(transfer_pkg, "classify")


# --------------------------------------------------------------------------- #
# report script
# --------------------------------------------------------------------------- #
def _write_eval(path, bench, tri, singles):
    payload = {"benchmark": bench, "results": {"TRINITY": tri,
                                               **{f"single::{m}": v for m, v in singles.items()}}}
    Path(path).write_text(json.dumps(payload))


def test_report_script_end_to_end(tmp_path):
    _write_eval(tmp_path / "eval_math500.json", "math500", 0.92, {"glm": 0.81, "ds": 0.70})
    _write_eval(tmp_path / "eval_gpqa.json", "gpqa", 0.70, {"glm": 0.74})
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "transfer_report.py"), "--root", str(tmp_path)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0, out.stderr
    assert "in_distribution" in out.stdout and "held_out" in out.stdout
    assert "FAILED" in out.stdout          # gpqa loses -> held-out margin negative


def test_report_script_tolerates_a_null_single_score(tmp_path):
    _write_eval(tmp_path / "eval_math500.json", "math500", 0.9, {"glm": None, "ds": 0.8})
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "transfer_report.py"), "--root", str(tmp_path)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0, out.stderr
    assert "insufficient evidence" in out.stdout.lower()   # in-dist only


def test_report_script_no_files_is_graceful():
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "transfer_report.py")],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0 and "no eval JSONs" in out.stdout
