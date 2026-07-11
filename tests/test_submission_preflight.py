"""Offline tests for trinity.submission preflight and gate modules."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity.llm import cost_ledger as CL
from trinity.submission.constants import EXPECTED_HEAD_SHAPE, EXPECTED_TOTAL_PARAMS
from trinity.submission.gates import (
    GateResult,
    OFFLINE_GATES,
    PreflightContext,
    check_duplicate,
    validate_ledger_receipt_cost,
    validate_receipt,
    validate_weights,
)
from trinity.submission.pack import load_submission_pack, parse_submission_identity
from trinity.submission.preflight import PreflightRunner

_REPO = Path(__file__).resolve().parents[1]
_HEAD_SHAPE = EXPECTED_HEAD_SHAPE
_N_SVF = 7168


def _rand_head(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.05, _HEAD_SHAPE).astype(np.float32)


def _near_identity_svf(seed: int, std: float = 0.02) -> np.ndarray:
    return (1.0 + np.random.default_rng(seed).normal(0, std, _N_SVF)).astype(np.float32)


def _honest_receipt(cost: float = 21.5, *, benchmark: str = "math500") -> dict:
    gens = [
        (0.30, 0.50, 0.50), (0.42, 0.58, 0.58), (0.39, 0.61, 0.61),
        (0.55, 0.69, 0.69), (0.58, 0.66, 0.69), (0.60, 0.72, 0.72),
    ]
    return {
        "benchmark": benchmark,
        "pool_models": ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"],
        "n_total": EXPECTED_TOTAL_PARAMS,
        "total_cost_usd": cost,
        "generations": len(gens),
        "best_fitness": 0.72,
        "seed": 42,
        "fitness_history": [
            {"generation": i, "mean_fitness": m, "max_fitness": mx, "best_fitness": b}
            for i, (m, mx, b) in enumerate(gens)
        ],
    }


def _write_submission(
    root: Path,
    miner: str,
    gen: int,
    *,
    head: np.ndarray | None = None,
    svf: np.ndarray | None = None,
    receipt: dict | None = None,
) -> Path:
    d = root / miner / str(gen)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "head_weights.npy", head if head is not None else _rand_head(gen))
    np.save(d / "svf_scales.npy", svf if svf is not None else _near_identity_svf(gen))
    if receipt is not None:
        (d / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# Pack loading
# --------------------------------------------------------------------------- #
def test_parse_submission_identity_from_layout(tmp_path: Path):
    subs = tmp_path / "submissions"
    pack_dir = _write_submission(subs, "alice", 3)
    miner, gen = parse_submission_identity(pack_dir, subs)
    assert miner == "alice"
    assert gen == 3


def test_load_submission_pack_round_trip(tmp_path: Path):
    subs = tmp_path / "submissions"
    receipt = _honest_receipt()
    pack_dir = _write_submission(subs, "bob", 1, receipt=receipt)
    pack = load_submission_pack(pack_dir, submissions_root=subs)
    assert pack is not None
    assert pack.miner == "bob"
    assert pack.generation == 1
    assert pack.head_weights.shape == _HEAD_SHAPE
    assert pack.receipt["best_fitness"] == 0.72


def test_load_submission_pack_missing_weights_returns_none(tmp_path: Path):
    subs = tmp_path / "submissions"
    d = subs / "carol" / "1"
    d.mkdir(parents=True)
    assert load_submission_pack(d, submissions_root=subs) is None


# --------------------------------------------------------------------------- #
# Gate 2: weights
# --------------------------------------------------------------------------- #
def test_validate_weights_accepts_trained_head():
    assert validate_weights(_rand_head(1), _near_identity_svf(1)) is None


def test_validate_weights_rejects_all_zero_head():
    head = np.zeros(_HEAD_SHAPE, dtype=np.float32)
    assert validate_weights(head, _near_identity_svf(1)) == "head_weights_all_zeros"


def test_validate_weights_rejects_nan():
    head = _rand_head(1)
    head[0, 0] = np.nan
    assert validate_weights(head, _near_identity_svf(1)) == "weights_contain_NaN"


# --------------------------------------------------------------------------- #
# Gate 4: receipt
# --------------------------------------------------------------------------- #
def test_validate_receipt_accepts_honest_pack_receipt():
    assert validate_receipt(_honest_receipt()) is None


def test_validate_receipt_rejects_fabricated_best_fitness():
    receipt = _honest_receipt()
    receipt["best_fitness"] = 0.99
    reason = validate_receipt(receipt)
    assert reason is not None
    assert reason.startswith("receipt_best_fitness_mismatch")


# --------------------------------------------------------------------------- #
# Gate 5: ledger / receipt cost
# --------------------------------------------------------------------------- #
def test_ledger_cost_gate_skipped_without_ledger_env():
    assert validate_ledger_receipt_cost(_honest_receipt(), None) is None


def test_ledger_cost_gate_passes_when_totals_match(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 2_000_000, 500_000)
    from trinity.llm.openrouter_pricing import verified_ledger_total_usd

    total = verified_ledger_total_usd(ledger)
    assert total is not None
    receipt = _honest_receipt(cost=round(total, 4))
    assert validate_ledger_receipt_cost(receipt, str(ledger)) is None


def test_ledger_cost_gate_fails_on_mismatch(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "minimax-m3", 1_000_000, 1_000_000)
    receipt = _honest_receipt(cost=1.0)
    reason = validate_ledger_receipt_cost(receipt, str(ledger))
    assert reason is not None
    assert reason.startswith("ledger_receipt_cost_mismatch")


def test_ledger_cost_gate_fails_on_tampered_chain(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 100, 50)
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(text.replace('"p":100', '"p":999'), encoding="utf-8")
    receipt = _honest_receipt(cost=21.5)
    assert validate_ledger_receipt_cost(receipt, str(ledger)) == (
        "ledger_cost_unverifiable: TRINITY_COST_LEDGER failed hash-chain verification"
    )


# --------------------------------------------------------------------------- #
# Gate 3: duplicate (head-only)
# --------------------------------------------------------------------------- #
def test_check_duplicate_catches_copied_head_with_rerolled_svf(tmp_path: Path):
    head = _rand_head(1)
    _write_submission(tmp_path, "alice", 1, head=head, svf=_near_identity_svf(10))
    err = check_duplicate(
        head, _near_identity_svf(999, std=0.05), tmp_path, "bob", 1, leaderboard={"benchmarks": {}}
    )
    assert err is not None and err.startswith("duplicate_of_alice_gen_1")


# --------------------------------------------------------------------------- #
# Preflight runner + CLI
# --------------------------------------------------------------------------- #
def test_preflight_runner_passes_clean_submission(tmp_path: Path, monkeypatch):
    subs = tmp_path / "submissions"
    _write_submission(subs, "dana", 1, receipt=_honest_receipt())
    (tmp_path / "leaderboard.json").write_text(json.dumps({"benchmarks": {}}))
    runner = PreflightRunner(repo_root=tmp_path, benchmark="math500")
    report = runner.run("dana/1")
    assert report.passed
    assert len(report.results) == len(OFFLINE_GATES)
    assert "All offline gates passed." in report.summary_lines()[-1]


def test_preflight_runner_fails_on_duplicate(tmp_path: Path):
    subs = tmp_path / "submissions"
    head = _rand_head(7)
    _write_submission(subs, "king", 1, head=head, receipt=_honest_receipt())
    _write_submission(subs, "eve", 2, head=head, receipt=_honest_receipt())
    (tmp_path / "leaderboard.json").write_text(json.dumps({"benchmarks": {}}))
    runner = PreflightRunner(repo_root=tmp_path, benchmark="math500")
    report = runner.run("eve/2")
    assert not report.passed
    failure = report.first_failure
    assert failure is not None
    assert failure.gate == "duplicate"


def test_preflight_cli_exits_zero_on_pass(tmp_path: Path, monkeypatch):
    subs = tmp_path / "submissions"
    _write_submission(subs, "frank", 1, receipt=_honest_receipt())
    (tmp_path / "leaderboard.json").write_text(json.dumps({"benchmarks": {}}))
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "preflight_submission.py"),
            "--submission",
            "frank/1",
            "--benchmark",
            "math500",
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "All offline gates passed." in proc.stdout


def test_gate_result_failed_property():
    ok = GateResult("weights", True)
    bad = GateResult("receipt", False, "receipt_missing")
    assert not ok.failed
    assert bad.failed
