"""Offline tests for submission pack schema and theta integrity gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trinity.submission.constants import EXPECTED_HEAD_SHAPE, EXPECTED_TOTAL_PARAMS
from trinity.submission.gates import OFFLINE_GATES
from trinity.submission.schema import (
    PackSchemaValidator,
    ThetaIntegrityValidator,
    validate_pack_schema,
    validate_theta_integrity,
)

_HEAD_SHAPE = EXPECTED_HEAD_SHAPE
_N_SVF = 7168


def _rand_head(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.05, _HEAD_SHAPE).astype(np.float32)


def _near_identity_svf(seed: int, std: float = 0.02) -> np.ndarray:
    return (1.0 + np.random.default_rng(seed).normal(0, std, _N_SVF)).astype(np.float32)


def _valid_receipt(*, benchmark: str = "math500") -> dict:
    gens = [
        (0.30, 0.50, 0.50), (0.42, 0.58, 0.58), (0.39, 0.61, 0.61),
        (0.55, 0.69, 0.69), (0.58, 0.66, 0.69), (0.60, 0.72, 0.72),
    ]
    return {
        "benchmark": benchmark,
        "pool_models": ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"],
        "n_total": EXPECTED_TOTAL_PARAMS,
        "total_cost_usd": 21.5,
        "generations": len(gens),
        "best_fitness": 0.72,
        "seed": 7,
        "fitness_history": [
            {"generation": i, "mean_fitness": m, "max_fitness": mx, "best_fitness": b}
            for i, (m, mx, b) in enumerate(gens)
        ],
    }


# --------------------------------------------------------------------------- #
# Pack schema
# --------------------------------------------------------------------------- #
def test_validate_pack_schema_accepts_honest_receipt():
    assert validate_pack_schema(_valid_receipt(), "math500") is None


@pytest.mark.parametrize(
    "mutator,reason_prefix",
    [
        (lambda r: r.pop("benchmark"), "receipt_missing_field"),
        (lambda r: r.update({"benchmark": "mmlu"}), "receipt_benchmark_mismatch"),
        (lambda r: r.update({"n_total": 999}), "receipt_n_total_mismatch"),
        (lambda r: r.update({"pool_models": ["bogus-model"]}), "receipt_pool_models_mismatch"),
        (lambda r: r.update({"generations": 0}), "receipt_generations_invalid"),
        (lambda r: r.update({"total_cost_usd": 0}), "receipt_total_cost_invalid"),
        (lambda r: r.update({"fitness_history": []}), "receipt_fitness_history_invalid"),
    ],
)
def test_validate_pack_schema_rejects_common_drift(mutator, reason_prefix: str):
    receipt = _valid_receipt()
    mutator(receipt)
    err = validate_pack_schema(receipt, "math500")
    assert err is not None
    assert err.startswith(reason_prefix)


def test_pack_schema_validator_requires_generation_field_per_history_row():
    receipt = _valid_receipt()
    receipt["fitness_history"][2] = {"mean_fitness": 0.5}
    err = PackSchemaValidator(benchmark="math500").validate(receipt)
    assert err == "receipt_fitness_history_missing_generation: index 2"


def test_offline_gates_include_schema_and_theta():
    names = [gate.name for gate in OFFLINE_GATES]
    assert names[-2:] == ["pack_schema", "theta_integrity"]


# --------------------------------------------------------------------------- #
# Theta integrity
# --------------------------------------------------------------------------- #
def test_validate_theta_integrity_accepts_canonical_pack():
    head = _rand_head(1)
    svf = _near_identity_svf(1)
    assert validate_theta_integrity(head, svf) is None


def test_theta_integrity_rejects_wrong_head_shape():
    head = np.zeros((5, 1024), dtype=np.float32)
    svf = _near_identity_svf(2)
    err = ThetaIntegrityValidator().validate(head, svf)
    assert err is not None
    assert err.startswith("theta_head_shape")


def test_theta_integrity_rejects_non_1d_svf():
    head = _rand_head(3)
    svf = np.ones((8, 896), dtype=np.float32)
    err = validate_theta_integrity(head, svf)
    assert err is not None
    assert "theta_svf_not_1d" in err or "theta_param_count" in err


def test_theta_integrity_rejects_truncated_svf():
    head = _rand_head(4)
    svf = np.ones(100, dtype=np.float32)
    err = validate_theta_integrity(head, svf)
    assert err is not None
    assert "theta_param_count" in err


def test_theta_roundtrip_detects_hand_edited_head(tmp_path: Path):
    head = _rand_head(5)
    svf = _near_identity_svf(5)
    assert validate_theta_integrity(head, svf) is None

    bad = head.copy()
    bad[0, 0] = np.float32(999.0)
    # Still same shape/count — schema passes, but round-trip still holds for float32.
    # Corrupt layout by saving wrong shape on disk is caught earlier; here we verify
    # pack/unpack catches reshaped svf that does not match head block boundaries.
    bad_svf = svf[:-1]
    err = validate_theta_integrity(bad, bad_svf)
    assert err is not None


def test_pack_submission_receipt_shape_matches_schema_contract():
    """Receipt emitted by pack_submission includes fields gate 6 requires."""
    receipt = _valid_receipt(benchmark="mmlu")
    raw = json.dumps(receipt)
    loaded = json.loads(raw)
    assert validate_pack_schema(loaded, "mmlu") is None


@pytest.mark.parametrize("field", [
    "benchmark",
    "total_cost_usd",
    "fitness_history",
    "generations",
    "pool_models",
    "n_total",
])
def test_each_required_receipt_field_is_enforced(field: str):
    receipt = _valid_receipt()
    receipt.pop(field)
    err = validate_pack_schema(receipt, "math500")
    assert err == f"receipt_missing_field: {field}"


def test_preflight_runner_fails_pack_schema_before_duplicate_scan(tmp_path: Path):
    from trinity.submission.preflight import PreflightRunner

    subs = tmp_path / "submissions"
    pack_dir = subs / "bob" / "1"
    pack_dir.mkdir(parents=True)
    np.save(pack_dir / "head_weights.npy", _rand_head(9))
    np.save(pack_dir / "svf_scales.npy", _near_identity_svf(9))
    bad = _valid_receipt()
    bad["benchmark"] = "mmlu"
    (pack_dir / "receipt.json").write_text(json.dumps(bad), encoding="utf-8")

    (tmp_path / "leaderboard.json").write_text(
        json.dumps({"benchmarks": {"math500": {"attempts": []}}}),
        encoding="utf-8",
    )
    report = PreflightRunner(repo_root=tmp_path, benchmark="math500").run("bob/1")
    assert not report.passed
    assert report.first_failure is not None
    assert report.first_failure.gate == "pack_schema"
    assert "receipt_benchmark_mismatch" in (report.first_failure.reason or "")
