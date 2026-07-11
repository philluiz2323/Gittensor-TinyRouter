"""Offline validation for routing-head submission packs."""

from trinity.submission.constants import (
    DEFAULT_POOL_MODELS,
    DUPLICATE_HEAD_COSINE_THRESHOLD,
    EXPECTED_HEAD_SHAPE,
    EXPECTED_TOTAL_PARAMS,
    MIN_TRAINING_COST_USD,
    RATE_LIMIT_MAX_SUBMISSIONS,
    RATE_LIMIT_WINDOW_DAYS,
)
from trinity.submission.gates import (
    GateResult,
    OFFLINE_GATES,
    PreflightContext,
    SubmissionGate,
    check_duplicate,
    check_rate_limit,
    cosine_similarity,
    parse_utc_timestamp,
    rate_limit_entries,
    run_gate,
    run_offline_gates,
    validate_ledger_receipt_cost,
    validate_receipt,
    validate_weights,
)
from trinity.submission.leaderboard import (
    headroom_captured,
    leaderboard_report,
    verify_leaderboard,
)
from trinity.submission.pack import SubmissionPack, load_submission_pack, parse_submission_identity
from trinity.submission.preflight import PreflightReport, PreflightRunner, load_leaderboard_json
from trinity.submission.schema import (
    PackSchemaValidator,
    ThetaIntegrityValidator,
    validate_pack_schema,
    validate_theta_integrity,
)

__all__ = [
    "verify_leaderboard",
    "leaderboard_report",
    "headroom_captured",
    "DEFAULT_POOL_MODELS",
    "DUPLICATE_HEAD_COSINE_THRESHOLD",
    "EXPECTED_HEAD_SHAPE",
    "EXPECTED_TOTAL_PARAMS",
    "MIN_TRAINING_COST_USD",
    "RATE_LIMIT_MAX_SUBMISSIONS",
    "RATE_LIMIT_WINDOW_DAYS",
    "GateResult",
    "OFFLINE_GATES",
    "PreflightContext",
    "PreflightReport",
    "PreflightRunner",
    "SubmissionGate",
    "SubmissionPack",
    "check_duplicate",
    "check_rate_limit",
    "cosine_similarity",
    "load_leaderboard_json",
    "load_submission_pack",
    "parse_submission_identity",
    "parse_utc_timestamp",
    "rate_limit_entries",
    "run_gate",
    "run_offline_gates",
    "validate_ledger_receipt_cost",
    "validate_pack_schema",
    "validate_receipt",
    "validate_theta_integrity",
    "validate_weights",
    "PackSchemaValidator",
    "ThetaIntegrityValidator",
]
