"""Receipt schema and theta-layout validation for routing-head submissions.

Gate 6 (pack schema) and gate 7 (theta integrity) run offline with no GPU and
no OpenRouter calls. They catch mismatched receipts and corrupted weight packs
before ``pr_eval`` loads the encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trinity.coordinator.params import make_spec, pack, unpack
from trinity.submission.constants import (
    DEFAULT_POOL_MODELS,
    EXPECTED_HEAD_SHAPE,
    EXPECTED_TOTAL_PARAMS,
)

__all__ = [
    "REQUIRED_RECEIPT_FIELDS",
    "PackSchemaValidator",
    "ThetaIntegrityValidator",
    "validate_pack_schema",
    "validate_theta_integrity",
]

REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "benchmark",
    "total_cost_usd",
    "fitness_history",
    "generations",
    "pool_models",
    "n_total",
)


@dataclass(frozen=True)
class PackSchemaValidator:
    """Validate ``receipt.json`` structure and cross-field consistency."""

    benchmark: str
    pool_models: tuple[str, ...] = DEFAULT_POOL_MODELS
    expected_n_total: int = EXPECTED_TOTAL_PARAMS

    def validate(self, receipt: dict[str, Any]) -> str | None:
        if not receipt:
            return "receipt_missing"

        for field in REQUIRED_RECEIPT_FIELDS:
            if field not in receipt:
                return f"receipt_missing_field: {field}"

        bench = receipt.get("benchmark")
        if not isinstance(bench, str) or not bench.strip():
            return "receipt_benchmark_invalid"
        if bench != self.benchmark:
            return f"receipt_benchmark_mismatch: receipt {bench!r} != expected {self.benchmark!r}"

        n_total = receipt.get("n_total")
        if not isinstance(n_total, int) or n_total <= 0:
            return "receipt_n_total_invalid"
        if n_total != self.expected_n_total:
            return f"receipt_n_total_mismatch: got {n_total}, expected {self.expected_n_total}"

        pool = receipt.get("pool_models")
        if not isinstance(pool, list) or not pool:
            return "receipt_pool_models_invalid"
        if not all(isinstance(m, str) and m.strip() for m in pool):
            return "receipt_pool_models_invalid"
        expected = set(self.pool_models)
        got = set(pool)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            return (
                f"receipt_pool_models_mismatch: missing={missing or []} extra={extra or []}"
            )

        gens = receipt.get("generations")
        if not isinstance(gens, int) or gens < 1:
            return "receipt_generations_invalid"

        history = receipt.get("fitness_history")
        if not isinstance(history, list) or not history:
            return "receipt_fitness_history_invalid"

        cost = receipt.get("total_cost_usd")
        if not isinstance(cost, (int, float)) or float(cost) <= 0.0:
            return "receipt_total_cost_invalid"

        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                return f"receipt_fitness_history_entry_not_object: index {idx}"
            if "generation" not in entry:
                return f"receipt_fitness_history_missing_generation: index {idx}"

        return None


@dataclass(frozen=True)
class ThetaIntegrityValidator:
    """Verify head + SVF weights round-trip through the canonical theta layout."""

    head_shape: tuple[int, int] = EXPECTED_HEAD_SHAPE
    expected_n_total: int = EXPECTED_TOTAL_PARAMS

    def validate(self, head_weights: np.ndarray, svf_scales: np.ndarray) -> str | None:
        head = np.asarray(head_weights)
        svf = np.asarray(svf_scales)

        if head.shape != self.head_shape:
            return f"theta_head_shape: got {head.shape}, expected {self.head_shape}"
        if svf.ndim != 1:
            return f"theta_svf_not_1d: got shape {svf.shape}"
        if head.size + svf.size != self.expected_n_total:
            return (
                f"theta_param_count: got {head.size + svf.size}, "
                f"expected {self.expected_n_total}"
            )

        spec = make_spec(n_a=self.head_shape[0], d_h=self.head_shape[1], n_svf=svf.size)
        try:
            theta = pack(head, svf)
        except ValueError as exc:
            return f"theta_pack_failed: {exc}"

        if theta.size != self.expected_n_total:
            return f"theta_vector_length: got {theta.size}, expected {self.expected_n_total}"

        try:
            head_back, svf_back = unpack(theta, spec)
        except ValueError as exc:
            return f"theta_unpack_failed: {exc}"

        head64 = np.ascontiguousarray(head, dtype=np.float64)
        svf64 = np.ascontiguousarray(svf, dtype=np.float64).ravel()
        if not np.array_equal(head_back, head64):
            return "theta_roundtrip_head_mismatch"
        if not np.array_equal(svf_back, svf64):
            return "theta_roundtrip_svf_mismatch"

        return None


def validate_pack_schema(receipt: dict[str, Any], benchmark: str) -> str | None:
    """Gate 6: receipt schema and benchmark/pool consistency."""
    return PackSchemaValidator(benchmark=benchmark).validate(receipt)


def validate_theta_integrity(head_weights: np.ndarray, svf_scales: np.ndarray) -> str | None:
    """Gate 7: canonical theta pack/unpack round-trip."""
    return ThetaIntegrityValidator().validate(head_weights, svf_scales)
