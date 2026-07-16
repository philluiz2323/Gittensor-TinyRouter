"""Offline anti-cheat gates for routing-head submissions (pr_eval gates 1–7).

These checks run with no GPU and no OpenRouter calls. ``scripts/pr_eval.py``
imports this module; miners can run the same logic locally via
``scripts/preflight_submission.py`` before opening a PR.
"""
from __future__ import annotations

import calendar
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from trinity.llm.openrouter_pricing import verified_ledger_total_usd
from trinity.submission.constants import (
    DUPLICATE_HEAD_COSINE_THRESHOLD,
    EXPECTED_HEAD_SHAPE,
    EXPECTED_TOTAL_PARAMS,
    LEDGER_RECEIPT_COST_TOLERANCE_USD,
    MAX_WEIGHT_MAGNITUDE,
    N_HEAD_MODELS,
    RATE_LIMIT_MAX_SUBMISSIONS,
    RATE_LIMIT_WINDOW_DAYS,
)
from trinity.submission.pack import SubmissionPack
from trinity.submission.schema import validate_pack_schema, validate_theta_integrity

__all__ = [
    "GateResult",
    "PreflightContext",
    "SubmissionGate",
    "parse_utc_timestamp",
    "rate_limit_entries",
    "check_rate_limit",
    "validate_weights",
    "check_duplicate",
    "cosine_similarity",
    "validate_receipt",
    "validate_ledger_receipt_cost",
    "validate_pack_schema",
    "validate_theta_integrity",
    "OFFLINE_GATES",
    "run_gate",
    "run_offline_gates",
]


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate or the full offline preflight chain."""

    gate: str
    ok: bool
    reason: str | None = None

    @property
    def failed(self) -> bool:
        return not self.ok


@dataclass
class PreflightContext:
    """Runtime inputs shared across gates."""

    benchmark: str
    leaderboard: dict[str, Any]
    submissions_root: Path
    pr_number: int | None = None
    ledger_path: str | None = None
    load_leaderboard: Callable[[], dict[str, Any]] | None = None


class SubmissionGate:
    """Named gate with a stable identifier for logging and tests."""

    name: str
    _check: Callable[[SubmissionPack, PreflightContext], Optional[str]]

    def __init__(self, name: str, check: Callable[[SubmissionPack, PreflightContext], Optional[str]]):
        self.name = name
        self._check = check

    def run(self, pack: SubmissionPack, ctx: PreflightContext) -> GateResult:
        reason = self._check(pack, ctx)
        return GateResult(gate=self.name, ok=reason is None, reason=reason)


def parse_utc_timestamp(ts_str: str) -> Optional[float]:
    """Parse ``YYYY-MM-DDTHH:MM:SSZ`` to a UTC Unix epoch."""
    if not ts_str:
        return None
    try:
        return float(calendar.timegm(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, OSError):
        return None


def rate_limit_entries(bench_entry: dict[str, Any]) -> list[dict[str, Any]]:
    # A tampered leaderboard.json can carry a non-list `attempts`/`history` (a scalar
    # written where an array belongs); iterating it must not crash the gate or the
    # verifier that share this accessor. A non-list ledger yields no entries -- the
    # integrity verifier flags the malformation separately (report, don't crash).
    if not isinstance(bench_entry, dict):
        return []
    attempts = bench_entry.get("attempts")
    if attempts is not None:
        return list(attempts) if isinstance(attempts, list) else []
    history = bench_entry.get("history")
    return list(history) if isinstance(history, list) else []


def check_rate_limit(
    miner_name: str,
    benchmark: str,
    leaderboard: dict[str, Any],
    *,
    current_pr: int | None = None,
) -> Optional[str]:
    benches = leaderboard.get("benchmarks", {})
    bench_entry = benches.get(benchmark, {}) if isinstance(benches, dict) else {}
    entries = rate_limit_entries(bench_entry)
    cutoff = time.time() - RATE_LIMIT_WINDOW_DAYS * 86400
    recent = 0
    for entry in entries:
        ts = parse_utc_timestamp(entry.get("timestamp", ""))
        if ts is None:
            continue
        if entry.get("miner") != miner_name or ts <= cutoff:
            continue
        # Re-evaluating the SAME PR is not a second submission. The attempt was
        # recorded when Gate 1 first passed; if the eval was then re-run (a CI
        # retry, or a transient GPU/API failure during live scoring), that same
        # attempt must not count against the miner and self-reject their PR.
        # A distinct PR still counts, preserving the anti-probe intent.
        if current_pr is not None and entry.get("pr") == current_pr:
            continue
        recent += 1
    if recent >= RATE_LIMIT_MAX_SUBMISSIONS:
        return (
            f"rate_limited: {recent} submission(s) in the last "
            f"{RATE_LIMIT_WINDOW_DAYS} days (max {RATE_LIMIT_MAX_SUBMISSIONS})"
        )
    return None


def validate_weights(head_W: np.ndarray, svf_scales: np.ndarray) -> Optional[str]:
    if head_W.size + svf_scales.size != EXPECTED_TOTAL_PARAMS:
        return f"param_count: got {head_W.size + svf_scales.size}, expected {EXPECTED_TOTAL_PARAMS}"
    if head_W.shape != EXPECTED_HEAD_SHAPE:
        return f"head_shape: got {head_W.shape}, expected {EXPECTED_HEAD_SHAPE}"
    if np.any(np.isnan(head_W)) or np.any(np.isnan(svf_scales)):
        return "weights_contain_NaN"
    if np.any(np.isinf(head_W)) or np.any(np.isinf(svf_scales)):
        return "weights_contain_Inf"
    if np.any(np.abs(head_W) > MAX_WEIGHT_MAGNITUDE):
        return f"head_weights_exceed_max: max_abs={np.max(np.abs(head_W)):.1f}"
    if np.any(np.abs(svf_scales) > MAX_WEIGHT_MAGNITUDE):
        return f"svf_scales_exceed_max: max_abs={np.max(np.abs(svf_scales)):.1f}"
    if np.allclose(head_W, 0.0):
        return "head_weights_all_zeros"
    if np.allclose(svf_scales, 0.0):
        return "svf_scales_all_zeros"
    # Norm threshold removed for launch — a legitimately small head (e.g. early
    # in CMA-ES or with a small-activation encoder) can have norm < 0.001 and
    # still route meaningfully. The all-zeros check above catches the degenerate
    # case; the overfit gate catches a head that doesn't learn.
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


def routing_invariant_head(head_W: np.ndarray) -> Optional[np.ndarray]:
    """Return a routing-behaviour view of a head for duplicate comparison.

    ``LinearHead`` routes by argmax/softmax over two INDEPENDENT logit groups of
    ``z = W·h``: agent rows ``[0:N_HEAD_MODELS)`` and role rows
    ``[N_HEAD_MODELS:]``. Two per-group transforms leave the routed ``argmax``
    unchanged for every input, so a plagiarised head can use either to drive the
    raw-weight cosine below the gate threshold while behaving identically:

    * an **additive** per-group shift (add a common vector ``c`` to every row of a
      group -> every logit shifts by the scalar ``c·h``), fixed in #152; and
    * a **positive per-group scaling** (multiply a group's rows by ``α > 0`` ->
      every logit scales by ``α``), issue #256.

    To be a faithful routing fingerprint the view must be invariant to BOTH. For
    each group: mean-center its rows (removes the additive shift), then L2-normalize
    (removes the positive scale). An exact copy and any of its shifted and/or
    rescaled variants collapse to the same representation (cosine ``1.0``), while
    genuinely different heads stay distinct.

    Args:
        head_W: A head weight matrix, expected shape ``(n_a, d_h)`` with
            ``n_a > N_HEAD_MODELS``.

    Returns:
        The centered + per-group-normalized head flattened to 1-D, or ``None`` when
        ``head_W`` is not a 2-D head with more than ``N_HEAD_MODELS`` rows.
    """
    W = np.asarray(head_W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] <= N_HEAD_MODELS:
        return None
    out = W.copy()
    for group in (out[:N_HEAD_MODELS], out[N_HEAD_MODELS:]):
        group -= group.mean(axis=0, keepdims=True)   # kill the additive per-group shift (#152)
        norm = float(np.linalg.norm(group))
        if norm > 0.0:
            group /= norm                            # kill the positive per-group scale (#256)
    return out.ravel()


def _same_generation(a: str | int, b: str | int) -> bool:
    """Whether two generation labels denote the same generation.

    ``pr_eval`` identifies a submission's generation with ``int(parts[1])``, so a
    non-canonical directory name (``07``, ``007``, ``+7``) is still accepted as the
    integer it parses to. Comparing generations as strings here would then fail to
    recognise a submission's own directory and flag it as a copy of itself. Compare
    as integers when both parse, falling back to string equality for genuinely
    non-numeric names.

    Args:
        a: A generation label (directory name ``str`` or ``int``).
        b: The other generation label.

    Returns:
        ``True`` if the two denote the same generation.
    """
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def check_duplicate(
    head_W: np.ndarray,
    svf_scales: np.ndarray,
    submissions_root: Path,
    current_miner: str,
    current_gen: int,
    *,
    leaderboard: dict[str, Any] | None = None,
    load_leaderboard: Callable[[], dict[str, Any]] | None = None,
) -> Optional[str]:
    head = routing_invariant_head(head_W)
    svf = np.asarray(svf_scales, dtype=np.float64).ravel()

    def _match(other_hw: np.ndarray, other_sv: np.ndarray) -> Optional[tuple[float, float]]:
        other_head = routing_invariant_head(other_hw)
        if head is None or other_head is None or other_head.size != head.size:
            return None
        h_sim = cosine_similarity(head, other_head)
        s_sim = cosine_similarity(svf, np.asarray(other_sv, dtype=np.float64).ravel())
        return (h_sim, s_sim) if h_sim > DUPLICATE_HEAD_COSINE_THRESHOLD else None

    for sub_dir in sorted(submissions_root.glob("*/*/")):
        parts = sub_dir.relative_to(submissions_root).parts
        if len(parts) < 2:
            continue
        other_miner, other_gen = parts[0], parts[1]
        if other_miner == current_miner and _same_generation(other_gen, current_gen):
            continue
        hw_path = sub_dir / "head_weights.npy"
        sv_path = sub_dir / "svf_scales.npy"
        if not hw_path.exists() or not sv_path.exists():
            continue
        try:
            other_hw = np.load(str(hw_path))
            other_sv = np.load(str(sv_path))
        except (ValueError, OSError):
            continue
        hit = _match(other_hw, other_sv)
        if hit is not None:
            return (
                f"duplicate_of_{other_miner}_gen_{other_gen}"
                f"_head_sim_{hit[0]:.4f}_svf_sim_{hit[1]:.4f}"
            )

    lb = leaderboard if leaderboard is not None else (
        load_leaderboard() if load_leaderboard is not None else {}
    )
    for _bench_name, bench_entry in lb.get("benchmarks", {}).items():
        king_miner = bench_entry.get("best_miner", "")
        king_gen = bench_entry.get("best_generation", 0)
        if not king_miner or king_miner == current_miner:
            continue
        king_dir = submissions_root / king_miner / str(king_gen)
        hw_path = king_dir / "head_weights.npy"
        sv_path = king_dir / "svf_scales.npy"
        if not (hw_path.exists() and sv_path.exists()):
            continue
        try:
            king_hw = np.load(str(hw_path))
            king_sv = np.load(str(sv_path))
        except (ValueError, OSError):
            continue
        hit = _match(king_hw, king_sv)
        if hit is not None:
            return (
                f"duplicate_of_king_{king_miner}_gen_{king_gen}"
                f"_head_sim_{hit[0]:.4f}_svf_sim_{hit[1]:.4f}"
            )
    return None


def validate_receipt(receipt: dict[str, Any]) -> Optional[str]:
    """Gate 4: receipt proves real training happened.

    Requires cost > 0 and more than 2 fitness entries (>= 3 generations) so a
    miner cannot submit an essentially untrained head after 1-2 generations.
    """
    cost = receipt.get("total_cost_usd", 0.0)
    if cost <= 0.0:
        return "receipt_cost_zero_or_missing"

    history = receipt.get("fitness_history", [])
    if not history or len(history) < 3:
        return "receipt_fitness_history_too_short: need > 2 entries (>= 3 generations)"

    return None


def validate_ledger_receipt_cost(
    receipt: dict[str, Any],
    ledger_path: str | None,
) -> Optional[str]:
    """Gate 5: receipt ``total_cost_usd`` must match the verified ledger total."""
    if not ledger_path:
        return None
    receipt_cost = float(receipt.get("total_cost_usd", 0.0))
    if receipt_cost <= 0.0:
        return None
    ledger_total = verified_ledger_total_usd(ledger_path)
    if ledger_total is None:
        return "ledger_cost_unverifiable: TRINITY_COST_LEDGER failed hash-chain verification"
    delta = abs(receipt_cost - ledger_total)
    if delta > LEDGER_RECEIPT_COST_TOLERANCE_USD:
        return (
            f"ledger_receipt_cost_mismatch: receipt ${receipt_cost:.4f} vs "
            f"verified ledger ${ledger_total:.4f} (delta ${delta:.4f})"
        )
    return None


def _gate_rate_limit(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    return check_rate_limit(
        pack.miner, ctx.benchmark, ctx.leaderboard, current_pr=ctx.pr_number,
    )


def _gate_weights(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    del ctx
    return validate_weights(pack.head_weights, pack.svf_scales)


def _gate_duplicate(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    return check_duplicate(
        pack.head_weights,
        pack.svf_scales,
        ctx.submissions_root,
        pack.miner,
        pack.generation,
        leaderboard=ctx.leaderboard,
        load_leaderboard=ctx.load_leaderboard,
    )


def _gate_receipt(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    del ctx
    if not pack.receipt:
        return "receipt_missing"
    return validate_receipt(pack.receipt)


def _gate_ledger_cost(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    if not pack.receipt:
        return None
    return validate_ledger_receipt_cost(pack.receipt, ctx.ledger_path)


def _gate_pack_schema(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    if not pack.receipt:
        return "receipt_missing"
    return validate_pack_schema(pack.receipt, ctx.benchmark)


def _gate_theta_integrity(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    del ctx
    return validate_theta_integrity(pack.head_weights, pack.svf_scales)


OFFLINE_GATES: tuple[SubmissionGate, ...] = (
    SubmissionGate("rate_limit", _gate_rate_limit),
    SubmissionGate("weights", _gate_weights),
    SubmissionGate("duplicate", _gate_duplicate),
    SubmissionGate("receipt", _gate_receipt),
    SubmissionGate("ledger_cost", _gate_ledger_cost),
    SubmissionGate("pack_schema", _gate_pack_schema),
    SubmissionGate("theta_integrity", _gate_theta_integrity),
)


def run_gate(gate: SubmissionGate, pack: SubmissionPack, ctx: PreflightContext) -> GateResult:
    return gate.run(pack, ctx)


def run_offline_gates(
    pack: SubmissionPack,
    ctx: PreflightContext,
    *,
    gates: tuple[SubmissionGate, ...] = OFFLINE_GATES,
    collect_all: bool = False,
) -> list[GateResult]:
    """Run the offline gates in order.

    By default this is **fail-fast** — it stops at the first failing gate, matching
    the scoring path. With ``collect_all=True`` every gate runs so a caller (e.g.
    the local :class:`~trinity.submission.preflight.PreflightRunner`) can surface
    all problems in one pass instead of one-per-run. A gate that raises is captured
    as a failing :class:`GateResult` rather than aborting the collection, since a
    later gate may assume an earlier one passed.
    """
    results: list[GateResult] = []
    for gate in gates:
        try:
            result = run_gate(gate, pack, ctx)
        except Exception as exc:  # noqa: BLE001 - a later gate may assume earlier ones passed
            result = GateResult(
                gate=gate.name, ok=False,
                reason=f"gate_error: {type(exc).__name__}: {exc}",
            )
        results.append(result)
        if result.failed and not collect_all:
            break
    return results
