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

from trinity.llm.cost_ledger import read_ledger_entries, verify_ledger_chain
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
    "validate_fitness_history_sequence",
    "audit_ledger_call_volume",
    "audit_head_routing_diversity",
    "validate_pack_schema",
    "validate_theta_integrity",
    "OFFLINE_GATES",
    "OFFLINE_ADVISORIES",
    "run_gate",
    "run_offline_gates",
    "run_offline_advisories",
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


# --------------------------------------------------------------------------- #
# Gate 8 + advisories (issue #208)
# --------------------------------------------------------------------------- #
_FITNESS_TOL = 1e-9

#: Agent-row spread (relative to row scale) below which a head routes to one model always.
_ROUTING_COLLAPSE_RATIO = 1e-3


def _as_number(x: Any) -> Optional[float]:
    """``x`` as a float, or ``None`` when it is not a real number."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _row_value(row: dict[str, Any], *keys: str) -> Optional[float]:
    """First numeric value among ``keys`` (tolerating the legacy ``gen_*`` names)."""
    for k in keys:
        v = _as_number(row.get(k))
        if v is not None:
            return v
    return None


def validate_fitness_history_sequence(receipt: dict[str, Any]) -> Optional[str]:
    """Internal-consistency invariants of a genuine CMA-ES fitness history (gate 8, #208).

    :func:`validate_receipt` (gate 4) checks the history's **aggregate** shape — length,
    starting value, flatness, a too-perfect monotone mean, claimed-vs-actual generation
    count, and peak vs ``best_fitness``. It never reads ``generation`` and never compares a
    row's mean against its **own** max, so a fabricated curve can keep a plausible
    aggregate while carrying:

    * scrambled, duplicated, or gappy ``generation`` indices (a real run logs each
      generation once, in order);
    * a row whose ``mean_fitness`` exceeds its ``max_fitness`` — a population's mean cannot
      beat its own best; and
    * a running ``best_fitness`` that **decreases** — best-so-far can only improve.

    These are structural facts about any real optimizer run, not style preferences, so they
    are safe to enforce as a hard gate. Legacy receipts whose history rows are bare numbers
    (or lack ``generation``) are left to gate 4: the corresponding check is skipped rather
    than rejecting an older-but-honest pack.

    Args:
        receipt: The parsed ``receipt.json``.

    Returns:
        A short reason string, or ``None`` when the history is self-consistent.
    """
    history = receipt.get("fitness_history")
    if not isinstance(history, list) or not history:
        return None                       # gate 4 owns missing/short histories
    rows = [h for h in history if isinstance(h, dict)]
    if not rows:
        return None                       # legacy numeric history -> gate 4's aggregate checks

    # `validate_pack_schema` (gate 6) already requires a `generation` on every row; this
    # checks the SEQUENCE those indices form. Skipped unless every row carries an integer
    # generation, so a legacy pack is never rejected for a field it predates.
    ints: list[int] = []
    if len(rows) == len(history):
        for row in rows:
            g = _as_number(row.get("generation"))
            if g is None or not g.is_integer():
                ints = []
                break
            ints.append(int(g))
    if ints:
        dups = sorted({g for g in ints if ints.count(g) > 1})
        if dups:
            return f"receipt_fitness_history_duplicate_generation: {dups}"
        if ints != sorted(ints):
            return "receipt_fitness_history_generations_out_of_order"
        if ints[-1] - ints[0] != len(ints) - 1:
            return (f"receipt_fitness_history_generations_not_consecutive: {len(ints)} rows "
                    f"span {ints[0]}..{ints[-1]}")

    for row in rows:
        mean = _row_value(row, "mean_fitness", "gen_mean_fitness")
        peak = _row_value(row, "max_fitness", "gen_max_fitness")
        if mean is not None and peak is not None and mean > peak + _FITNESS_TOL:
            return (f"receipt_fitness_row_mean_exceeds_max: generation "
                    f"{row.get('generation')} mean {mean:.4f} > max {peak:.4f}")

    bests = [_row_value(r, "best_fitness") for r in rows]
    if all(b is not None for b in bests) and len(bests) >= 2:
        for i in range(1, len(bests)):
            prev, cur = bests[i - 1], bests[i]
            if cur is not None and prev is not None and cur < prev - _FITNESS_TOL:
                return (f"receipt_fitness_best_decreased: generation "
                        f"{rows[i].get('generation')} best {cur:.4f} < {prev:.4f}")
    return None


def audit_ledger_call_volume(
    receipt: dict[str, Any],
    ledger_path: str | None,
) -> Optional[str]:
    """ADVISORY: does the ledger show enough traffic for the claimed CMA-ES run? (#208 gate 9)

    Gate 5 verifies the ledger's dollar total against the receipt, but a tampered ledger
    holding one expensive row can match that total while implying far fewer model calls than
    the claimed ``generations x popsize`` evaluations.

    **This is advisory, never a rejection.** The ledger records only ``{model, prompt_tokens,
    completion_tokens}`` with *no run identifier*, so it cannot be bound to THIS submission:
    unrelated historical traffic in a shared ledger can satisfy any volume threshold, and a
    legitimately-resumed or partially-logged run can fall under it. Until ledger entries
    carry run-scoped provenance, a thin ledger is a signal to look, not proof of fraud
    (reviewed rationale on the closed #210).

    Returns:
        A warning string when the ledger looks implausibly thin for the claim, else ``None``.
        Skipped (``None``) when no ledger path is supplied, matching gate 5.
    """
    if not ledger_path:
        return None
    ok, _n, _tip = verify_ledger_chain(ledger_path)
    if not ok:
        return "ledger_chain_unverified (gate 5 owns rejection; reported here for context)"
    entries = read_ledger_entries(ledger_path)
    if not entries:
        return "ledger_has_no_entries for the claimed training run"

    gens = _as_number(receipt.get("generations")) or 0.0
    popsize = _as_number(receipt.get("popsize")) or 0.0
    expected_calls = int(gens * popsize)
    if expected_calls > 0 and len(entries) < expected_calls:
        return (f"ledger_call_volume_low: {len(entries)} ledger row(s) for a claimed "
                f"{int(gens)} x {int(popsize)} = {expected_calls} evaluations "
                f"(ledger is not run-scoped, so this is informational only)")
    distinct = {e.model for e in entries}
    if len(distinct) < 2:
        return (f"ledger_single_model_traffic: only {sorted(distinct)} seen; a pool run "
                "normally calls >= 2 models (informational only)")
    return None


def audit_head_routing_diversity(head_W: np.ndarray) -> Optional[str]:
    """ADVISORY: do the agent logit rows collapse to a single routing choice? (#208 gate 10)

    Duplicate detection (gate 3) compares a head against *other* submissions, never against
    itself, so a head whose agent rows are near-identical — which always routes to the same
    model (argmax tie -> index 0) — passes unnoticed.

    **This is advisory, never a rejection.** The competition contract is score-based: a
    constant or near-constant router may be strategically poor, but it is a *valid*
    submission if it scores better. Rejecting agent-row collapse at preflight would change
    the contest from outcome-based to style-based (reviewed rationale on the closed #210),
    so this only warns the miner that their head likely ignores the query.

    Collapse means the agent rows are near-**equal**, so every model's logit ``z_m = W_m·h``
    is identical for any query and the argmax always ties to index 0. Equality — not mere
    parallelism — is what matters: two parallel rows of different magnitude still route
    differently. So this measures the rows' spread about their mean relative to their own
    scale. (A pairwise cosine *after mean-centering*, as originally proposed, is degenerate
    here: centering identical rows yields zero vectors, and centering near-identical rows
    leaves only noise whose cosine is random — it misses the very collapse it targets.)

    Returns:
        A warning string when the agent rows carry no meaningful spread, else ``None``.
    """
    W = np.asarray(head_W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] < N_HEAD_MODELS:
        return None
    agent = W[:N_HEAD_MODELS]
    scale = float(np.linalg.norm(agent, axis=1).mean())
    if scale <= 0.0:
        return None                       # an all-zero head is gate 2's rejection, not ours
    spread = float(np.linalg.norm(agent - agent.mean(axis=0, keepdims=True), axis=1).max())
    if spread / scale < _ROUTING_COLLAPSE_RATIO:
        return (f"head_agent_rows_collapsed: agent-row spread is {spread / scale:.2e} of the "
                f"row scale (< {_ROUTING_COLLAPSE_RATIO:g}); every model receives the same "
                "logit, so this head routes to one model for every query (a valid submission "
                "— informational only)")
    return None


def _gate_receipt(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    del ctx
    if not pack.receipt:
        return "receipt_missing"
    return validate_receipt(pack.receipt)


def _gate_fitness_history_sequence(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    del ctx
    if not pack.receipt:
        return None                       # gate 4/6 own a missing receipt
    return validate_fitness_history_sequence(pack.receipt)


def _advisory_ledger_call_volume(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    if not pack.receipt:
        return None
    return audit_ledger_call_volume(pack.receipt, ctx.ledger_path)


def _advisory_head_routing_diversity(pack: SubmissionPack, ctx: PreflightContext) -> Optional[str]:
    del ctx
    return audit_head_routing_diversity(pack.head_weights)


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

#: Checks that REPORT but never reject (issue #208). Each is a real signal a miner (or a
#: maintainer) wants to see, but neither is a sound rejection criterion:
#:
#: * ``ledger_call_volume`` — the ledger carries no run identifier, so a shared ledger's
#:   unrelated traffic can satisfy any threshold; it needs run-scoped provenance before it
#:   could safely reject.
#: * ``head_routing_diversity`` — the competition contract is score-based, so a collapsed
#:   router is *valid* if it scores better; rejecting it would make the contest
#:   style-based rather than outcome-based.
#:
#: * ``fitness_history_sequence`` — a fabricated curve is worth telling a miner about, but
#:   ``4bb03a7`` deliberately relaxed the gate chain "to attract miners, not repel them",
#:   dropping the receipt's fitness-shape checks entirely. Adding a new *blocking* check
#:   would run against that; as an advisory it costs a miner nothing.
#:
#: The first two rationales come from the review on the closed #210. Callers print these as
#: ``[WARN]`` and MUST NOT fail a submission on them.
OFFLINE_ADVISORIES: tuple[SubmissionGate, ...] = (
    SubmissionGate("fitness_history_sequence", _gate_fitness_history_sequence),
    SubmissionGate("ledger_call_volume", _advisory_ledger_call_volume),
    SubmissionGate("head_routing_diversity", _advisory_head_routing_diversity),
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


def run_offline_advisories(
    pack: SubmissionPack,
    ctx: PreflightContext,
    *,
    advisories: tuple[SubmissionGate, ...] = OFFLINE_ADVISORIES,
) -> list[GateResult]:
    """Run the ADVISORY checks (issue #208); every advisory always runs.

    These **never reject a submission** — see :data:`OFFLINE_ADVISORIES` for why neither is
    a sound rejection criterion. A returned :class:`GateResult` with ``failed`` True means
    "this advisory fired", i.e. a ``[WARN]`` for the caller to print; callers must not turn
    it into a rejection. An advisory that raises is reported as a fired advisory rather than
    aborting, so a broken advisory can never take down the scoring path.
    """
    results: list[GateResult] = []
    for advisory in advisories:
        try:
            results.append(run_gate(advisory, pack, ctx))
        except Exception as exc:  # noqa: BLE001 - an advisory must never break scoring
            results.append(GateResult(
                gate=advisory.name, ok=False,
                reason=f"advisory_error: {type(exc).__name__}: {exc}",
            ))
    return results
