#!/usr/bin/env python3
"""Evaluate a PR submission against the hidden benchmark.

This is the maintainer's tool. It evaluates a miner's submitted routing head
against the HIDDEN benchmark (stored OUTSIDE the repo — never committed) and
determines whether the head beats the current best accuracy.

4 pre-eval gates run before GPU/API work (a 5th overfit gate runs post-eval). A failing gate
rejects the submission immediately with zero cost to the maintainer.

Usage:
    source ~/.config/trinity/secrets.env
    python scripts/pr_eval.py --pr 42 --benchmark math500 --submission alice/1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.submission.constants import (
    COMPETITION_BENCHMARKS,
    DEFAULT_POOL_MODELS,
    DUPLICATE_HEAD_COSINE_THRESHOLD,
    EXPECTED_TOTAL_PARAMS,
    MAX_WEIGHT_MAGNITUDE,
    MIN_TRAINING_COST_USD,
    N_HEAD_MODELS,
    RATE_LIMIT_MAX_SUBMISSIONS,
    RATE_LIMIT_WINDOW_DAYS,
    WIN_MARGIN,
)
from trinity.submission.gates import (
    audit_head_routing_diversity,
    audit_ledger_call_volume,
    check_duplicate as _check_duplicate,
    check_rate_limit as _check_rate_limit,
    cosine_similarity as _cosine_similarity,
    parse_utc_timestamp as _parse_utc_timestamp,
    rate_limit_entries as _rate_limit_entries,
    routing_invariant_head as _routing_invariant_head,
    validate_fitness_history_sequence,
    validate_ledger_receipt_cost,
    validate_pack_schema,
    validate_receipt as _validate_receipt,
    validate_theta_integrity,
    validate_weights as _validate_weights,
)

# Back-compat aliases for tests that import pr_eval directly.
_EXPECTED_HEAD_PARAMS = 6144
_EXPECTED_SVF_PARAMS = 7168
_EXPECTED_TOTAL = EXPECTED_TOTAL_PARAMS
_MIN_TRAINING_COST = MIN_TRAINING_COST_USD
_MAX_WEIGHT_MAGNITUDE = MAX_WEIGHT_MAGNITUDE
_COPY_THRESHOLD = DUPLICATE_HEAD_COSINE_THRESHOLD
_OVERFIT_HARD_REJECT = 0.15  # raised from 0.10 (too strict for launch)
_OVERFIT_PENALTY = 0.08     # raised from 0.05
_RATE_LIMIT_WINDOW_DAYS = RATE_LIMIT_WINDOW_DAYS
_RATE_LIMIT_MAX_SUBMISSIONS = RATE_LIMIT_MAX_SUBMISSIONS
_POOL_MODELS = list(DEFAULT_POOL_MODELS)
_N_HEAD_MODELS = N_HEAD_MODELS


# ==========================================================================
# AES decryption (matches build_benchmark.py's encryption)
# ==========================================================================

def _derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)


def _decrypt_json(filepath: Path, password: str) -> dict:
    """Decrypt an AES-256-GCM encrypted JSON benchmark file."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        print("ERROR: cryptography package required. pip install cryptography")
        sys.exit(1)

    ciphertext_b64 = filepath.read_text().strip()
    combined = base64.b64decode(ciphertext_b64)
    salt = combined[:16]
    nonce = combined[16:28]
    ct = combined[28:]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    plain = aesgcm.decrypt(nonce, ct, None)
    data = json.loads(plain.decode("utf-8"))
    return data["items"]  # extract the items list


# ==========================================================================
# Hidden benchmark loader
# ==========================================================================

def _load_hidden_benchmark(benchmark_name: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """Load encrypted eval, audit, and live question sets for a benchmark."""
    bench_dir = os.environ.get(
        "TINYROUTER_BENCHMARK_DIR",
        str(_REPO.parent / "tinyrouter-benchmark"),
    )
    bench_path = Path(bench_dir) / benchmark_name
    password = os.environ.get("BENCHMARK_PASSWORD", "")

    if not bench_path.exists():
        print(f"[pr_eval] ERROR: Hidden benchmark not found at {bench_path}")
        print(f"  Set TINYROUTER_BENCHMARK_DIR or run scripts/build_benchmark.py first.")
        sys.exit(2)
    if not password:
        print("[pr_eval] ERROR: BENCHMARK_PASSWORD env var not set.")
        sys.exit(2)

    eval_items = []
    audit_items = []
    live_items = []

    for filename, target in [("eval.json", eval_items), ("audit.json", audit_items), ("live.json", live_items)]:
        fp = bench_path / filename
        if fp.exists():
            try:
                items = _decrypt_json(fp, password)
                target.extend(items)
            except Exception as exc:
                print(f"[pr_eval] ERROR: Failed to decrypt {filename}: {exc}")
                sys.exit(2)

    return eval_items, audit_items, live_items


# ==========================================================================
# Load submission
# ==========================================================================

def _load_submission(submission_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Load head_weights.npy, svf_scales.npy, and receipt.json."""
    hw = submission_dir / "head_weights.npy"
    sv = submission_dir / "svf_scales.npy"
    rc = submission_dir / "receipt.json"

    if not hw.exists() or not sv.exists():
        print(f"[pr_eval] ERROR: Missing submission files in {submission_dir}")
        return None

    head_W = np.load(str(hw))
    svf_scales = np.load(str(sv))
    receipt = json.loads(rc.read_text()) if rc.exists() else {}

    return head_W.astype(np.float32), svf_scales.astype(np.float32), receipt


# Gate implementations live in trinity.submission.gates (imported above).


# ==========================================================================
# Cached evaluation
# ==========================================================================

def _evaluate_cached(policy, items: List[dict], pool_model_names: List[str]) -> float:
    """Evaluate a configured policy on cached benchmark items. Returns accuracy [0, 1].

    Scoring routes through the benchmark **adapter** (``adapter.score_output``)
    rather than calling ``reward.score_text`` directly (issue #154), so the
    maintainer scorer honours the same adapter contract the rest of the repo
    advertises. For math/mmlu/gpqa/livecodebench the delegating adapter forwards to
    ``score_text`` -- identical results -- while execution-aware benchmarks such as
    SWE-bench Verified are graded through their own ``score_output`` (the patch
    scorer) instead of being mis-dispatched.
    """
    from trinity.adapters import get_adapter
    from trinity.adapters.hidden_item import from_protocol_item
    from trinity.orchestration.session import routing_transcript

    if not items:
        return 0.0

    correct = 0
    for item in items:
        canonical = from_protocol_item(item)
        agent_idx, _role = policy.decide(routing_transcript(canonical["prompt"]), sample=False)
        model_name = pool_model_names[agent_idx % len(pool_model_names)]
        cached = canonical["cached_model_answers"].get(model_name, "")
        adapter = get_adapter(canonical["benchmark"] or "math500")
        if adapter.score_output(cached, canonical["reference"]) > 0.0:
            correct += 1

    return correct / len(items)


# ==========================================================================
# Live evaluation
# ==========================================================================

async def _evaluate_live(
    policy, pool, pool_models, items: List[dict],
    max_turns: int = 5, max_tokens: int = 4096,
) -> Tuple[float, float]:
    """Run full multi-turn eval with real API calls. Returns (accuracy, avg_turns).

    Live trajectories are graded through the benchmark **adapter**
    (``adapter.score_trajectory``) rather than ``reward.score`` directly (issue
    #154), so the maintainer live scorer uses the same adapter contract as the
    main evaluator (``trinity.eval``) and can grade execution-aware benchmarks such
    as SWE-bench Verified (whose adapter scores the final patch) end-to-end. The
    adapter's default ``score_trajectory`` picks the committed answer across turns
    and delegates to ``score_output`` -- the documented evaluator rule -- so
    non-SWE benchmarks are scored exactly as ``trinity.eval`` scores them.
    """
    import httpx
    from trinity.adapters import get_adapter
    from trinity.adapters.hidden_item import from_protocol_item
    from trinity.orchestration.session import run_trajectory
    from trinity.types import Task

    if not items:
        return 0.0, 0.0

    tasks = []
    for i, item in enumerate(items):
        canonical = from_protocol_item(item)
        tasks.append(Task(
            task_id=canonical["task_id"] or f"q{i}",
            benchmark=canonical["benchmark"] or "math500",
            prompt=canonical["prompt"],
            answer=canonical["reference"],
        ))

    correct = 0
    total_turns = 0

    async with httpx.AsyncClient() as client:
        trajs = await asyncio.gather(*[
            run_trajectory(t, policy, pool, pool_models, sample=False, client=client,
                           max_turns=max_turns, max_tokens=max_tokens, reasoning="minimal")
            for t in tasks
        ], return_exceptions=True)

    for i, result in enumerate(trajs):
        if isinstance(result, BaseException):
            total_turns += max_turns
            continue
        traj = result
        adapter = get_adapter(traj.task.benchmark or "math500")
        if adapter.score_trajectory(traj) > 0.0:
            correct += 1
        total_turns += traj.n_turns

    n = len(tasks)
    return correct / n, total_turns / n


# ==========================================================================
# Scoring
# ==========================================================================

def _compute_score(hidden_acc: float, live_acc: float, avg_turns: float,
                   novelty: float) -> float:
    """Composite score: 70% hidden + 15% live + 10% efficiency + 5% novelty."""
    max_turns = 5
    efficiency = max(0.0, (max_turns - avg_turns) / (max_turns - 1)) * live_acc if live_acc > 0 else 0.0
    return 0.70 * hidden_acc + 0.15 * live_acc + 0.10 * efficiency + 0.05 * novelty


# ==========================================================================
# Gate 7: Novelty Computation
# ==========================================================================

def _king_submission_dir(
    benchmark: str,
    lb: dict,
    submissions_root: Path,
) -> Optional[Path]:
    """Return the reigning submission directory, if any.

    Checks the composite ``competition`` king first (the single king across all
    benchmarks), then falls back to a per-benchmark king for backward compat.
    """
    # Composite competition: one king across all benchmarks.
    comp = lb.get("competition", {})
    king_miner = comp.get("best_miner")
    king_gen = comp.get("best_generation", 0)
    if not king_miner or not king_gen:
        # Backward compat: per-benchmark king (legacy leaderboard shape).
        bench_entry = lb.get("benchmarks", {}).get(benchmark, {})
        king_miner = bench_entry.get("best_miner")
        king_gen = bench_entry.get("best_generation", 0)
    if not king_miner or not king_gen:
        return None
    king_dir = submissions_root / str(king_miner) / str(king_gen)
    if not (king_dir / "head_weights.npy").exists() or not (king_dir / "svf_scales.npy").exists():
        return None
    return king_dir


def _routing_decisions(policy, items: List[dict], *, ref_count: int) -> List[tuple]:
    """Collect turn-1 routing decisions via the configured policy."""
    from trinity.adapters.hidden_item import from_protocol_item
    from trinity.orchestration.session import routing_transcript

    decisions: List[tuple] = []
    for item in items[:ref_count]:
        canonical = from_protocol_item(item)
        transcript = routing_transcript(canonical["prompt"])
        agent_idx, role = policy.decide(transcript, sample=False)
        decisions.append((agent_idx, role))
    return decisions


def _compute_novelty(
    benchmark: str,
    policy,
    spec,
    eval_items: List[dict],
) -> float:
    """Compare submitter vs benchmark king using full policy state (head + SVF)."""
    from trinity.novelty import NEUTRAL_NOVELTY, novelty_score

    lb = _load_leaderboard()
    king_dir = _king_submission_dir(benchmark, lb, _REPO / "submissions")
    ref_count = min(50, len(eval_items))
    if king_dir is None or ref_count == 0:
        return NEUTRAL_NOVELTY

    try:
        king_hw = np.load(str(king_dir / "head_weights.npy"))
        king_svf = np.load(str(king_dir / "svf_scales.npy"))
    except (ValueError, OSError):
        return NEUTRAL_NOVELTY

    submitter_decisions = _routing_decisions(policy, eval_items, ref_count=ref_count)
    king_theta = np.concatenate([
        np.asarray(king_hw, dtype=np.float64).ravel(),
        np.asarray(king_svf, dtype=np.float64).ravel(),
    ])
    policy.configure(king_theta, spec)
    king_decisions = _routing_decisions(policy, eval_items, ref_count=ref_count)

    return novelty_score(submitter_decisions, king_decisions)


# ==========================================================================
# Leaderboard
# ==========================================================================

def _load_leaderboard() -> dict:
    lb_path = _REPO / "leaderboard.json"
    if lb_path.exists():
        return json.loads(lb_path.read_text())
    return {"benchmarks": {}}


def _empty_bench_entry() -> dict:
    return {
        "best_score": 0.0,
        "best_miner": None,
        "best_generation": 0,
        "best_pr": None,
        "baseline_random": None,
        "best_single_model": None,
        "oracle_ceiling": None,
        "history": [],
        "attempts": [],
    }


def _record_attempt(benchmark: str, miner_name: str, generation: int,
                    pr_number: int) -> None:
    """Consume one daily submission slot for ``miner_name`` on ``benchmark``.

    Called as soon as Gate 1 passes so later gate failures and score-rejections
    still count toward the rate limit (SUBMITTING.md: 1 submission / day).
    On first write, seeds ``attempts`` from legacy win-only ``history`` so a
    recent winner remains rate-limited after this change rolls out.
    """
    lb = _load_leaderboard()
    # Write into benchmarks.composite so check_rate_limit (which reads
    # lb["benchmarks"][benchmark]) finds the attempts. The competition
    # metadata lives under lb["competition"]; rate-limiting lives here.
    bench_entry = lb.setdefault("benchmarks", {}).setdefault(
        benchmark, _empty_bench_entry(),
    )
    if "attempts" not in bench_entry:
        # Seed from legacy win-only history so a recent winner stays rate-limited
        # after this change rolls out (backward compat for pre-attempts leaderboards).
        bench_entry["attempts"] = [
            dict(entry) for entry in bench_entry.get("history", [])
        ]
    already = any(
        e.get("miner") == miner_name and e.get("pr") == pr_number
        for e in bench_entry["attempts"]
    )
    if already:
        print(f"[pr_eval] attempt for PR #{pr_number} already recorded; not double-counting")
        return
    bench_entry["attempts"].append({
        "miner": miner_name,
        "generation": generation,
        "pr": pr_number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    lb["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (_REPO / "leaderboard.json").write_text(json.dumps(lb, indent=2) + "\n")
    print(f"[pr_eval] attempt recorded for rate limit ({miner_name}/{generation})")


def _update_leaderboard(miner_name: str, generation: int,
                         pr_number: int, composite_score: float,
                         per_benchmark: dict) -> None:
    """Update leaderboard.json with a new winning composite submission."""
    lb = _load_leaderboard()
    comp = lb.setdefault("competition", {})
    comp["best_composite_score"] = round(composite_score, 4)
    comp["best_miner"] = miner_name
    comp["best_generation"] = generation
    comp["best_pr"] = pr_number
    comp["best_per_benchmark"] = {k: v["score"] for k, v in per_benchmark.items()}
    comp["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    comp.setdefault("history", []).append({
        "miner": miner_name,
        "generation": generation,
        "score": round(composite_score, 4),
        "per_benchmark": {k: v["score"] for k, v in per_benchmark.items()},
        "pr": pr_number,
        "merged": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    lb["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (_REPO / "leaderboard.json").write_text(json.dumps(lb, indent=2) + "\n")
    print(f"[pr_eval] leaderboard.json updated — new king: {miner_name} ({composite_score:.4f})")


# ==========================================================================
# Main
# ==========================================================================

async def evaluate_pr(pr_number: int, benchmark: str,
                       submission_subpath: str) -> dict:
    """Evaluate a PR submission against ALL hidden benchmarks with all 8 gates.

    The head is evaluated on every benchmark in COMPETITION_BENCHMARKS
    (math500, mmlu, livecodebench). The composite score = mean of the
    per-benchmark scores. A new king must beat the previous king's composite
    by >= WIN_MARGIN (0.02).
    """
    benchmarks = list(COMPETITION_BENCHMARKS)
    print(f"\n{'='*60}")
    print(f"[pr_eval] PR #{pr_number} — composite ({', '.join(benchmarks)})")
    print(f"{'='*60}")

    # ---- Parse submission path ----
    sub_dir = _REPO / "submissions" / submission_subpath
    if not sub_dir.exists():
        return _reject("submission_not_found", f"Directory not found: submissions/{submission_subpath}")
    if not (sub_dir / "head_weights.npy").exists():
        return _reject("submission_incomplete", f"No head_weights.npy in submissions/{submission_subpath}")

    parts = sub_dir.relative_to(_REPO / "submissions").parts
    miner_name = parts[0] if len(parts) >= 1 else "unknown"
    generation = int(parts[1]) if len(parts) >= 2 else 0

    print(f"[pr_eval] Miner: {miner_name}  Generation: {generation}")
    print(f"[pr_eval] Path: submissions/{submission_subpath}")

    # ---- Load submission ----
    loaded = _load_submission(sub_dir)
    if loaded is None:
        return _reject("load_failed", "Failed to load submission files")
    head_W, svf_scales, receipt = loaded

    # ══════════════════════════════════════════════════════════════
    # GATE 1: Rate Limiting (before any GPU/API work)
    # ══════════════════════════════════════════════════════════════
    lb = _load_leaderboard()
    err = _check_rate_limit(miner_name, "composite", lb, current_pr=pr_number)
    if err:
        return _reject(err)
    # Slot consumed even if later gates fail or the score loses — otherwise
    # rejected attempts never counted and miners could probe daily.
    _record_attempt("composite", miner_name, generation, pr_number)

    # ══════════════════════════════════════════════════════════════
    # GATE 2: Weight Validation (NaN, Inf, extreme values, untrained)
    # ══════════════════════════════════════════════════════════════
    err = _validate_weights(head_W, svf_scales)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 3: Duplicate Detection (cosine similarity vs all history)
    # ══════════════════════════════════════════════════════════════
    submissions_root = _REPO / "submissions"
    err = _check_duplicate(head_W, svf_scales, submissions_root, miner_name, generation,
                           leaderboard=lb, load_leaderboard=_load_leaderboard)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 4: Receipt Cross-Validation
    # ══════════════════════════════════════════════════════════════
    err = _validate_receipt(receipt)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 5: Ledger / receipt cost consistency (offline, no GPU)
    # ══════════════════════════════════════════════════════════════
    ledger_path = os.environ.get("TRINITY_COST_LEDGER")
    err = validate_ledger_receipt_cost(receipt, ledger_path)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 6: Receipt schema / benchmark consistency (offline)
    # ══════════════════════════════════════════════════════════════
    err = validate_pack_schema(receipt, "composite")
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 7: Theta pack/unpack integrity (offline)
    # ══════════════════════════════════════════════════════════════
    err = validate_theta_integrity(head_W, svf_scales)
    if err:
        return _reject(err)

    print("[pr_eval] All 4 pre-eval gates passed ✓\n")

    # Advisories (issue #208): report-only signals that NEVER reject. The gate chain was
    # deliberately relaxed for launch (4bb03a7) "to attract miners, not repel them", so
    # these only inform: a fabricated fitness curve is worth flagging but not blocking, the
    # ledger is not run-scoped, and a collapsed router is a valid (if weak) submission
    # under the score-based contract.
    for _name, _warn in (
        ("fitness_history_sequence", validate_fitness_history_sequence(receipt)),
        ("ledger_call_volume", audit_ledger_call_volume(receipt, ledger_path)),
        ("head_routing_diversity", audit_head_routing_diversity(head_W)),
    ):
        if _warn:
            print(f"[pr_eval] [WARN] {_name}: {_warn}")

    # ══════════════════════════════════════════════════════════════
    # Load encoder ONCE (shared across all benchmarks)
    # ══════════════════════════════════════════════════════════════
    from trinity.coordinator.policy import CoordinatorPolicy
    from trinity.llm.openrouter_client import OpenRouterPool

    cfg = yaml.safe_load((_REPO / "configs" / "trinity.yaml").read_text())
    cc = cfg["coordinator"]

    print("[pr_eval] Loading encoder on GPU...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cc["encoder_model"],
        device=cc.get("device", "cuda:0"),
        dtype=cc.get("dtype", "bfloat16"),
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=3, n_roles=3,
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )
    policy.configure(np.concatenate([
        np.asarray(head_W, dtype=np.float64).ravel(),
        np.asarray(svf_scales, dtype=np.float64).ravel(),
    ]), spec)

    pool = OpenRouterPool(str(_REPO / "configs" / "models.yaml"))

    # ══════════════════════════════════════════════════════════════
    # Evaluate on EACH benchmark
    # ══════════════════════════════════════════════════════════════
    per_benchmark: dict[str, dict] = {}
    for bench in benchmarks:
        print(f"\n[pr_eval] --- {bench} ---")
        eval_items, audit_items, live_items = _load_hidden_benchmark(bench)
        print(f"  {len(eval_items)} eval + {len(audit_items)} audit + {len(live_items)} live")

        t0 = time.time()
        hidden_acc = _evaluate_cached(policy, eval_items, _POOL_MODELS)
        print(f"  hidden_acc = {hidden_acc:.4f}  ({time.time() - t0:.1f}s)")

        audit_acc = _evaluate_cached(policy, audit_items, _POOL_MODELS) if audit_items else hidden_acc
        print(f"  audit_acc  = {audit_acc:.4f}")
        gap = hidden_acc - audit_acc
        overfit_penalty = 1.0
        if gap > _OVERFIT_HARD_REJECT:
            return _reject(f"overfit_rejected ({bench}): eval-audit gap {gap:.4f} > {_OVERFIT_HARD_REJECT}")
        elif gap > _OVERFIT_PENALTY:
            overfit_penalty = 0.85
            print(f"  overfit_penalty = 0.85 (gap {gap:.4f})")
        else:
            print(f"  eval-audit gap = {gap:.4f} (clean)")

        print(f"  live eval ({len(live_items)} questions, real API)...")
        live_acc, avg_turns = await _evaluate_live(policy, pool, _POOL_MODELS, live_items)
        print(f"  live_acc   = {live_acc:.4f}  avg_turns = {avg_turns:.2f}")

        novelty = _compute_novelty(bench, policy, spec, eval_items) if bench == benchmarks[0] else 0.0
        if novelty:
            print(f"  novelty    = {novelty:.4f}")

        bench_score = _compute_score(hidden_acc, live_acc, avg_turns, novelty) * overfit_penalty
        print(f"  bench_score = {bench_score:.4f}")
        per_benchmark[bench] = {
            "score": round(bench_score, 4),
            "hidden_acc": round(hidden_acc, 4),
            "audit_acc": round(audit_acc, 4),
            "live_acc": round(live_acc, 4),
            "avg_turns": round(avg_turns, 2),
        }

    # ══════════════════════════════════════════════════════════════
    # Composite score + margin check
    # ══════════════════════════════════════════════════════════════
    composite_score = sum(v["score"] for v in per_benchmark.values()) / len(per_benchmark)
    print(f"\n[pr_eval] Composite: {composite_score:.4f}")
    for b, v in per_benchmark.items():
        print(f"  {b:20s} = {v['score']:.4f}")

    lb = _load_leaderboard()
    comp = lb.get("competition", {})
    best_composite = comp.get("best_composite_score", 0.0)
    margin = comp.get("win_margin", WIN_MARGIN)

    print(f"\n  Current king:  {best_composite:.4f}")
    print(f"  Submission:    {composite_score:.4f}")
    print(f"  Delta:         {composite_score - best_composite:+.4f}")
    print(f"  Win margin:    >= {margin:.4f}")
    clears = composite_score >= best_composite + margin
    print(f"  Clears margin: {'YES' if clears else 'NO'}")

    if clears:
        print(f"\n  *** APPROVED *** — beats king by {composite_score - best_composite:+.4f}")
        _update_leaderboard(miner_name, generation, pr_number,
                            composite_score, per_benchmark)
        report = _format_pr_report(miner_name, composite_score, best_composite,
                                    composite_score - best_composite, per_benchmark)
        print(f"\n{report}")
        return {
            "approved": True,
            "score": round(composite_score, 4),
            "best_score": round(composite_score, 4),
            "delta": round(composite_score - best_composite, 4),
            "per_benchmark": per_benchmark,
            "report": report,
            "message": f"APPROVED: composite {composite_score:.4f} beats king {best_composite:.4f} by {composite_score - best_composite:+.4f} (margin {margin:.4f})",
        }
    else:
        print(f"\n  *** REJECTED *** — does not clear margin {margin:.4f} vs king {best_composite:.4f}")
        return {
            "approved": False,
            "score": round(composite_score, 4),
            "best_score": best_composite,
            "delta": round(composite_score - best_composite, 4),
            "message": f"REJECTED: composite {composite_score:.4f} does not beat king {best_composite:.4f} by margin {margin:.4f} (delta: {composite_score - best_composite:+.4f})",
        }


# ==========================================================================
# Helpers
# ==========================================================================

def _reject(reason: str, detail: str = "") -> dict:
    """Return a rejection result. Never leaks component scores."""
    msg = f"GATE_FAILED: {reason}"
    if detail:
        msg += f" — {detail}"
    print(f"[pr_eval] {msg}")
    return {
        "approved": False,
        "score": 0.0,
        "best_score": None,
        "delta": None,
        "message": msg,
    }


def _format_pr_report(miner_name: str, composite: float, best_prev: float,
                       delta: float, per_benchmark: dict) -> str:
    """Format a per-benchmark Markdown report for a winning submission."""
    lines = [
        f"## 🏆 New King: {miner_name}",
        "",
        f"**Composite: {composite:.4f}** (beats previous king {best_prev:.4f} by {delta:+.4f})",
        "",
        "| Benchmark | Score | Cached | Live | Avg Turns |",
        "|---|---:|---:|---:|---:|",
    ]
    for bench, v in sorted(per_benchmark.items()):
        lines.append(
            f"| {bench} | {v['score']:.4f} | {v['hidden_acc']:.4f} | "
            f"{v['live_acc']:.4f} | {v['avg_turns']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a TinyRouter PR submission against the hidden benchmark"
    )
    ap.add_argument("--pr", type=int, required=True, dest="pr_number",
                    help="GitHub PR number")
    ap.add_argument("--benchmark", default="composite",
                    help="(deprecated, kept for CI compat — the competition is composite)")
    ap.add_argument("--submission", required=True, dest="submission_subpath",
                    help="Path to submission relative to repo root (e.g. alice/1)")
    args = ap.parse_args()

    result = asyncio.run(evaluate_pr(
        pr_number=args.pr_number,
        benchmark=args.benchmark,
        submission_subpath=args.submission_subpath,
    ))

    print(f"\n[pr_eval] RESULT:")
    print(json.dumps(result, indent=2))

    if not result["approved"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
