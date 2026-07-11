#!/usr/bin/env python3
"""Evaluate a PR submission against the hidden benchmark.

This is the maintainer's tool. It evaluates a miner's submitted routing head
against the HIDDEN benchmark (stored OUTSIDE the repo — never committed) and
determines whether the head beats the current best accuracy.

All 8 anti-cheat gates run BEFORE any GPU work or API calls. A failing gate
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
    DEFAULT_POOL_MODELS,
    DUPLICATE_HEAD_COSINE_THRESHOLD,
    EXPECTED_TOTAL_PARAMS,
    MAX_WEIGHT_MAGNITUDE,
    MIN_TRAINING_COST_USD,
    N_HEAD_MODELS,
    RATE_LIMIT_MAX_SUBMISSIONS,
    RATE_LIMIT_WINDOW_DAYS,
)
from trinity.submission.gates import (
    check_duplicate as _check_duplicate,
    check_rate_limit as _check_rate_limit,
    cosine_similarity as _cosine_similarity,
    parse_utc_timestamp as _parse_utc_timestamp,
    rate_limit_entries as _rate_limit_entries,
    routing_invariant_head as _routing_invariant_head,
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
_OVERFIT_HARD_REJECT = 0.10
_OVERFIT_PENALTY = 0.05
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
    """Evaluate a configured policy on cached benchmark items. Returns accuracy [0, 1]."""
    from trinity.adapters.hidden_item import from_protocol_item
    from trinity.orchestration.reward import score_text
    from trinity.orchestration.session import routing_transcript

    if not items:
        return 0.0

    correct = 0
    for item in items:
        canonical = from_protocol_item(item)
        agent_idx, _role = policy.decide(routing_transcript(canonical["prompt"]), sample=False)
        model_name = pool_model_names[agent_idx % len(pool_model_names)]
        cached = canonical["cached_model_answers"].get(model_name, "")
        if score_text(canonical["benchmark"] or "math500", cached, canonical["reference"]) > 0.0:
            correct += 1

    return correct / len(items)


# ==========================================================================
# Live evaluation
# ==========================================================================

async def _evaluate_live(
    policy, pool, pool_models, items: List[dict],
    max_turns: int = 5, max_tokens: int = 4096,
) -> Tuple[float, float]:
    """Run full multi-turn eval with real API calls. Returns (accuracy, avg_turns)."""
    import httpx
    from trinity.adapters.hidden_item import from_protocol_item
    from trinity.orchestration.reward import score
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
        if score(traj) > 0.0:
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
    """Return the reigning submission directory for ``benchmark``, if any."""
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
    """Consume one weekly submission slot for ``miner_name`` on ``benchmark``.

    Called as soon as Gate 1 passes so later gate failures and score-rejections
    still count toward the rate limit (SUBMITTING.md: 1 submission / week).
    On first write, seeds ``attempts`` from legacy win-only ``history`` so a
    recent winner remains rate-limited after this change rolls out.
    """
    lb = _load_leaderboard()
    bench_entry = lb.setdefault("benchmarks", {}).setdefault(
        benchmark, _empty_bench_entry(),
    )
    if "attempts" not in bench_entry:
        bench_entry["attempts"] = [
            dict(entry) for entry in bench_entry.get("history", [])
        ]
    # Idempotent per PR: a re-run of the same PR (CI retry, transient failure
    # after this point on a prior run) must not append a second attempt and
    # inflate the miner's weekly count. A distinct PR always records.
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


def _update_leaderboard(benchmark: str, miner_name: str, generation: int,
                         pr_number: int, score: float, hidden_acc: float,
                         live_acc: float, avg_turns: float) -> None:
    """Update leaderboard.json with a new winning submission."""
    lb = _load_leaderboard()
    bench_entry = lb.setdefault("benchmarks", {}).setdefault(
        benchmark, _empty_bench_entry(),
    )

    bench_entry["best_score"] = round(score, 4)
    bench_entry["best_miner"] = miner_name
    bench_entry["best_generation"] = generation
    bench_entry["best_pr"] = pr_number
    bench_entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    bench_entry.setdefault("history", []).append({
        "miner": miner_name,
        "generation": generation,
        "score": round(score, 4),
        "pr": pr_number,
        "merged": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    lb["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (_REPO / "leaderboard.json").write_text(json.dumps(lb, indent=2) + "\n")
    print(f"[pr_eval] leaderboard.json updated")


# ==========================================================================
# Main
# ==========================================================================

async def evaluate_pr(pr_number: int, benchmark: str,
                       submission_subpath: str) -> dict:
    """Evaluate a PR submission against the hidden benchmark with all 8 gates."""
    print(f"\n{'='*60}")
    print(f"[pr_eval] PR #{pr_number} — {benchmark}")
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
    err = _check_rate_limit(miner_name, benchmark, lb, current_pr=pr_number)
    if err:
        return _reject(err)
    # Slot consumed even if later gates fail or the score loses — otherwise
    # rejected attempts never counted and miners could probe weekly.
    _record_attempt(benchmark, miner_name, generation, pr_number)

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
    err = validate_pack_schema(receipt, benchmark)
    if err:
        return _reject(err)

    # ══════════════════════════════════════════════════════════════
    # GATE 7: Theta pack/unpack integrity (offline)
    # ══════════════════════════════════════════════════════════════
    err = validate_theta_integrity(head_W, svf_scales)
    if err:
        return _reject(err)

    print("[pr_eval] All 7 pre-eval gates passed ✓\n")

    # ══════════════════════════════════════════════════════════════
    # Load benchmark + encoder (GPU work starts here)
    # ══════════════════════════════════════════════════════════════
    eval_items, audit_items, live_items = _load_hidden_benchmark(benchmark)
    print(f"[pr_eval] Loaded encrypted benchmark: {len(eval_items)} eval + "
          f"{len(audit_items)} audit + {len(live_items)} live questions")

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

    # ---- Cached eval ----
    print("[pr_eval] Running cached eval (150 questions)...")
    t0 = time.time()
    hidden_acc = _evaluate_cached(policy, eval_items, _POOL_MODELS)
    print(f"  hidden_acc = {hidden_acc:.4f}  ({time.time() - t0:.1f}s)")

    # ---- Audit eval ----
    print("[pr_eval] Running audit eval (50 questions)...")
    audit_acc = _evaluate_cached(policy, audit_items, _POOL_MODELS) if audit_items else hidden_acc
    print(f"  audit_acc  = {audit_acc:.4f}")

    # ══════════════════════════════════════════════════════════════
    # GATE 5: Overfit Rejection (HARD gate — not just a warning)
    # ══════════════════════════════════════════════════════════════
    gap = hidden_acc - audit_acc
    overfit_penalty = 1.0
    if gap > _OVERFIT_HARD_REJECT:
        return _reject(f"overfit_rejected: eval-audit gap {gap:.4f} > {_OVERFIT_HARD_REJECT}")
    elif gap > _OVERFIT_PENALTY:
        overfit_penalty = 0.85
        print(f"  overfit_penalty = 0.85 (gap {gap:.4f} > {_OVERFIT_PENALTY})")
    else:
        print(f"  eval-audit gap = {gap:.4f} (clean)")

    # ---- Live eval ----
    print("[pr_eval] Running live multi-turn eval (20 questions, real API calls)...")
    pool = OpenRouterPool(str(_REPO / "configs" / "models.yaml"))
    live_acc, avg_turns = await _evaluate_live(policy, pool, _POOL_MODELS, live_items)
    print(f"  live_acc   = {live_acc:.4f}  avg_turns = {avg_turns:.2f}")

    # ══════════════════════════════════════════════════════════════
    # GATE 7: Compute actual novelty
    # ══════════════════════════════════════════════════════════════
    novelty = _compute_novelty(benchmark, policy, spec, eval_items)
    print(f"  novelty    = {novelty:.4f}")

    # ---- Composite score ----
    score = _compute_score(hidden_acc, live_acc, avg_turns, novelty)
    score *= overfit_penalty
    print(f"  composite  = {score:.4f}")

    # ---- Compare to leaderboard ----
    lb = _load_leaderboard()
    best_score = lb.get("benchmarks", {}).get(benchmark, {}).get("best_score", 0.0)

    print(f"\n  Current best: {best_score:.4f}")
    print(f"  Submission:   {score:.4f}")
    print(f"  Delta:        {score - best_score:+.4f}")

    if score > best_score:
        print(f"\n  *** APPROVED *** — beats current best by {score - best_score:+.4f}")
        _update_leaderboard(benchmark, miner_name, generation, pr_number,
                            score, hidden_acc, live_acc, avg_turns)
        return {
            "approved": True,
            "score": round(score, 4),
            "best_score": round(score, 4),
            "delta": round(score - best_score, 4),
            "message": f"APPROVED: score {score:.4f} beats current best {best_score:.4f} by {score - best_score:+.4f}",
        }
    else:
        # ══════════════════════════════════════════════════════════════
        # GATE 6: Minimize score feedback on rejection
        # Only reveal composite score + delta. NEVER reveal component scores.
        # ══════════════════════════════════════════════════════════════
        print(f"\n  *** REJECTED *** — does not beat current best ({best_score:.4f})")
        print(f"  [internal] hidden={hidden_acc:.4f} audit={audit_acc:.4f} "
              f"live={live_acc:.4f} novelty={novelty:.4f} turns={avg_turns:.2f}")
        return {
            "approved": False,
            "score": round(score, 4),
            "best_score": best_score,
            "delta": round(score - best_score, 4),
            "message": f"REJECTED: score {score:.4f} does not beat current best {best_score:.4f} (delta: {score - best_score:+.4f})",
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a TinyRouter PR submission against the hidden benchmark"
    )
    ap.add_argument("--pr", type=int, required=True, dest="pr_number",
                    help="GitHub PR number")
    ap.add_argument("--benchmark", default="math500",
                    help="Benchmark name (math500 or mmlu)")
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
