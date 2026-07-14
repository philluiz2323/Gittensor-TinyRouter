#!/usr/bin/env python3
"""Pack a trained routing head for submission to TinyRouter.

Extracts head weights + SVF scales from a trained best_theta.npy, builds a
training receipt, and writes the submission files ready for a PR.

Usage:
    python scripts/pack_submission.py \
        --run-dir experiments/math500/my-run \
        --miner-name alice \
        --benchmark math500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _estimate_cost() -> float:
    """Estimate training cost from the hash-chain cost ledger."""
    ledger_path = os.environ.get("TRINITY_COST_LEDGER")
    if not ledger_path:
        return 0.0
    from trinity.llm.openrouter_pricing import verified_ledger_total_usd

    total = verified_ledger_total_usd(ledger_path)
    return round(total, 4) if total is not None else 0.0


def _resolve_seed(summary: dict) -> int | None:
    """Recover the training seed from a run summary, or ``None`` if unrecorded.

    Runs packed before issue #109 have no ``seed`` key. Report those as ``None``
    rather than defaulting to ``0``: a real ``--seed 0`` run is a different claim
    from "the seed was never recorded", and pycma historically read seed ``0`` as
    "seed from the wall clock" (issue #38).

    Args:
        summary: Parsed ``summary.json`` for the run, possibly empty.

    Returns:
        The recorded seed, or ``None`` when the run predates seed recording.
    """
    seed = summary.get("seed")
    if seed is None:
        print(
            "WARNING: summary.json has no 'seed' — this run predates seed recording "
            "(issue #109). Recording seed=null; the fitness curve cannot be re-derived. "
            "Re-run training to produce a receipt with full provenance.",
            file=sys.stderr,
        )
        return None
    return int(seed)


def build_receipt(run_dir: Path, benchmark: str) -> dict:
    """Build a training receipt from run artifacts."""
    summary = _load_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    history = _load_json(run_dir / "history.json") if (run_dir / "history.json").exists() else []

    return {
        "benchmark": benchmark,
        "pool_models": summary.get("pool", ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"]),
        "n_total": summary.get("n_total", 13312),
        "popsize": summary.get("popsize", 33),
        "m_cma": summary.get("m_cma", 16),
        "generations": summary.get("generations", 60),
        "best_fitness": summary.get("best_fitness", 0.0),
        "fitness_history": [
            {"generation": h["generation"], "mean_fitness": h["gen_mean_fitness"],
             "max_fitness": h["gen_max_fitness"], "best_fitness": h["best_fitness"]}
            for h in history
        ],
        "total_cost_usd": _estimate_cost(),
        "seed": _resolve_seed(summary),
        "packed_at": int(time.time()),
    }


def next_generation(submissions_dir: Path) -> int:
    """Pick the next submission generation number for a miner directory.

    Returns ``max(existing numeric generation) + 1``, considering ONLY
    numerically-named subdirectories — so a gap in the numbering (a thrown-out
    training run whose generation directory was deleted) or a stray non-generation
    entry never causes a collision. A missing/empty directory yields ``1``.

    Counting entries instead (``len(existing) + 1``) silently overwrites an
    existing generation whenever the numbering is not contiguous: with gens
    ``1`` and ``3`` present, a count gives ``3`` and clobbers the real gen 3,
    whereas the correct next generation is ``4``.

    Args:
        submissions_dir: ``submissions/<miner-name>/`` (need not exist).

    Returns:
        The next free generation number (>= 1).
    """
    if not submissions_dir.exists():
        return 1
    nums = [
        int(p.name)
        for p in submissions_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return max(nums) + 1 if nums else 1


def _theta_generation(path: Path) -> tuple[int, str]:
    """Sort key selecting the latest-generation ``best_theta*`` file.

    Parses the integer generation from names like ``best_theta_gen11.npy`` so gen 11
    ranks above gen 9 — a plain string sort orders ``'gen11' < 'gen9'`` and would pick an
    OLDER generation. A file with no encoded generation falls back to name order
    (preserving the previous lexicographic pick when no generation is present). Mirrors
    the repo's integer-generation convention (``gates._same_generation``, #262).
    """
    m = re.search(r"gen(\d+)", path.stem) or re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else -1, path.name)


def extract_head_and_svf(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract head weights + SVF scales from best_theta.npy via θ unpack."""
    from trinity.coordinator import params as P

    spec = P.make_spec(n_a=6, d_h=1024, n_svf=P.DEFAULT_N_SVF)

    theta_path = run_dir / "best_theta.npy"
    if not theta_path.exists():
        candidates = list(run_dir.glob("best_theta*"))
        if not candidates:
            raise FileNotFoundError(f"No best_theta found in {run_dir}")
        # Pick the numerically-latest generation, not the lexicographically-last name:
        # sorted(...)[-1] returns 'best_theta_gen9' over 'best_theta_gen11', packing an
        # OLDER generation's weights into the submission.
        theta_path = max(candidates, key=_theta_generation)

    theta = np.load(str(theta_path))
    head_W, svf_scales = P.unpack(theta, spec)
    return head_W.astype(np.float32), svf_scales.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack a trained routing head for submission")
    ap.add_argument("--run-dir", required=True, dest="run_dir",
                    help="Path to the training run directory (e.g. experiments/math500/my-run)")
    ap.add_argument("--miner-name", required=True, dest="miner_name",
                    help="Your miner name (used in submission path)")
    ap.add_argument("--benchmark", default="math500",
                    help="Benchmark name (math500 or mmlu)")
    ap.add_argument("--generation", type=int, default=0, dest="generation",
                    help="Submission generation number (auto-detected if 0)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        sys.exit(1)

    miner_name = args.miner_name.lower().replace(" ", "-")
    benchmark = args.benchmark

    # Auto-detect generation number
    gen = args.generation
    if gen == 0:
        gen = next_generation(_REPO / "submissions" / miner_name)

    # Create submission directory
    sub_dir = _REPO / "submissions" / miner_name / str(gen)
    sub_dir.mkdir(parents=True, exist_ok=True)
    print(f"Packing submission to: {sub_dir}")

    # Extract head + SVF
    print("Extracting head weights + SVF scales...")
    head_W, svf_scales = extract_head_and_svf(run_dir)

    # Save artifacts
    np.save(str(sub_dir / "head_weights.npy"), head_W)
    np.save(str(sub_dir / "svf_scales.npy"), svf_scales)
    print(f"  head_weights.npy: {head_W.shape} {head_W.dtype}")
    print(f"  svf_scales.npy:   {svf_scales.shape} {svf_scales.dtype}")

    # Build and save receipt
    receipt = build_receipt(run_dir, benchmark)
    (sub_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"  receipt.json:     cost=${receipt['total_cost_usd']:.2f}, "
          f"best_fitness={receipt['best_fitness']:.4f}")

    # Write README for the submission
    readme = f"""# Submission: {miner_name} generation {gen}

- **Miner:** {miner_name}
- **Benchmark:** {benchmark}
- **Generation:** {gen}
- **Training cost:** ${receipt['total_cost_usd']:.2f}
- **Best fitness:** {receipt['best_fitness']:.4f}
- **Generations:** {receipt['generations']}
- **Population size:** {receipt['popsize']}
- **Packed at:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(receipt['packed_at']))}

## Files
- `head_weights.npy` — linear head W (6 × 1024) float32
- `svf_scales.npy` — SVF singular-value scales (7168,) float32
- `receipt.json` — training metadata

## PR Instructions
1. Create a branch: `git checkout -b submission/{miner_name}-gen{gen}`
2. Add this directory: `git add submissions/{miner_name}/{gen}/`
3. Commit and push
4. Open a PR with title: `[submission] {miner_name} gen {gen} — {benchmark}`
"""
    (sub_dir / "README.md").write_text(readme)

    print(f"\nSubmission ready at: submissions/{miner_name}/{gen}/")
    print(f"Next step: open a PR with title '[submission] {miner_name} gen {gen} — {benchmark}'")


if __name__ == "__main__":
    main()
