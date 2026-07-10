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
import sys
import time
from pathlib import Path

import numpy as np
import yaml

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
        "seed": summary.get("seed", 0),
        "packed_at": int(time.time()),
    }


def extract_head_and_svf(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the coordinator, install best theta, extract head + SVF."""
    from trinity.coordinator.policy import CoordinatorPolicy
    from trinity.coordinator import params as P

    cfg = yaml.safe_load((_REPO / "configs" / "trinity.yaml").read_text())
    cc = cfg["coordinator"]

    print("Loading encoder on GPU (this may take a moment)...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cc["encoder_model"],
        device=cc.get("device", "cuda:0"),
        dtype=cc.get("dtype", "bfloat16"),
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=3,
        n_roles=3,
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )

    theta_path = run_dir / "best_theta.npy"
    if not theta_path.exists():
        # Try without .npy extension patterns
        candidates = sorted(run_dir.glob("best_theta*"))
        if not candidates:
            raise FileNotFoundError(f"No best_theta found in {run_dir}")
        theta_path = candidates[-1]

    theta = np.load(str(theta_path))
    policy.configure(theta, spec)

    # Extract head weight from the LinearHead module
    head_W = policy.head.weight.detach().cpu().float().numpy()

    # Extract SVF scales
    try:
        svf_scales = policy.svf.current_scales()
    except AttributeError:
        svf_scales = np.ones(spec.n_svf, dtype=np.float32)

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
        submissions_dir = _REPO / "submissions" / miner_name
        existing = sorted(submissions_dir.glob("*")) if submissions_dir.exists() else []
        gen = len(existing) + 1

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
