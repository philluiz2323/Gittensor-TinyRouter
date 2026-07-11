# Submitting a Routing Head

TinyRouter is a **routing accuracy competition**. Miners train routing heads via
CMA-ES (or any method), submit them as PRs, and earn TAO through
[Gittensor](https://github.com/entrius/gittensor) when their PR is merged.

**A PR is only merged if the submitted head beats the current best accuracy
on the maintainer's hidden benchmark.** This document explains how to compete.

## Overview

```
  ┌──────────┐      ┌──────────────┐      ┌──────────┐      ┌────────┐
  │ 1. TRAIN │ ──► │ 2. PACK      │ ──► │ 3. SUBMIT│ ──► │ 4. EARN│
  │  CMA-ES  │      │  extract     │      │  open PR │      │  TAO   │
  │  on GPU  │      │  head + SVF  │      │  to repo │      │ (if win)│
  └──────────┘      └──────────────┘      └──────────┘      └────────┘
```

## Step 1: Train

Train a routing head using the existing CMA-ES pipeline. You need:
- A GPU (T4 or better) or CPU (slower)
- An OpenRouter API key (`OPENROUTER_API_KEY` env var)
- ~$25-65 in API credits for a full training run

```bash
git clone https://github.com/<org>/tinyrouter.git
cd tinyrouter
pip install -e ".[dev]"
source ~/.config/trinity/secrets.env   # exports OPENROUTER_API_KEY

# Train on math500
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmark math500 \
    --run-name my-submission \
    --generations 60
```

The run saves artifacts to `experiments/math500/my-submission/`:
- `best_theta.npy` — the trained parameter vector (13,312 floats)
- `history.json` — per-generation fitness values
- `summary.json` — run metadata

## Step 2: Pack

Extract the head weights and SVF scales from your trained theta, and build a
training receipt:

```bash
python scripts/pack_submission.py \
    --run-dir experiments/math500/my-submission \
    --miner-name your-name \
    --benchmark math500
```

This creates:
```
submissions/your-name/1/
├── head_weights.npy    # (6, 1024) float32 — the linear routing head
├── svf_scales.npy      # (7168,) float32 — SVF singular-value scales
├── receipt.json        # training metadata (cost, seed, gens, fitness curve)
└── README.md           # auto-generated submission summary
```

### Step 2b: Preflight (recommended)

Run the same offline anti-cheat gates as `pr_eval` **before** opening a PR.
No GPU and no API calls are required:

```bash
export TRINITY_COST_LEDGER=~/trinity/cost_ledger.jsonl   # optional, gate 5
python scripts/preflight_submission.py \
    --submission your-name/1 \
    --benchmark math500
```

Preflight runs seven offline gates: rate limit, weight sanity, duplicate
detection, receipt plausibility, ledger/receipt cost match (when a ledger is
provided), receipt schema/benchmark consistency, and theta pack/unpack integrity.

## Step 3: Submit

1. **Fork** this repo and create a branch: `git checkout -b submission/your-name-gen1`
2. **Add** the submission directory: `submissions/your-name/1/`
3. **Commit** and push
4. **Open a PR** with title: `[submission] your-name gen 1 — math500`
5. The PR body must include:
   - Benchmark trained on
   - Training method (CMA-ES or other)
   - Approximate API cost
   - Any notable techniques used

The maintainer runs `scripts/pr_eval.py` on your PR, which evaluates your head
against a **hidden benchmark** (not in the repo, never revealed). You will
receive a PR comment with your score.

## Step 4: Earn

- **If your accuracy > current best:** the PR is merged, your name goes on the
  [leaderboard](leaderboard.json), and Gittensor validators see the merged PR →
  you earn TAO.
- **If your accuracy ≤ current best:** the PR is closed with your score and the
  current best shown. You can retrain and try again.

## Rules

| Rule | Detail |
|---|---|
| **Submission rate** | 1 submission per benchmark per week |
| **Original work** | You must train the head yourself. Every submission is checked for duplicate weights (cosine similarity against all previous submissions) and rejected if copied |
| **Hidden benchmark** | The evaluation questions are NEVER revealed. Do not ask for them |
| **Score feedback** | If you win: full results published. If you lose: you receive ONLY your composite score and the current best — no per-component breakdown (prevents benchmark probing) |
| **Receipt required** | Your training receipt must show cost ≥ $15 and a plausible fitness curve. Fabricated receipts are rejected |
| **Multiple benchmarks** | You can submit to math500 and MMLU independently |
| **General PRs** | Docs, bug fixes, and refactors are welcome but do NOT earn TAO — only accuracy-improving routing heads count |

## How scoring works

Your head is evaluated on:
- **Cached single-turn accuracy (70% weight):** 150 hidden questions with
  pre-computed model answers. Zero API cost, deterministic.
- **Live multi-turn accuracy (15% weight):** 20 questions run through the full
  Thinker→Worker→Verifier loop with real API calls.
- **Efficiency (10%):** Fewer turns per correct answer = higher score.
- **Novelty (5%):** Making different routing choices from other miners.

The composite score is checked against the current best in
[leaderboard.json](leaderboard.json).

## Current leaders

See [leaderboard.json](leaderboard.json) or the [dashboard](https://tinyrouter.ai).

## Questions

Open a [Discussion](https://github.com/<org>/tinyrouter/discussions) or check
[CONTRIBUTING.md](CONTRIBUTING.md) for setup help.
