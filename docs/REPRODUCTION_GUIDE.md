# Reproduction Guide

> Clone to submission in 30 minutes. Every step is concrete and copy-pasteable.
> No guessing, no "figure it out."

## Prerequisites

| Requirement | What |
|---|---|
| **Python** | 3.10+ |
| **GPU** | T4 or better (for training); CPU works but is slower |
| **OpenRouter API key** | `OPENROUTER_API_KEY` env var (~$25–65 for a full run) |
| **Git** | For cloning and submitting |

---

## Step 0: Install

```bash
git clone https://github.com/James-CUDA/Gittensor-TinyRouter.git
cd Gittensor-TinyRouter
pip install -e ".[dev]"
```

Set up your API key:

```bash
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

Verify the pool is reachable:

```bash
python -m trinity.llm.openrouter_client --selftest
```

You should see `OK` from all three models: `qwen3.5`, `gemini-flash-lite`,
`deepseek-v4-flash`.

---

## Step 1: Run the baselines (see what you're beating)

```bash
# Always pick model A (qwen3.5) on every question
python baselines/always_model.py --model qwen3.5 --benchmark math500

# Always pick model B (gemini-flash-lite)
python baselines/always_model.py --model gemini-flash-lite --benchmark math500

# Random routing (the floor — your head must beat this)
python baselines/random_router.py --benchmark math500 --seeds 100

# Perfect-router oracle ceiling (the ceiling — how much headroom exists)
python scripts/oracle_ceiling.py --analyze experiments/final/oracle_matrix_math500.json
```

The oracle ceiling tells you **how much routing headroom exists** on each
benchmark. If the ceiling is close to best-single, there's little to route
— focus on a benchmark with more spread.

---

## Step 2: Train a routing head

Train ONE head across all three benchmarks using the mixed-benchmark pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmarks math500,mmlu,livecodebench \
    --run-name my-first-head \
    --generations 60 \
    --seed 0
```

**What happens:**
1. The frozen Qwen3-0.6B encoder loads on GPU.
2. Each CMA-ES generation samples 33 candidate θ vectors.
3. Each candidate is evaluated on a minibatch of 16 tasks drawn from ALL
   THREE benchmarks (mixed).
4. Fitness = mean binary reward (correct/wrong) across the minibatch.
5. After 60 generations, the best θ is saved.

**Output artifacts** (`experiments/composite/my-first-head/`):

| File | What |
|---|---|
| `best_theta.npy` | The trained parameter vector (13,312 floats) |
| `history.json` | Per-generation fitness values (for the receipt) |
| `summary.json` | Run metadata (seed, cost, generations, pool) |

**Typical cost:** ~$25–65 in OpenRouter API calls (tracked exactly via the
hash-chain ledger). Set `TRINITY_COST_LEDGER` to record it:

```bash
export TRINITY_COST_LEDGER=cost_ledger.jsonl
```

---

## Step 3: Evaluate locally (before submitting)

Run the same evaluation pipeline the maintainer uses, minus the hidden
benchmark (you don't have it). Use the public test split:

```bash
python -m trinity.eval \
    --benchmark math500 \
    --theta experiments/composite/my-first-head/best_theta.npy \
    --max-items 120 \
    --single-reps 3 \
    --rand-seeds 100 \
    --out experiments/composite/my-first-head/eval_math500.json
```

Repeat for each benchmark. Check the invariants:

```
R1/R2: TRINITY > best single model?     ← the headline claim
R4:    TRINITY > random routing?        ← the floor
```

If TRINITY doesn't beat random routing, the head hasn't learned anything
useful — retrain with a different seed or more generations.

---

## Step 4: Pack the submission

```bash
python scripts/pack_submission.py \
    --run-dir experiments/composite/my-first-head \
    --miner-name your-name \
    --benchmark composite
```

This creates `submissions/your-name/1/` with:
- `head_weights.npy` — (6, 1024) float32
- `svf_scales.npy` — (7168,) float32
- `receipt.json` — training metadata (cost, seed, gens, fitness curve)
- `README.md` — auto-generated summary

---

## Step 5: Preflight (recommended)

Run the same offline anti-cheat gates the maintainer uses, BEFORE opening
a PR:

```bash
export TRINITY_COST_LEDGER=cost_ledger.jsonl
python scripts/preflight_submission.py \
    --submission your-name/1 \
    --benchmark composite
```

If any gate fails, fix the issue before submitting. Common failures:
- **Receipt cost < $15** → train longer or verify the ledger path.
- **Flat fitness curve** → the optimizer didn't learn; check the pool and seed.
- **Duplicate detection** → you accidentally packed a copy of someone's head.

---

## Step 6: Submit

1. **Fork** the repo on GitHub.
2. **Create a branch:** `git checkout -b submission/your-name-gen1`
3. **Copy** your submission directory: `submissions/your-name/1/`
4. **Commit** and push to your fork.
5. **Open a PR** with title: `[submission] your-name gen 1 — composite`
6. **PR body** must include:
   - Benchmarks trained on
   - Training method (CMA-ES, warm-start, etc.)
   - Approximate API cost
   - Any notable techniques

---

## Step 7: Wait for evaluation

The maintainer runs `scripts/pr_eval.py` on your PR within 48 hours. You
will receive a PR comment with your result:

**If you win** (composite beats king by ≥ 0.02):
```
APPROVED: composite 0.823 beats king 0.801 by +0.022 (margin 0.020)

  math500         = 0.815
  mmlu            = 0.901
  livecodebench   = 0.753
```

**If you lose:**
```
REJECTED: composite 0.789 does not beat king 0.801 by margin 0.020
(delta: -0.012)
```

The PR is merged on a win (TAO flows via Gittensor) or closed on a loss.
Either way, your weekly submission slot is consumed.

---

## Step 8: Iterate

Wait 7 days, then resubmit with an improved head:

```bash
# Train with more generations, different seed, or warm-start
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmarks math500,mmlu,livecodebench \
    --run-name my-second-head \
    --generations 80 \
    --seed 42 \
    --warmstart-theta experiments/composite/my-first-head/best_theta.npy

# Pack as generation 2
python scripts/pack_submission.py \
    --run-dir experiments/composite/my-second-head \
    --miner-name your-name \
    --benchmark composite
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Refusing to train on the offline toy set` | Install `datasets` (`pip install datasets`) and verify network access to HuggingFace |
| `best_theta not found` | Check the `--run-dir` path; the file is `best_theta.npy` |
| Receipt cost too low | Set `TRINITY_COST_LEDGER` before training; ensure ≥ $15 spend |
| All models score the same | The pool may lack complementarity; check with `oracle_ceiling.py` |
| TRINITY = random routing | The head hasn't learned; try more generations, a different seed, or warm-start |
| `BENCHMARK_PASSWORD not set` | Only the maintainer needs this (for `pr_eval.py`); miners don't |
