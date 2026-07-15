# Submitting a Routing Head

TinyRouter is a **routing accuracy competition**. Miners train ONE routing head
across three benchmarks (math500, MMLU, LiveCodeBench), submit it as a PR, and
earn TAO through [Gittensor](https://github.com/entrius/gittensor) when the PR
is merged.

**A PR is only merged if the submitted head's composite score beats the current
king's composite by a margin of ≥ 0.02 (2 percentage points).**

## Overview

```
  ┌───────────┐     ┌──────────────┐     ┌──────────┐     ┌────────┐
  │ 1. TRAIN  │ ──► │ 2. PACK      │ ──► │ 3. SUBMIT│ ──► │ 4. EARN│
  │ One head  │     │ extract      │     │ open PR  │     │  TAO   │
  │ 3 bench-  │     │ head + SVF   │     │ to repo  │     │ (if win│
  │ marks     │     │ + receipt    │     │          │     │  margin)│
  └───────────┘     └──────────────┘     └──────────┘     └────────┘
```

## The three benchmarks

| Benchmark | Task type | What it tests |
|---|---|---|
| **math500** | Math (boxed-answer) | Math reasoning |
| **mmlu** | Multiple-choice (A–D) | Domain knowledge |
| **livecodebench** | Code (pass@1) | Code generation |

**One head routes across all three.** The head must generalize — a per-task
lookup table is not routing skill.

## Step 1: Train

Train ONE routing head across all three benchmarks using the mixed-benchmark
CMA-ES pipeline:

```bash
git clone https://github.com/James-CUDA/Gittensor-TinyRouter.git
cd Gittensor-TinyRouter
pip install -e ".[dev]"
source ~/.config/trinity/secrets.env   # exports OPENROUTER_API_KEY

# Train one head across all three benchmarks
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmarks math500,mmlu,livecodebench \
    --run-name my-submission \
    --generations 60
```

The mixed-benchmark minibatch draws tasks from all three benchmarks each
generation, so the head learns general routing (not a per-task lookup).

**You need:**
- A GPU (T4 or better) or CPU (slower)
- An OpenRouter API key (`OPENROUTER_API_KEY` env var)
- ~$25–65 in API credits for a full training run

**Output artifacts** (`experiments/mixed/my-submission/`):
- `best_theta.npy` — the trained parameter vector (13,312 floats)
- `history.json` — per-generation fitness values
- `summary.json` — run metadata

## Step 2: Pack

Extract the head weights and SVF scales from your trained theta, and build a
training receipt:

```bash
python scripts/pack_submission.py \
    --run-dir experiments/mixed/my-submission \
    --miner-name your-name \
    --benchmark composite
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

Run the same offline anti-cheat gates as `pr_eval` **before** opening a PR:

```bash
export TRINITY_COST_LEDGER=~/trinity/cost_ledger.jsonl   # optional, gate 5
python scripts/preflight_submission.py \
    --submission your-name/1 \
    --benchmark composite
```

**Advisories (`[WARN]`, never blocking).** Preflight and `pr_eval` also print report-only
signals. They **cannot fail your submission** — they only tell you something worth a look:

| Advisory | Warns when | Why it never blocks |
|---|---|---|
| `ledger_call_volume` | the ledger holds far fewer rows than `generations × popsize`, or only one model | ledger entries carry no run identifier, so a shared ledger's unrelated traffic can satisfy any threshold — it needs run-scoped provenance before it could reject |
| `head_routing_diversity` | the agent logit rows are effectively equal, so the head routes to one model for every query | the contract is score-based: a collapsed router is strategically weak but **valid** if it scores better |
| `fitness_history_sequence` | your `receipt.json` curve is internally inconsistent — a `generation` logged twice/out of order/with gaps, a row whose `mean_fitness` exceeds its own `max_fitness`, or a `best_fitness` that decreases | the gate chain is deliberately launch-friendly; this flags a likely-fabricated curve without blocking you |

## Step 3: Submit

1. **Fork** this repo and create a branch: `git checkout -b submission/your-name-gen1`
2. **Add** the submission directory: `submissions/your-name/1/`
3. **Commit** and push
4. **Open a PR** with title: `[submission] your-name gen 1 — composite`
5. The PR body must include:
   - Benchmarks trained on
   - Training method (CMA-ES or other)
   - Approximate API cost
   - Any notable techniques used

The maintainer runs `scripts/pr_eval.py` on your PR, which evaluates your head
against **hidden benchmarks** for all three tasks (never revealed). You will
receive a PR comment with your composite score.

## Step 4: Earn

- **If your composite beats the current king by ≥ 0.02:** the PR is merged,
  your name goes on the [leaderboard](leaderboard.json), and Gittensor
  validators see the merged PR → you earn TAO.
- **If your composite does not beat the king by the margin:** the PR is closed
  with your composite score and the current king's score shown.

## The margin rule

A new king must beat the previous king's composite score by **≥ 0.02 (2
percentage points)**. This prevents flip-flopping on eval noise — at n=100–120
questions per benchmark with 3 reps, the noise band is ~±0.01–0.02, so a
2-point margin ensures the improvement is real, not statistical luck.

## How scoring works

Your head is evaluated on **each** of the three benchmarks:

- **Cached single-turn accuracy (70% weight per benchmark):** hidden questions
  with pre-computed model answers. Zero API cost, deterministic.
- **Live multi-turn accuracy (15% weight per benchmark):** questions run through
  the full Thinker→Worker→Verifier loop with real API calls.
- **Efficiency (10%):** Fewer turns per correct answer = higher score.
- **Novelty (5%):** Making different routing choices from the current king.

The **composite score** = mean of the three per-benchmark scores. The composite
is checked against the current king's composite + the 0.02 margin.

## Rules

| Rule | Detail |
|---|---|
| **Submission rate** | 1 submission per day (the head covers all 3 benchmarks) |
| **Original work** | You must train the head yourself. Every submission is cosine-similarity checked against all previous submissions and rejected if copied |
| **Hidden benchmarks** | The evaluation questions are NEVER revealed. Do not ask for them |
| **Score feedback** | Winners get full per-benchmark results. Losers get ONLY the composite score + delta — no per-component breakdown (prevents probing) |
| **Receipt required** | Your training receipt must show cost ≥ $15 and a plausible fitness curve. Fabricated receipts are rejected |
| **One head** | A single head routes across all three benchmarks — no per-benchmark heads |
| **General PRs** | Docs, bug fixes, and refactors are welcome but do NOT earn TAO — only composite-improving routing heads count |

## Current leaders

See [leaderboard.json](leaderboard.json) or the
[dashboard](https://james-cuda.github.io/Gittensor-TinyRouter/).

## Questions

Open a [Discussion](https://github.com/James-CUDA/Gittensor-TinyRouter/discussions)
or check [CONTRIBUTING.md](CONTRIBUTING.md) for setup help.
