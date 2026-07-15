# Competition Rules

> The complete, authoritative rules for the TinyRouter routing competition.
> If something isn't answered here, open a
> [Discussion](https://github.com/James-CUDA/Gittensor-TinyRouter/discussions).

## What the competition is

Miners train **one routing head** (13,312 parameters) that decides, for each
query, **which model** to call and **what role** it plays (Thinker, Worker, or
Verifier). The head sits on a frozen Qwen3-0.6B encoder and is trained
derivative-free via separable CMA-ES.

The head is evaluated across **three benchmarks** (math500, MMLU,
LiveCodeBench). The composite score must beat the current king by **≥ 0.02**
(2 percentage points) to win.

Winners earn **TAO** via [Gittensor](https://github.com/entrius/gittensor)
(Subnet 74) when their PR is merged.

---

## What you submit

```
submissions/your-name/1/
├── head_weights.npy    # (6, 1024) float32 — 6,144 params
├── svf_scales.npy      # (7168,) float32 — 7,168 params
├── receipt.json        # training metadata
└── README.md           # auto-generated summary
```

**Total: 13,312 trainable parameters.** No more, no less.

---

## What you are allowed to modify

| You CAN | You CANNOT |
|---|---|
| Train the head + SVF via CMA-ES (or any method) | Modify the frozen encoder (Qwen3-0.6B) |
| Choose your training data, seed, hyperparameters | Modify the hidden benchmark questions |
| Use warm-start, shaped fitness, any training trick | Modify the grader (`reward.py`) |
| Submit multiple generations (1 per week) | Submit per-benchmark heads (one head across all 3) |
| Inspect the public train split | Probe the hidden eval/audit/live splits |

---

## Frozen files (do not touch)

These files are **frozen** — modifying them in a submission PR is cheating:

| File | Why it's frozen |
|---|---|
| `scripts/pr_eval.py` | The evaluation orchestrator |
| `scripts/build_benchmark.py` | The hidden benchmark builder |
| `src/trinity/orchestration/reward.py` | The shared grader |
| `src/trinity/submission/*.py` | The anti-cheat gates |
| `src/trinity/submission/constants.py` | Frozen constants (pool, margin, params) |
| `leaderboard.json` | King-of-the-hill state (maintainer-only writes) |
| Any file under `$TINYROUTER_BENCHMARK_DIR/` | Encrypted hidden benchmarks |

General-improvement PRs (bug fixes, new adapters, docs) are welcome but
**do not earn TAO** — only composite-improving routing heads count.

---

## What counts as cheating

| Behavior | Consequence |
|---|---|
| Copying another miner's head (cosine ≥ 0.999) | Gate 3 rejects |
| Fabricating a training receipt | Gate 4 rejects (fitness curve must be plausible) |
| Forging cost in the ledger | Gate 5 rejects (hash-chain verification) |
| Overfitting to the hidden eval (eval−audit gap > 0.10) | Gate 5 hard-rejects |
| Submitting more than once per week | Gate 1 rejects |
| Modifying frozen files in a submission PR | Rejected by maintainer |
| Probing the hidden benchmark (repeated losing submissions to extract per-component scores) | Only composite + delta revealed on loss; rate-limited weekly |

---

## How a score is calculated

### Per-benchmark score

```
bench_score = (0.70 × cached_acc + 0.15 × live_acc
              + 0.10 × efficiency + 0.05 × novelty) × overfit_penalty
```

| Component | Weight | What it measures |
|---|---|---|
| **Cached accuracy** | 70% | Correctness on 150 hidden questions (pre-stored answers, deterministic, $0) |
| **Live accuracy** | 15% | Correctness on 20 questions via full multi-turn T→W→V loop (real API) |
| **Efficiency** | 10% | `((5 − avg_turns) / 4) × live_acc` — fewer turns is better, gated on live accuracy |
| **Novelty** | 5% | Disagreement with the current king's routing decisions |
| **Overfit penalty** | ×1.0 / ×0.85 / reject | eval−audit gap ≤ 0.05 / ≤ 0.10 / > 0.10 |

### Composite score

```
composite = mean(bench_score_math500, bench_score_mmlu, bench_score_livecodebench)
```

### Win condition

```
if composite >= king_composite + 0.02:
    APPROVED → PR merged, TAO earned
else:
    REJECTED → PR closed, slot consumed
```

---

## The 8 anti-cheat gates

| # | Gate | When | Cost |
|---|---|---|---|
| 1 | Rate limit (1/week) | Pre-eval | $0 |
| 2 | Weight validation (shape, NaN, norm) | Pre-eval | $0 |
| 3 | Duplicate detection (cosine < 0.999) | Pre-eval | $0 |
| 4 | Receipt plausibility (cost ≥ $15, fitness curve) | Pre-eval | $0 |
| 5 | Ledger/receipt cost consistency | Pre-eval | $0 |
| 6 | Schema/benchmark validation | Pre-eval | $0 |
| 7 | Theta pack/unpack integrity | Pre-eval | $0 |
| 8 | Overfit rejection (eval−audit gap) | Post-eval | GPU + API |

---

## Submission rate

- **1 submission per week** (7-day rolling window, enforced by Gate 1).
- The slot is consumed when Gate 1 passes — **even if later gates fail or the
  score loses**. This prevents probing via repeated losing submissions.
- Generation number auto-increments: `submissions/your-name/1/`, `2/`, `3/`...

---

## Score feedback

| Outcome | What you see |
|---|---|
| **Win** | Full per-benchmark breakdown (cached, live, efficiency, novelty per benchmark) + composite + delta |
| **Loss** | Composite score + delta only (no per-component breakdown — prevents benchmark probing) |
| **Gate fail** | Gate name + reason (no score at all) |

---

## Model pool (fixed for all miners)

| Slot | Model | Provider |
|---|---|---|
| A | `qwen3.5-35b-a3b` | OpenRouter |
| B | `gemini-3.1-flash-lite` | OpenRouter |
| C | `deepseek-v4-flash` | OpenRouter |

All miners route to the **same pool**. The competition measures routing
intelligence, not model quality.

---

## How to get started

1. Read [`SUBMITTING.md`](../SUBMITTING.md) for the step-by-step guide.
2. Read [`docs/EVALUATION_PIPELINE.md`](EVALUATION_PIPELINE.md) for how scoring works.
3. Read [`docs/REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) for a full end-to-end walkthrough.
4. Run the baselines in [`baselines/`](../baselines/) to see what you're beating.
5. Train, pack, submit.

---

## FAQ

**Can I use a different encoder?**
No. The encoder is frozen at Qwen3-0.6B for all miners. You train the head + SVF scales only.

**Can I submit separate heads for each benchmark?**
No. One head routes across all three benchmarks. This tests generalization, not memorization.

**Can I see the hidden questions?**
No. They are AES-256-GCM encrypted and never revealed. The sealed seed (`271828182`) and `BENCHMARK_PASSWORD` are maintainer-only.

**What if two miners tie?**
The margin rule (≥ 0.02) prevents ties on noise. If the composite exactly equals king + 0.02, the first to submit wins (king-of-the-hill).

**Can I retrain and resubmit?**
Yes, after 7 days. Each submission consumes one weekly slot regardless of outcome.

**How is TAO calculated?**
TAO distribution is governed by the [Gittensor subnet weights](https://github.com/entrius/gittensor), not by this repo. This repo determines the king; the subnet determines emissions.
