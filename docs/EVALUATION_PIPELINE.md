# Evaluation Pipeline

> How a miner's routing head is evaluated, scored, and ranked.
> Every stage is documented here so miners can trust the process.

## Overview

```
  Submission (head_weights.npy + svf_scales.npy + receipt.json)
      │
      ▼
  ┌──────────────────────────┐
  │  1. PRE-EVAL GATES (7)   │  offline, zero GPU/API cost
  │  rate limit, weights,    │
  │  duplicate, receipt,     │
  │  ledger, schema, theta   │
  └──────────┬───────────────┘
             │ all gates pass
             ▼
  ┌──────────────────────────┐
  │  2. ENCODER LOAD         │  Qwen3-0.6B on GPU (once)
  │  configure full theta    │  head + SVF → policy
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  3. PER-BENCHMARK EVAL   │  × 3 benchmarks
  │  (math500, mmlu, LCB)    │
  │                          │
  │  For each benchmark:     │
  │   a. Cached eval (150q)  │  pre-stored model answers
  │   b. Audit eval (50q)    │  overfit detection
  │   c. Live eval (20q)     │  real API, multi-turn
  │   d. Novelty (50q)       │  vs current king
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  4. COMPOSITE SCORE      │  mean of 3 per-benchmark scores
  │  + MARGIN CHECK          │  must beat king by ≥ 0.02
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  5. LEADERBOARD UPDATE   │  king-of-the-hill
  └──────────────────────────┘
```

---

## Stage 1: Pre-Evaluation Gates (offline)

Before any GPU or API work, 7 gates run on the submission. A failing gate
rejects immediately at zero cost.

| Gate | What it checks | Source |
|---|---|---|
| **1. Rate limit** | ≤ 1 submission per miner per 7 days | `submission/gates.py:check_rate_limit` |
| **2. Weight validation** | Head shape `(6, 1024)`, SVF shape `(7168,)`, no NaN/Inf, norm > 0.001 | `submission/gates.py:validate_weights` |
| **3. Duplicate detection** | Cosine similarity < 0.999 vs all prior submissions + current king | `submission/gates.py:check_duplicate` |
| **4. Receipt plausibility** | Cost ≥ $15, ≥ 3 fitness entries, gen-0 fitness ≤ 0.98, non-flat, non-monotonic, `best_fitness` matches history peaks | `submission/gates.py:validate_receipt` |
| **5. Ledger consistency** | Receipt cost matches verified hash-chain ledger (±$0.05) | `submission/gates.py:validate_ledger_receipt_cost` |
| **6. Schema validation** | Receipt benchmark field matches expected, pool models correct | `submission/schema.py:validate_pack_schema` |
| **7. Theta integrity** | Head + SVF pack/unpack round-trip matches submitted shapes | `submission/schema.py:validate_theta_integrity` |

**Gate 5 (overfit rejection)** runs AFTER evaluation, not before: the eval−audit
accuracy gap must be ≤ 0.10 (hard reject) or ≤ 0.05 (0.85× penalty).

---

## Stage 2: Encoder Load

The frozen **Qwen3-0.6B** encoder is loaded once on GPU. The submitted
theta vector (13,312 floats = 6,144 head weights + 7,168 SVF scales) is
installed via `policy.configure(theta, spec)`, which:

1. Unpacks theta into head weights W `(6, 1024)` and SVF scales `(7168,)`.
2. Loads W into the `LinearHead` module.
3. Applies SVF scales to the 7 linear weight matrices in layer 26 of the
   encoder (Singular Value Fine-tuning — multiplies the singular values of
   each matrix by the learned scales).

The resulting policy is the **full submitted coordinator** — not just the
head, but the SVF-adapted encoder too. All subsequent evaluation uses this
configured policy via `policy.decide(transcript_text)`.

---

## Stage 3: Per-Benchmark Evaluation

The head is evaluated on **three benchmarks**:

| Benchmark | Task type | Grader | What it tests |
|---|---|---|---|
| **math500** | Math | Boxed-answer + symbolic equality | Math reasoning |
| **mmlu** | Multiple-choice (A–D) | Letter extraction | Domain knowledge |
| **livecodebench** | Code | pass@1 (sandboxed subprocess) | Code generation |

### 3a. Cached evaluation (70% weight per benchmark)

- **150 hidden questions** per benchmark, encrypted (AES-256-GCM).
- Each question has **pre-stored model answers** for every pool model
  (collected once via `build_benchmark.py` at `temperature=0.0`).
- The router's `(model, role)` decision selects which pre-stored answer to
  grade — **zero API cost, fully deterministic**.
- Grading uses the shared `reward.score_text(benchmark, answer, reference)`.
- This is the primary accuracy signal (70% of the per-benchmark score).

### 3b. Audit evaluation (overfit gate)

- **50 hidden audit questions** per benchmark (different from eval, same format).
- Same cached-lookup scoring as 3a.
- The **eval−audit gap** detects overfitting:
  - Gap > 0.10 → **hard reject** (submission rejected).
  - Gap > 0.05 → **0.85× penalty** on the per-benchmark score.
  - Gap ≤ 0.05 → clean (no penalty).

### 3c. Live evaluation (15% weight per benchmark)

- **20 live questions** per benchmark (no pre-stored answers).
- The full **multi-turn Thinker → Worker → Verifier loop** runs with real
  API calls (OpenRouter).
- Up to 5 turns; terminates early on `VERIFIER ACCEPT` (guarded: only after
  a Worker output exists).
- Graded via `reward.score(trajectory)` — uses the committed answer (most
  recent turn with an extractable answer), not just `final_answer`.

### 3d. Novelty (5% weight, first benchmark only)

- Compares the submitter's turn-1 routing decisions on 50 questions against
  the **current king's** decisions using the same configured policy.
- Novelty = fraction of decisions that differ from the king.
- If no king exists, novelty = 0.5 (neutral).

---

## Stage 4: Composite Score + Margin Check

### Per-benchmark score

For each benchmark, the score is:

```
bench_score = (0.70 × cached_acc + 0.15 × live_acc
              + 0.10 × efficiency + 0.05 × novelty) × overfit_penalty
```

Where:
- `cached_acc` = accuracy on the 150 cached eval questions
- `live_acc` = accuracy on the 20 live multi-turn questions
- `efficiency` = `((5 − avg_turns) / 4) × live_acc` (rewards fewer turns,
  gated on live accuracy being positive)
- `novelty` = disagreement with the current king's routing (first benchmark
  only; 0.0 for the other two)
- `overfit_penalty` = 1.0 (clean), 0.85 (eval−audit gap > 0.05), or reject
  (gap > 0.10)

### Composite score

```
composite = mean(bench_score_math500, bench_score_mmlu, bench_score_livecodebench)
```

### Win margin

```
if composite >= king_composite + 0.02:
    APPROVED (new king)
else:
    REJECTED
```

The **0.02 (2 percentage point) margin** prevents flip-flopping on eval
noise. At n=100–120 questions per benchmark with 3 reps, the noise band is
~±0.01–0.02, so a 2-point margin ensures the improvement is real.

---

## Stage 5: Leaderboard Update

On approval:
- `leaderboard.json` → `competition.best_composite_score` updated
- `competition.best_miner` = miner name
- `competition.best_per_benchmark` = per-benchmark breakdown
- `competition.history` appends the new king entry
- The PR is merged → Gittensor validators see it → TAO flows to the miner

On rejection:
- Only the composite score + delta are revealed (no per-component breakdown)
- The PR is closed
- The weekly submission slot is still consumed (prevents probing)

---

## Data Splits

Each benchmark has four data roles:

| Split | Size | Purpose | Visible to miners? |
|---|---|---|---|
| **Public train** | Full benchmark | Miners train their heads here | ✅ yes (via `load_tasks`) |
| **Hidden eval** | 150 questions | Cached scoring (70% weight) | ❌ encrypted, never revealed |
| **Hidden audit** | 50 questions | Overfit detection | ❌ encrypted, never revealed |
| **Hidden live** | 20 questions | Live multi-turn eval (15% weight) | ❌ encrypted, never revealed |

The hidden splits are built once via `scripts/build_benchmark.py` with a
sealed seed (`271828182`) and encrypted with AES-256-GCM (key derived from
`BENCHMARK_PASSWORD` via PBKDF2-HMAC-SHA256, 200k iterations). The password
never enters the repo.

---

## Model Pool

All miners route to the **same pool** — routing skill is what's measured:

| Slot | Model | Provider |
|---|---|---|
| A | `qwen3.5` | OpenRouter |
| B | `gemini-flash-lite` | OpenRouter |
| C | `deepseek-v4-flash` | OpenRouter |

All calls use `temperature=0.0` (greedy) for deterministic scoring, with
`reasoning_effort="low"` (mapped from the paper's "minimal reasoning").

---

## Reproducibility Guarantees

| Property | How it's ensured |
|---|---|
| **Deterministic cached eval** | Pre-stored answers at temp=0; same theta → same routing → same score |
| **Deterministic live eval** | temp=0 greedy decoding; `torch.manual_seed(seed)` for policy sampling |
| **Sealed benchmark** | Sealed seed + AES encryption; questions never revealed |
| **Auditable scoring** | `reward.score_text` is the single shared grader across all paths |
| **Receipt trail** | Hash-chain cost ledger + fitness history + theta integrity check |
| **Margin gate** | 2-point margin prevents noise-driven king flips |

---

## Files involved

| File | Role |
|---|---|
| `scripts/pr_eval.py` | Main evaluation orchestrator (all gates + scoring + leaderboard) |
| `scripts/build_benchmark.py` | One-time hidden benchmark builder (sealed seed, AES encryption) |
| `scripts/verify_benchmark.py` | Integrity verifier (decrypts + checks hash + counts) |
| `src/trinity/submission/gates.py` | The 7 pre-eval gates |
| `src/trinity/submission/schema.py` | Schema + theta integrity validation |
| `src/trinity/submission/constants.py` | Frozen constants (params, thresholds, pool, margin) |
| `src/trinity/orchestration/reward.py` | The shared grader (math/choice/code) |
| `src/trinity/adapters/` | Per-benchmark adapters (load, prompt, score) |
| `leaderboard.json` | King-of-the-hill state |
