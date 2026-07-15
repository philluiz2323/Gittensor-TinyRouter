<div align="center">
  <img src="Gittensor-TinyRouter.png" alt="TinyRouter" width="100%">
  
  # 🐡 TinyRouter
  
  ### The incentivized open benchmark for LLM routing intelligence.
  
  Train a **13,312-parameter** routing head. Beat the king across **3 benchmarks**. Earn **TAO**.

  [![Gittensor](https://img.shields.io/badge/Gittensor-SN74-orange?style=for-the-badge)](https://github.com/entrius/gittensor)
  [![CI](https://github.com/James-CUDA/Gittensor-TinyRouter/actions/workflows/ci.yml/badge.svg?style=for-the-badge)](https://github.com/James-CUDA/Gittensor-TinyRouter/actions)
  [![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
  
  [Submit a head](SUBMITTING.md) · [Rules](docs/COMPETITION_RULES.md) · [How scoring works](docs/EVALUATION_PIPELINE.md) · [Leaderboard](leaderboard.json)
</div>

---

## The challenge

Three models. Three benchmarks. One tiny head.

```
  Your query
      │
      ▼
  Qwen3-0.6B encoder (frozen)
      │
      ▼
  13K-param routing head
      │
      ├──▶ which model?  ──▶  qwen3.5-35b-a3b  |  gemini-3.1-flash-lite  |  deepseek-v4-flash
      │
      └──▶ which role?   ──▶  Thinker  |  Worker  |  Verifier
```

The head never solves the task. It only learns **who to ask**. Train it via separable CMA-ES against a binary correct/wrong reward. The best head wins.

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/James-CUDA/Gittensor-TinyRouter.git
cd Gittensor-TinyRouter
pip install -e ".[dev]"
export OPENROUTER_API_KEY=sk-or-v1-...

# 2. Train one head across all three benchmarks
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmarks math500,mmlu,livecodebench \
    --run-name my-head

# 3. Pack
python scripts/pack_submission.py \
    --run-dir experiments/composite/my-head \
    --miner-name your-name --benchmark composite

# 4. Submit — open a PR with submissions/your-name/1/
# 5. Earn TAO if you beat the king by ≥ 2 points 🏆
```

> **Full walkthrough:** [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md)

## Competition

| | |
|---|---|
| **Benchmarks** | `math500` (math) · `mmlu` (knowledge) · `livecodebench` (code) |
| **Model pool** | `qwen3.5-35b-a3b` · `gemini-3.1-flash-lite` · `deepseek-v4-flash` |
| **Same pool** | All miners route to the same three models. Routing skill is what matters. |
| **One head** | A single head routes across all three benchmarks. No per-benchmark heads. |
| **Win margin** | Composite must beat the current king by **≥ 0.02** (2 percentage points). |
| **Rate limit** | 1 submission per week. |
| **Scoring** | 70% cached accuracy · 15% live multi-turn · 10% efficiency · 5% novelty |

<details>
<summary><b>How scoring works</b></summary>

Each benchmark is scored independently, then averaged into a composite:

```
bench_score = (0.70 × cached_acc + 0.15 × live_acc
              + 0.10 × efficiency + 0.05 × novelty) × overfit_penalty

composite = mean(bench_score_math500, bench_score_mmlu, bench_score_livecodebench)
```

- **Cached accuracy (70%):** 150 hidden questions with pre-stored model answers. Zero API cost, fully deterministic.
- **Live accuracy (15%):** 20 questions through the full Thinker→Worker→Verifier loop with real API calls.
- **Efficiency (10%):** Fewer turns per correct answer = higher score.
- **Novelty (5%):** Making different routing decisions from the current king.
- **Overfit gate:** Eval−audit accuracy gap > 10% = hard reject. > 5% = 0.85× penalty.

Full details: [`docs/EVALUATION_PIPELINE.md`](docs/EVALUATION_PIPELINE.md)
</details>

<details>
<summary><b>Anti-cheat gates (8)</b></summary>

| # | Gate | Catches |
|---|---|---|
| 1 | Rate limit | Submission flooding (1/week) |
| 2 | Weight validation | NaN/Inf/degenerate heads |
| 3 | Duplicate detection | Copied heads (cosine ≥ 0.999) |
| 4 | Receipt plausibility | Fabricated training (cost < $15, flat fitness) |
| 5 | Ledger verification | Forged cost (hash-chain check) |
| 6 | Schema validation | Malformed receipt |
| 7 | Theta integrity | Pack/unpack mismatch |
| 8 | Overfit rejection | Eval−audit gap > 10% |
</details>

## Baselines — what you're beating

```bash
# Single-model floor (best of these = "best-single", the simplest strategy to beat)
python baselines/always_model.py --model qwen3.5-35b-a3b --benchmark math500
python baselines/always_model.py --model gemini-3.1-flash-lite --benchmark math500
python baselines/always_model.py --model deepseek-v4-flash --benchmark math500

# Random routing floor (your head MUST beat this)
python baselines/random_router.py --benchmark math500 --seeds 100

# Perfect-router ceiling (how much routing headroom exists)
python scripts/oracle_ceiling.py --collect --benchmark math500
python scripts/oracle_ceiling.py --analyze experiments/final/oracle_matrix_math500.json
```

## Documentation

| 📖 | Document | For |
|---|---|---|
| 📋 | [`COMPETITION_RULES.md`](docs/COMPETITION_RULES.md) | What you can/can't do, frozen files, cheating criteria |
| ⚙️ | [`EVALUATION_PIPELINE.md`](docs/EVALUATION_PIPELINE.md) | How every score is calculated (every stage) |
| 🏗️ | [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Repo structure, design principles, subsystem map |
| 🚀 | [`REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md) | 8-step clone-to-submit walkthrough |
| 📝 | [`SUBMITTING.md`](SUBMITTING.md) | Submission format + PR workflow |
| 📖 | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Term definitions |

## Repository

```
src/trinity/
├── coordinator/          Routing engine (Qwen3-0.6B encoder + 13K head + SVF)
├── adapters/             Benchmark adapters (9 benchmarks behind one interface)
├── submission/           Competition gates + leaderboard + anti-cheat
├── orchestration/        Multi-turn session loop + shared grader
├── optim/                Separable CMA-ES trainer + fitness evaluation
├── llm/                  OpenRouter client + hash-chain cost ledger
├── analysis/             Oracle ceiling + convergence + selective prediction
└── fugu/                 Conductor orchestration (Tier 2 — future)

baselines/                Reference baselines (always-model, random-router)
scripts/                  pr_eval, build_benchmark, pack_submission, ...
configs/                  trinity.yaml (training) + models.yaml (pool)
tests/                    200+ offline tests
docs/                     Competition + architecture documentation
```

## Research

Built on [TRINITY](https://arxiv.org/abs/2512.04695) (Xu et al., ICLR 2026) —
a compact coordinator that delegates to a pool of LLMs via an evolved routing
head, without touching the models' weights.

- 📊 Honest results: [`docs/RESULTS.md`](docs/RESULTS.md)
- 🔬 Lab notebook: [`docs/JOURNAL.md`](docs/JOURNAL.md)
- 📐 Implementation spec: [`docs/SPEC.md`](docs/SPEC.md)

## License

MIT
