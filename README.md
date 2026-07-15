<img src="Gittensor-TinyRouter.png" alt="TinyRouter" width="100%">

# TinyRouter

**The incentivized open benchmark for LLM routing intelligence.**

Train a tiny routing head (13,312 params) that decides — for every query —
which model to call and what role it plays. Beat the current king across three
benchmarks and earn TAO via [Gittensor](https://github.com/entrius/gittensor) SN74.

[![Gittensor](https://img.shields.io/badge/Gittensor-SN74-orange)](https://github.com/entrius/gittensor)
[![CI](https://github.com/James-CUDA/Gittensor-TinyRouter/actions/workflows/ci.yml/badge.svg)](https://github.com/James-CUDA/Gittensor-TinyRouter/actions)

## Quick start

```bash
git clone https://github.com/James-CUDA/Gittensor-TinyRouter.git
cd Gittensor-TinyRouter && pip install -e ".[dev]"
export OPENROUTER_API_KEY=sk-or-v1-...

# Train one head across all three benchmarks
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmarks math500,mmlu,livecodebench --run-name my-head

# Pack and submit
python scripts/pack_submission.py \
    --run-dir experiments/composite/my-head --miner-name your-name --benchmark composite
```

→ Full walkthrough: [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md)

## How it works

```
Query → Qwen3-0.6B encoder (frozen) → 13K-param head → (model, role) decision
```

The head picks one of three models and one of three roles (Thinker / Worker /
Verifier) per turn. It's trained derivative-free via separable CMA-ES against a
binary correct/wrong reward. The head never solves the task — it only learns
**who to ask**.

## Benchmarks

| Benchmark | Tests |
|---|---|
| **math500** | Math reasoning |
| **mmlu** | Domain knowledge |
| **livecodebench** | Code generation |

One head routes across all three. The composite score = mean of per-benchmark
scores. A new king must beat the previous king by **≥ 0.02** (2 percentage points).

## Model pool (same for all miners)

| Model | Provider |
|---|---|
| `qwen3.5` | OpenRouter |
| `gemini-flash-lite` | OpenRouter |
| `deepseek-v4-flash` | OpenRouter |

## Documentation

| Doc | What |
|---|---|
| [`SUBMITTING.md`](SUBMITTING.md) | How to submit a routing head |
| [`docs/COMPETITION_RULES.md`](docs/COMPETITION_RULES.md) | Rules, frozen files, cheating criteria |
| [`docs/EVALUATION_PIPELINE.md`](docs/EVALUATION_PIPELINE.md) | How scoring works (every stage) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Repo structure + design principles |
| [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md) | End-to-step walkthrough |
| [`leaderboard.json`](leaderboard.json) | Current king + history |

## Baselines (what to beat)

```bash
python baselines/always_model.py --model qwen3.5 --benchmark math500
python baselines/random_router.py --benchmark math500 --seeds 100
```

## Repository

```
src/trinity/coordinator/    routing engine (encoder, head, SVF, policy)
src/trinity/adapters/       benchmark adapters (9 benchmarks)
src/trinity/submission/     competition gates + leaderboard
src/trinity/orchestration/  session loop + shared grader
src/trinity/optim/          CMA-ES trainer + fitness
src/trinity/llm/            OpenRouter client + cost ledger
baselines/                  reference baselines
scripts/                    pr_eval, build_benchmark, pack_submission, ...
docs/                       competition + architecture docs
tests/                      200+ offline tests
```

## Research basis

Built on TRINITY (Xu et al., ICLR 2026, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)).
Research results: [`docs/RESULTS.md`](docs/RESULTS.md) · Lab notebook: [`docs/JOURNAL.md`](docs/JOURNAL.md)

## License

MIT
