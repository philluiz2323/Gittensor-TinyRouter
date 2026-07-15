# Roadmap

TinyRouter is the incentivized open benchmark for LLM routing intelligence.
Miners compete to train the best routing head across three benchmarks.

## Current competition

| Benchmark | Status |
|---|---|
| **math500** | ✅ Active |
| **mmlu** | ✅ Active |
| **livecodebench** | ✅ Active |

| Model | Provider |
|---|---|
| `qwen3.5` | OpenRouter |
| `gemini-flash-lite` | OpenRouter |
| `deepseek-v4-flash` | OpenRouter |

**Win condition:** composite score beats current king by ≥ 0.02 (2 percentage points).

## Done

- Routing engine (Qwen3-0.6B encoder + 13K head + SVF)
- Separable CMA-ES trainer with mixed-benchmark minibatches
- 9 benchmark adapters (math500, mmlu, livecodebench, gpqa, bbh, drop, swebench, mmlu-pro, livecodebench-v6)
- Encrypted hidden benchmarks (AES-256-GCM, sealed seed)
- 8 anti-cheat gates
- Composite scoring with win margin
- King-of-the-hill leaderboard
- Oracle ceiling diagnostic
- Competition documentation (RULES, PIPELINE, ARCHITECTURE, REPRODUCTION_GUIDE)
- Baselines (always-model, random-router)
- PR report generator
- 200+ offline tests

## Next

- **Expand benchmark suite** — add harder benchmarks (AIME, GPQA-Diamond, BigCodeBench) where model variance creates more routing headroom
- **Cost-aware scoring** — reward heads that achieve the same accuracy at lower API cost
- **Multi-objective optimization** — quality + cost + latency in the composite score
- **Live public routing API** — serve the current best head as an OpenAI-compatible endpoint
- **Community-voted pool** — let miners vote on which models to add/remove

## Maintainer priorities

1. Keep the hidden benchmark secure and un-leaked
2. Run `pr_eval.py` on every submission PR within 48 hours
3. Keep the pool models current (update when OpenRouter adds/removes models)
4. Expand benchmarks to increase routing headroom (harder = more room to improve)
5. Improve anti-cheat defenses as miners get more sophisticated
