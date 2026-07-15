# Architecture

> How the repository is organized, and which subsystem does what.
> New contributors: read this first to know where you are.

## The three roles

TinyRouter serves three distinct audiences. Knowing which one you are tells
you which part of the repo to work in:

```
┌──────────────────────────────────────────────────────┐
│                TinyRouter Repository                  │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  FRAMEWORK   │  │ COMPETITION  │  │   RESEARCH  │ │
│  │              │  │              │  │             │ │
│  │ The routing  │  │ The hidden   │  │ The TRINITY │ │
│  │ engine,      │  │ benchmarks,  │  │ replication │ │
│  │ adapters,    │  │ anti-cheat   │  │ + analysis  │ │
│  │ training,    │  │ gates,       │  │ tools       │ │
│  │ eval         │  │ leaderboard  │  │             │ │
│  └─────────────┘  └──────────────┘  └─────────────┘ │
│                                                      │
│  "Am I building     "Am I submitting   "Am I         │
│   the router?"       to the            studying      │
│                       competition?"     routing?"    │
└──────────────────────────────────────────────────────┘
```

| You are... | Start here | What you do |
|---|---|---|
| **A miner** | [`SUBMITTING.md`](../SUBMITTING.md) → [`docs/REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md) | Train a head, pack it, submit a PR |
| **A framework contributor** | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (this file) | Improve the routing engine, add adapters, fix bugs |
| **A researcher** | [`docs/SPEC.md`](SPEC.md) → [`docs/RESULTS.md`](RESULTS.md) | Study routing, run diagnostics, analyze results |

---

## Repository map

```
Gittensor-TinyRouter/
│
├── src/trinity/                    THE FRAMEWORK
│   ├── coordinator/                Routing engine
│   │   ├── slm.py                  Frozen Qwen3-0.6B encoder
│   │   ├── head.py                 Linear routing head (6×1024)
│   │   ├── svf.py                  SVF adapter (layer 26)
│   │   ├── policy.py               CoordinatorPolicy (encoder + SVF + head)
│   │   ├── params.py               θ pack/unpack (head + SVF ↔ flat vector)
│   │   └── warmstart.py            Supervised warm-start for the head
│   │
│   ├── adapters/                   Benchmark adapters (the eval seam)
│   │   ├── base.py                 BenchmarkAdapter ABC + TaskType
│   │   ├── registry.py             Adapter registration + lookup
│   │   ├── builtin.py              Delegating adapters for built-in benchmarks
│   │   ├── loaders.py              HuggingFace dataset loaders
│   │   ├── scoring.py              Shared scoring dispatcher
│   │   ├── bbh.py                  BIG-Bench Hard adapter
│   │   ├── drop.py                 DROP adapter
│   │   ├── mmlu_pro.py             MMLU-Pro adapter
│   │   ├── swebench.py             SWE-bench Verified adapter
│   │   ├── swebench_runner.py      Sandboxed SWE-bench patch evaluator
│   │   └── conformance.py          Offline adapter contract auditor
│   │
│   ├── orchestration/              Session loop + grader
│   │   ├── session.py              Multi-turn T→W→V coordination loop
│   │   ├── reward.py               The shared grader (math/choice/code)
│   │   └── dataset.py              Task loading + minibatch sampling
│   │
│   ├── optim/                      Training
│   │   ├── sep_cmaes.py            Separable CMA-ES wrapper
│   │   ├── fitness.py              Candidate fitness evaluation
│   │   ├── baselines.py            RS/SFT/REINFORCE baselines (R8)
│   │   ├── budget.py               Atomic-eval budget tracking
│   │   └── sampling.py             Common-random-numbers task sampling
│   │
│   ├── llm/                        Model pool clients
│   │   ├── openrouter_client.py    Async OpenRouter chat client
│   │   ├── openrouter_pricing.py   Per-model pricing ($/1M tokens)
│   │   ├── cost_ledger.py          Hash-chain cost ledger
│   │   ├── cache.py                Response cache
│   │   └── pool_consistency.py     Pool membership/pricing verifier
│   │
│   ├── analysis/                   Diagnostics (research tools)
│   │   ├── oracle/                 Oracle ceiling diagnostic
│   │   ├── convergence.py          CMA-ES convergence analysis
│   │   ├── selective.py            Per-model selective prediction
│   │   ├── reconcile.py            Oracle-matrix ↔ eval reconciliation
│   │   ├── ensemble.py             Multi-model ensemble analysis
│   │   └── turns_monotonicity.py   SPEC R7 verification
│   │
│   ├── submission/                 COMPETITION INFRASTRUCTURE
│   │   ├── constants.py            Frozen constants (params, pool, margin)
│   │   ├── gates.py                The 7 pre-eval anti-cheat gates
│   │   ├── schema.py               Receipt + theta validation
│   │   ├── leaderboard.py          Leaderboard reader/verifier
│   │   ├── preflight.py            Offline preflight checker
│   │   └── pack.py                 Submission pack/unpack
│   │
│   ├── fugu/                       Conductor (Tier 2 — future)
│   │   ├── workflow.py             3-list workflow schema + executor
│   │   ├── grpo.py                 GRPO training loop
│   │   ├── conductor.py            Prompted/stub/trained conductor
│   │   ├── hf_backend.py           Trainable HF backend
│   │   ├── reward.py               Two-stage Conductor reward
│   │   └── eval.py                 Conductor evaluation
│   │
│   ├── train.py                    Training entrypoint (CMA-ES)
│   ├── eval.py                     Evaluation entrypoint
│   └── types.py                    Shared dataclasses (Task, Trajectory, etc.)
│
├── scripts/                        Maintainer + miner tools
│   ├── pr_eval.py                  PR evaluation (all gates + scoring)
│   ├── build_benchmark.py          Hidden benchmark builder (encrypted)
│   ├── pack_submission.py          Pack a trained head for submission
│   ├── preflight_submission.py     Offline preflight checker
│   ├── oracle_ceiling.py           Oracle ceiling diagnostic
│   ├── verify_benchmark.py         Hidden benchmark integrity verifier
│   └── ...                         Report generators, analysis tools
│
├── baselines/                      Reference baselines (what to beat)
│   ├── always_model.py             Always pick one model
│   ├── random_router.py            Random (agent, role) each turn
│   └── oracle_ceiling.py           Perfect-router upper bound
│
├── submissions/                    Miner submissions (PR'd)
├── experiments/                    Run outputs (gitignored)
├── configs/                        trinity.yaml, models.yaml
│
├── docs/                           Documentation
│   ├── SPEC.md                     Implementation spec (TRINITY replication)
│   ├── RESULTS.md                  Research results
│   ├── JOURNAL.md                  Lab notebook (mistakes + findings)
│   ├── EVALUATION_PIPELINE.md      How scoring works
│   ├── COMPETITION_RULES.md        Competition rules
│   ├── ARCHITECTURE.md             This file
│   └── REPRODUCTION_GUIDE.md       End-to-end walkthrough
│
├── tests/                          Test suite (200+ offline tests)
├── leaderboard.json                King-of-the-hill state
├── SUBMITTING.md                   Miner quick-start
└── CONTRIBUTING.md                 Contributor guide
```

---

## The routing engine (framework)

The core asset: a **13,312-parameter routing head** on a frozen **Qwen3-0.6B**
encoder, trained derivative-free via **separable CMA-ES**.

```
Query (transcript text)
    │
    ▼
Qwen3-0.6B encoder (frozen, layer 26 SVF-adapted)
    │
    ▼  penultimate-token hidden state (1024-dim, L2-normalized)
    │
LinearHead: z = W·h  (6×1024, no bias)
    │
    ├─ agent logits [0:3] → softmax → pick model
    └─ role logits [3:6]  → softmax → pick role (Thinker/Worker/Verifier)
    │
    ▼
(model, role) decision
```

**Trainable parameters:**
- Head weights W: 6 × 1024 = **6,144**
- SVF scales (layer 26, 7 matrices): 7 × 1024 = **7,168**
- **Total: 13,312**

The encoder itself is frozen; only the head and SVF scales are trained. SVF
(Singular Value Fine-tuning) adapts the encoder's layer-26 singular values
without changing the orthogonal factors — a light, learnable adaptation that
preserves the encoder's structure.

---

## The competition infrastructure

The competition is what makes TinyRouter unique. It is NOT just a benchmark
— it's an **incentivized** benchmark where verified improvements earn TAO.

**Key components:**
- **Encrypted hidden benchmarks** (AES-256-GCM, sealed seed, never revealed)
- **8 anti-cheat gates** (rate limit, duplicate, receipt, overfit, etc.)
- **Composite scoring with 0.02 win margin** (prevents noise-driven flips)
- **King-of-the-hill leaderboard** (one composite king across 3 benchmarks)
- **Per-benchmark audit split** (overfit detection)
- **Hash-chain cost ledger** (tamper-evident spend tracking)

See [`docs/COMPETITION_RULES.md`](COMPETITION_RULES.md) for the full rules
and [`docs/EVALUATION_PIPELINE.md`](EVALUATION_PIPELINE.md) for how scoring
works.

---

## The research layer

The repo originated as a TRINITY replication (Xu et al., ICLR 2026). The
research artifacts remain as credibility and analysis tooling:

- [`docs/SPEC.md`](SPEC.md) — the implementation spec (grounded in the paper)
- [`docs/RESULTS.md`](RESULTS.md) — honest results (thin multi-task win, math null)
- [`docs/JOURNAL.md`](JOURNAL.md) — lab notebook (every mistake, finding, decision)
- [`docs/ORACLE_CEILING_DIAGNOSTIC.md`](ORACLE_CEILING_DIAGNOSTIC.md) — the diagnostic methodology
- `src/trinity/analysis/` — 12 diagnostic modules (oracle, convergence, selective, etc.)

The research is **done** — it feeds the competition's credibility but does
not drive the roadmap. New research questions (which encoder? which head?
which loss?) are answered through the competition itself: miners try
different approaches, the leaderboard shows what works.

---

## Tier 1 vs Tier 2

| Tier | What | Status |
|---|---|---|
| **Tier 1 — Routing** | One head picks (model, role) per turn across 3 benchmarks | **Active competition** |
| **Tier 2 — Orchestration** | Conductor emits multi-step workflows (cascade, verify, self-correct) | Infrastructure built (`fugu/`), competition not yet active |

Tier 2 is the path to competing with Sakana Fugu's fugu-ultra tier. The
GRPO trainer exists but hit a dead-gradient problem on thin pools. It will
activate when the routing competition is stable and a complementary
high-variance pool is available.

---

## Design principles

1. **One head across all benchmarks** — tests generalization, not memorization.
2. **The grader is the source of truth** — `reward.py` is shared across eval, training, oracle, and Fugu paths; no path has its own grader.
3. **Adapters are the seam** — the evaluator never branches on benchmark name; everything goes through `get_adapter(name)`.
4. **Common random numbers** — all CMA-ES candidates in a generation share one minibatch (re-sampled per generation) so fitness differences reflect policy quality, not task luck.
5. **Honesty over performance** — the JOURNAL records every mistake; RESULTS reports null results plainly; the oracle diagnostic exists to catch false negatives.
