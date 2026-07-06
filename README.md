<img src="Gittensor-TinyRouter.png" alt="TinyRouter — Evolved LLM Coordinator" width="100%">

# TinyRouter

Evolved LLM coordinator — a **routing accuracy competition** on
[Gittensor](https://github.com/entrius/gittensor) (Bittensor Subnet 74).

[![Website](https://img.shields.io/badge/website-tinyrouter.ai-blue)](https://james-cuda.github.io/Gittensor-TinyRouter/)
[![Gittensor](https://img.shields.io/badge/Gittensor-SN74-orange)](https://github.com/entrius/gittensor)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Train a tiny coordinator head that decides which LLM to call for any question.
Beat the current best accuracy on the hidden benchmark, and your PR gets merged —
earning you TAO.

**Quick links:** [Website](https://james-cuda.github.io/Gittensor-TinyRouter/) ·
[Leaderboard](leaderboard.json) · [Submission Guide](SUBMITTING.md) ·
[Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md)

---

## Compete (earn TAO)

```
  1. TRAIN             2. PACK              3. SUBMIT           4. EARN
  CMA-ES on GPU        extract head         open a PR to        PR merged →
  ~$25–65/run          + SVF + receipt      this repo           TAO via Gittensor
```

### Quick start

```bash
git clone https://github.com/James-CUDA/Gittensor-TinyRouter.git
cd Gittensor-TinyRouter
pip install -e ".[dev]"
source ~/.config/trinity/secrets.env   # exports FIREWORKS_API_KEY

# 1. Train a routing head
CUDA_VISIBLE_DEVICES=0 python -m trinity.train \
    --benchmark math500 --run-name my-submission

# 2. Pack for submission
python scripts/pack_submission.py \
    --run-dir experiments/math500/my-submission \
    --miner-name your-name --benchmark math500

# 3. Open a PR with submissions/your-name/1/
# 4. Maintainer runs pr_eval.py — if you beat the current best, you win
```

### Rules

| Rule | Detail |
|---|---|
| **Submission rate** | 1 per benchmark per week (enforced in code) |
| **Original work** | Every submission is cosine-similarity checked against all previous heads. Copies rejected |
| **Hidden benchmark** | Encrypted, never revealed. 150 eval + 50 audit questions |
| **Score feedback** | Winners get full results. Losers get only composite score + delta |
| **Receipt** | Must show cost ≥ $15 and a plausible CMA-ES fitness curve |
| **Evaluation** | 70% cached single-turn + 15% live multi-turn + 10% efficiency + 5% novelty |

Full details: [SUBMITTING.md](SUBMITTING.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

### Current leaders

See the [live leaderboard](leaderboard.json) or the [website](https://james-cuda.github.io/Gittensor-TinyRouter/).

---

## What is TinyRouter?

A small **coordinator** that, for every question, decides two things: **which** of three
open-source LLMs should answer it, and **what role** that model should play (Thinker, Worker, or
Verifier). The coordinator is deliberately tiny and cheap. A frozen **0.6B** encoder reads the
question into a single vector, and a **~10K-parameter** head turns that vector into the routing
decision. It is trained by **separable CMA-ES**, a derivative-free evolution strategy, against a
simple right/wrong reward. The coordinator never solves the question itself; it only learns who to ask.

The method follows TRINITY (Xu et al., ICLR 2026, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)),
rebuilt from scratch with an all open-source model pool served through Fireworks AI.

## Model pool

| Slot | Model            | Strong at        |
| ---- | ---------------- | ---------------- |
| A    | `deepseek-v4-pro` | knowledge (MMLU) |
| B    | `glm-5p2`         | math             |
| C    | `kimi-k2p6`       | general          |

The 0.6B encoder and the evolution loop run on a single NVIDIA H200; the three LLMs are called over HTTP.
All miners route to the same pool — fair comparison, routing skill is what matters.

## How it works

1. The frozen 0.6B encoder turns the question into one 1024-dim vector.
2. The ~10K head reads that vector and picks a model and a role.
3. The chosen model answers in that role; its output is appended to the transcript.
4. Steps 1 to 3 repeat for up to 5 turns; a Verifier turn can accept and stop early.
5. The final answer is graded right/wrong, and that reward drives the evolutionary training.

## Research results

Rigorous eval: 120 held-out questions per task; single-model baselines are the mean over 3 runs.
Scores are fraction correct (0.792 = 79.2%).

**Math**

| system | score |
| --- | --- |
| glm-5p2 | 0.794 (best single) |
| **TinyRouter** | **0.792** |
| random routing | 0.792 |
| deepseek-v4-pro | 0.747 |
| kimi-k2p6 | 0.742 |

**MMLU**

| system | score |
| --- | --- |
| **TinyRouter** | **0.925** |
| deepseek-v4-pro | 0.922 (best single) |
| random routing | 0.875 |
| glm-5p2 | 0.783 |
| kimi-k2p6 | 0.539 |

**Both tasks together**

| system | math | MMLU | average |
| --- | --- | --- | --- |
| **TinyRouter** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro | 0.747 | 0.922 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 | 0.794 | 0.783 | 0.789 |
| kimi-k2p6 | 0.742 | 0.539 | 0.640 |

### What the numbers say

The tiny router scores **0.858 on average, higher than any single model**. No single model is good at
both tasks: deepseek is the knowledge specialist, glm is the math specialist. The router wins the
average by sending each task to the right specialist.

On MMLU, where the models differ a lot (0.54 to 0.92), routing clearly helps and the router beats
random routing (0.925 vs 0.875). On math, where all three models sit around 0.79, there is little
to route around, so the router ties both the best model and random routing. **Routing pays off when
the models genuinely differ.**

### Oracle ceiling diagnostic

| benchmark | best single | perfect router | real headroom (95% CI) | verdict |
| --- | --- | --- | --- | --- |
| math500 | 0.808 | **0.856** | **+0.049 [0.005, 0.085]** | ROUTER_BOUND |
| MMLU | 0.939 | ≥0.939 | +0.025 [0.000, 0.058] | inconclusive (near-ceiling) |

There is about 4.9 points of real, achievable headroom on math — the current router captures none of it.
This is the gap miners compete to close.

### Warm-start + shaped fitness experiment

| system | math (held-out 120) |
| --- | --- |
| best single (glm-5p2) | 0.817 |
| **TinyRouter (warm-start + shaped)** | **0.808** |
| prior router, same test | 0.792 |
| random routing | 0.733 |

The eval samples each model once per question, and that sampling noise is large: random routing alone
swung from 0.792 to 0.733 between runs with nothing changed. A swing that size swamps a 1.6-point
router delta. The warm-start and shaped-fitness upgrades are implemented and covered by 54 offline
tests; whether they move the held-out score is unproven.

## What we did

- Implemented the full coordinator: the 0.6B encoder feature, the ~10K routing head, the three roles,
  the multi-turn loop (up to 5 turns, terminated by a Verifier accept), and the sep-CMA-ES trainer.
- Wired a 3-model open-source pool plus an automatic grader (exact-match for math, letter-match for
  MMLU) that produces the binary reward.
- Trained per-task coordinators by evolution: breed thousands of candidate heads, keep the ones that
  route best, repeat.
- Evaluated rigorously on 120 held-out questions, with every single-model baseline averaged over 3 runs
  to remove run-to-run noise.
- Built an **oracle-ceiling diagnostic** to measure routing headroom.
- Built the **accuracy competition infrastructure**: submission pipeline, 8 anti-cheat gates,
  encrypted hidden benchmark, live leaderboard.
- Tracked every dollar of API spend (hash-chain verified cost ledger).

## Cost

Tracked exactly from the hash-chain-verified token ledgers at real Fireworks prices:

- Core replication and rigorous eval: **$20.89** (deepseek $6.56, glm $6.70, kimi $7.64).
- Oracle-ceiling diagnostic: **~$14**.
- Warm-start + shaped-fitness experiment (label collection, retrain, eval): **$27.22**.

## Repository

```
tinyrouter/
├── index.html             # project website
├── leaderboard.json        # live king-of-the-hill data
├── SUBMITTING.md           # miner competition guide
├── CONTRIBUTING.md         # contribution rules
├── ROADMAP.md              # future direction
├── src/trinity/            # coordinator (encoder, head, SVF, trainer, eval)
├── scripts/                # pack_submission, pr_eval, build_benchmark, audit_eval
├── configs/                # model pool + training config
├── tests/                  # 54+ tests
└── docs/                   # SPEC, JOURNAL, PAPER_NOTES
```

## Links

- [Website](https://james-cuda.github.io/Gittensor-TinyRouter/)
- [Leaderboard](leaderboard.json)
- [Submission Guide](SUBMITTING.md)
- [Contributing](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)
- [TRINITY Paper (arXiv:2512.04695)](https://arxiv.org/abs/2512.04695)
- [Gittensor](https://github.com/entrius/gittensor)
