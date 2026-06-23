# TRINITY: a tiny LLM router

We built a small **coordinator** that, for every question, decides two things: **which** of three
open-source LLMs should answer it, and **what role** that model should play (Thinker, Worker, or
Verifier). The coordinator is deliberately tiny and cheap. A frozen **0.6B** encoder reads the
question into a single vector, and a **~10K-parameter** head turns that vector into the routing
decision. It is trained by **separable CMA-ES**, a derivative-free evolution strategy, against a simple
right/wrong reward. The coordinator never solves the question itself; it only learns who to ask.

The method follows TRINITY (Xu et al., ICLR 2026, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)),
rebuilt from scratch with an all open-source model pool served through Fireworks AI.

## What we did

- Implemented the full coordinator: the 0.6B encoder feature, the ~10K routing head, the three roles,
  the multi-turn loop (up to 5 turns, terminated by a Verifier accept), and the sep-CMA-ES trainer.
- Wired a 3-model open-source pool plus an automatic grader (exact-match for math, letter-match for
  MMLU) that produces the binary reward.
- Trained per-task coordinators by evolution: breed thousands of candidate heads, keep the ones that
  route best, repeat.
- Evaluated rigorously on 120 held-out questions, with every single-model baseline averaged over 3 runs
  to remove run-to-run noise, against each model alone and against random routing.
- Tracked every dollar of API spend and logged each result.

## Model pool

| Slot | Model            | Strong at        |
| ---- | ---------------- | ---------------- |
| A    | `deepseek-v4-pro` | knowledge (MMLU) |
| B    | `glm-5p2`         | math             |
| C    | `kimi-k2p6`       | general          |

The 0.6B encoder and the evolution loop run on a single NVIDIA H200; the three LLMs are called over HTTP.

## How it works

1. The frozen 0.6B encoder turns the question into one 1024-dim vector.
2. The ~10K head reads that vector and picks a model and a role.
3. The chosen model answers in that role; its output is appended to the transcript.
4. Steps 1 to 3 repeat for up to 5 turns; a Verifier turn can accept and stop early.
5. The final answer is graded right/wrong, and that reward drives the evolutionary training.

## Results

Rigorous eval: 120 held-out questions per task; single-model baselines are the mean over 3 runs.
Scores are fraction correct (0.792 = 79.2%).

**Math**

| system | score |
| --- | --- |
| glm-5p2 | 0.794 (best single) |
| **TRINITY (router)** | **0.792** |
| random routing | 0.792 |
| deepseek-v4-pro | 0.747 |
| kimi-k2p6 | 0.742 |

**MMLU**

| system | score |
| --- | --- |
| **TRINITY (router)** | **0.925** |
| deepseek-v4-pro | 0.922 (best single) |
| random routing | 0.875 |
| glm-5p2 | 0.783 |
| kimi-k2p6 | 0.539 |

**Both tasks together**

| system | math | MMLU | average |
| --- | --- | --- | --- |
| **TRINITY (router)** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro | 0.747 | 0.922 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 | 0.794 | 0.783 | 0.789 |
| kimi-k2p6 | 0.742 | 0.539 | 0.640 |

### What the numbers say

The tiny router scores **0.858 on average, higher than any single model**. No single model is good at
both tasks: deepseek is the knowledge specialist, glm is the math specialist. The router wins the
average by sending each task to the right specialist.

Reading it honestly: the win is **across tasks, not within a task**. On MMLU, where the models differ a
lot (0.54 to 0.92), routing clearly helps and the router beats random (0.925 vs 0.875). On math, where
all three models sit around 0.79, there is nothing to route around, so the router ties both the best
model and random routing. Routing pays off when the models genuinely differ.

## Setup

```bash
# secrets live OUTSIDE the repo
cp .env.example ~/.config/trinity/secrets.env   # fill in FIREWORKS_API_KEY, then: chmod 600
source ~/.config/trinity/secrets.env

uv venv && source .venv/bin/activate
uv pip install -e .

# confirm the three models answer
python -m trinity.llm.fireworks_client --selftest
```

## Run

```bash
source ~/.config/trinity/secrets.env
# train a per-task coordinator on the GPU
bash scripts/run_remote.sh train --benchmark math500
# rigorous eval (120 items, baselines averaged over 3 runs)
python -m trinity.eval --benchmark math500 \
    --theta experiments/math500/full_pilot/best_theta.npy \
    --max-items 120 --single-reps 3 --out experiments/final/math_rigorous.json
python scripts/cost_report.py --ledger cost_ledger.jsonl   # spend
```

Secrets never live in this repo: the SSH key sits in `~/.ssh/`, the Fireworks key in
`~/.config/trinity/secrets.env`.

## Cost

**$20.89** total, exact from the token ledger at real Fireworks prices (deepseek $6.56, glm $6.70,
kimi $7.64).
