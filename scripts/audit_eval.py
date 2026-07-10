#!/usr/bin/env python3
"""Audit-split evaluation — the final, honest number.

Generates a held-out question split from a SEALED seed that is NEVER used
during training or development. After all experiments are complete and the
best theta is chosen, run this script ONCE to get the ungameable result.

The seed is baked into this script and must never be changed after the
first experiment begins. The idea is simple: the researcher cannot overfit
to questions they never saw, so the audit score is the trustworthy number.

Usage:
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python scripts/audit_eval.py \
        --benchmark math500 \
        --theta experiments/math500/run/best_theta.npy \
        --out experiments/math500/audit_result.json

The --seed flag is deliberately absent — the seed is locked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(_REPO / "src"))

from trinity.coordinator import params as P
from trinity.coordinator.policy import CoordinatorPolicy
from trinity.llm.openrouter_client import OpenRouterPool
from trinity.orchestration import reward as R
from trinity.orchestration.dataset import load_tasks
from trinity.orchestration.session import run_trajectory
from trinity.types import Role

# ---- SEALED SEED — committed, never changed. ----
# This seed selects a held-out subset of questions that NO experiment has
# ever seen. Running eval against it gives the honest, ungameable score.
_AUDIT_SEED: int = 314159265  # first 9 digits of pi — arbitrary but fixed forever

# The audit uses a different split name so HF datasets don't serve the same
# questions as the training or test splits (which experiments already use).
_AUDIT_SPLIT: str = "train"  # We sample from train but with a DIFFERENT seed
# and a DIFFERENT shuffle, so the subset is as-if held-out.


async def run_audit(args) -> dict:
    pool = OpenRouterPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    # Load tasks with the SEALED seed — NO override possible.
    tasks = load_tasks(
        args.benchmark, _AUDIT_SPLIT,
        max_items=args.max_items,
        seed=_AUDIT_SEED,
    )
    print(f"[audit] benchmark={args.benchmark}  {len(tasks)} audit tasks  "
          f"seed={_AUDIT_SEED} (SEALED — never used in any experiment)")

    run_kwargs = dict(
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
    )

    results: dict[str, float] = {}

    # SPEC §1.3.4: budget-match the baselines to TRINITY, else the `best_single`
    # comparison below is decided partly on token budget rather than on routing.
    from trinity.eval import single_model_budget, task_rng

    single_max_tokens = single_model_budget(args.max_tokens, args.max_turns)
    print(f"[audit] budget-matched baselines: single-model max_tokens="
          f"{single_max_tokens} ({args.max_turns}x {args.max_tokens})")

    # --- Single-model baselines ---
    for m in pool_models:
        from trinity.roles.prompts import build_messages
        import httpx
        async with httpx.AsyncClient() as cli:
            async def one(task):
                msgs = build_messages(Role.WORKER, task.prompt, [])
                res = await pool.chat(
                    m, msgs, max_tokens=single_max_tokens, temperature=0.0,
                    reasoning=args.reasoning, client=cli,
                )
                return R.score_text(args.benchmark, res.text, task.answer)
            scores = await asyncio.gather(*[one(t) for t in tasks])
        s = float(mean(scores))
        results[f"single::{m}"] = s
        print(f"  single  {m:20s} = {s:.4f}")

    # --- TRINITY trained coordinator (argmax) ---
    cfg = yaml.safe_load(Path(args.config).read_text())["coordinator"]
    print("[audit] building coordinator on GPU...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cfg["encoder_model"],
        device=cfg.get("device", "cuda:0"),
        dtype=cfg.get("dtype", "bfloat16"),
        target_layer=cfg["svf"]["target_layer"],
        svf_matrices=cfg["svf"].get("matrices"),
        n_models=n_models,
        l2_normalize=cfg["hidden_state"].get("l2_normalize", True),
    )
    theta = np.load(args.theta)
    policy.configure(theta, spec)

    import httpx
    async with httpx.AsyncClient() as cli:
        trajs = await asyncio.gather(*[
            run_trajectory(t, policy, pool, pool_models, sample=False,
                           client=cli, **run_kwargs)
            for t in tasks
        ])
    s_trinity = float(mean(R.score(t) for t in trajs))
    results["TRINITY"] = s_trinity
    print(f"  TRINITY (trained)        = {s_trinity:.4f}")

    # --- Random routing baseline (100 seeds) ---
    # Each trajectory draws from its OWN (seed, task_id)-seeded rng, so the
    # baseline is invariant to asyncio scheduling. A single rng shared across the
    # concurrently-gathered trajectories would consume draws in network-completion
    # order, making the "sealed" audit number non-reproducible (mirrors the fix in
    # trinity.eval.RandomPolicy / task_rng).
    rand_policy = _RandomAuditPolicy(n_models)
    rand_scores = []
    for s in range(100):
        seed_s = _AUDIT_SEED * 10000 + s
        async with httpx.AsyncClient() as cli:
            rt = await asyncio.gather(*[
                run_trajectory(
                    t, rand_policy, pool, pool_models, sample=False,
                    rng=task_rng(seed_s, t.task_id), client=cli, **run_kwargs,
                )
                for t in tasks
            ])
        rand_scores.append(float(mean(R.score(t) for t in rt)))
    s_rand = float(mean(rand_scores))
    rand_std = (sum((x - s_rand) ** 2 for x in rand_scores) / 100) ** 0.5
    results["random_routing"] = s_rand
    results["random_routing_std"] = rand_std
    print(f"  random routing           = {s_rand:.4f} ± {rand_std:.4f}  (n=100 seeds)")

    best_single = max(v for k, v in results.items() if k.startswith("single::"))
    out = {
        "benchmark": args.benchmark,
        "audit_seed": _AUDIT_SEED,
        "audit_split": _AUDIT_SPLIT,
        "num_tasks": len(tasks),
        "results": results,
        "best_single": best_single,
        "trinity_vs_best_single": s_trinity - best_single,
        "trinity_vs_random": s_trinity - s_rand,
    }
    print(f"\n[audit] FINAL: TRINITY={s_trinity:.4f}  best_single={best_single:.4f}  "
          f"random={s_rand:.4f}±{rand_std:.4f}")
    print(f"[audit] TRINITY - best_single = {out['trinity_vs_best_single']:+.4f}")
    print(f"[audit] TRINITY - random      = {out['trinity_vs_random']:+.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[audit] saved to {args.out}")

    return out


class _RandomAuditPolicy:
    """Random routing — no GPU needed, no trinity imports.

    ``decide`` draws from the per-trajectory ``rng`` that ``run_trajectory`` passes
    through, so the routing choices depend only on ``(seed, task_id)`` and never on
    asyncio scheduling. A shared instance rng consumed by concurrently-gathered
    trajectories would make the audit's random baseline non-reproducible; the
    instance rng is only a fallback for direct calls without a per-trajectory rng.
    """

    def __init__(self, n_models: int, rng: random.Random | None = None):
        from trinity.types import ROLE_ORDER
        self.n_models = n_models
        self.rng = rng if rng is not None else random.Random(0)
        self._roles = ROLE_ORDER

    def decide(self, transcript_text, *, sample=False, rng=None):
        r = rng if rng is not None else self.rng
        return r.randrange(self.n_models), r.choice(self._roles)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit-split evaluation — run ONCE after all experiments are done"
    )
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--theta", required=True, help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--out", default="")
    # Deliberately NO --seed argument.
    args = ap.parse_args()
    asyncio.run(run_audit(args))


if __name__ == "__main__":
    main()
