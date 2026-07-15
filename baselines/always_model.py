#!/usr/bin/env python3
"""Baseline: always pick one model for every question.

This is the simplest possible strategy — route every query to the same model.
The best of the three models on each benchmark is "best-single", the baseline
that any useful router must at least match.

Usage:
    python baselines/always_model.py --model qwen3.5-35b-a3b --benchmark math500
    python baselines/always_model.py --model gemini-3.1-flash-lite --benchmark mmlu
    python baselines/always_model.py --model deepseek-v4-flash --benchmark livecodebench
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


async def run(model: str, benchmark: str, max_items: int) -> float:
    from trinity.adapters import get_adapter
    from trinity.llm.openrouter_client import OpenRouterPool
    from trinity.orchestration import reward as R
    from trinity.roles.prompts import build_messages
    from trinity.types import Role

    adapter = get_adapter(benchmark)
    tasks = adapter.load_tasks("test", max_items=max_items, seed=42)
    pool = OpenRouterPool(str(_REPO / "configs" / "models.yaml"))
    print(f"[baseline] always-{model} on {benchmark}: {len(tasks)} tasks")

    import httpx
    scores = []
    async with httpx.AsyncClient() as cli:
        async def one(task):
            msgs = build_messages(Role.WORKER, adapter.build_prompt(task), [])
            res = await pool.chat(model, msgs, max_tokens=4096, temperature=0.0,
                                  reasoning="minimal", client=cli)
            return adapter.score_output(res.text, task.answer)
        scores = await asyncio.gather(*[one(t) for t in tasks])

    acc = float(mean(scores))
    print(f"[baseline] always-{model} on {benchmark}: accuracy = {acc:.4f}")
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description="Always-pick-one-model baseline")
    ap.add_argument("--model", required=True,
                    help="Model name (qwen3.5-35b-a3b, gemini-3.1-flash-lite, deepseek-v4-flash)")
    ap.add_argument("--benchmark", required=True,
                    help="Benchmark name (math500, mmlu, livecodebench)")
    ap.add_argument("--max-items", type=int, default=120, dest="max_items")
    args = ap.parse_args()
    asyncio.run(run(args.model, args.benchmark, args.max_items))


if __name__ == "__main__":
    main()
