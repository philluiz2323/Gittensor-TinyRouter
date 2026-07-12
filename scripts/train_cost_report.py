#!/usr/bin/env python3
"""Project the API spend of a sep-CMA-ES head-training run before launching it.

Prices the run (atomic evals = population x m_cma x generations, each making
avg_turns worker calls) from the pool prices, and flags a plan that would fall
below the $15 receipt floor. Exits non-zero if the projected cost is below that
floor, so a plan that could not produce a valid receipt is caught early.

    python scripts/train_cost_report.py --population 33 --m-cma 16 --generations 60 \\
        --workers qwen3.5-35b-a3b minimax-m3 deepseek-v4-flash

Prices default to fugu.cost.PRICES; pass --price NAME IN OUT to override/add.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.train_cost import estimate_cmaes_cost  # noqa: E402


def _prices(overrides: list[list[str]] | None) -> dict[str, tuple[float, float]]:
    try:
        from trinity.fugu.cost import PRICES
        prices = dict(PRICES)
    except Exception:
        prices = {}
    for name, pin, pout in (overrides or []):
        prices[name] = (float(pin), float(pout))
    return prices


def main(argv: list[str] | None = None) -> int:
    """Print the projected training cost; exit non-zero below the receipt floor."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--population", type=int, required=True)
    ap.add_argument("--m-cma", type=int, required=True, dest="m_cma")
    ap.add_argument("--generations", type=int, required=True)
    ap.add_argument("--workers", nargs="+", required=True)
    ap.add_argument("--avg-turns", type=float, default=2.5, dest="avg_turns")
    ap.add_argument("--avg-prompt-tokens", type=int, default=1200, dest="avg_prompt_tokens")
    ap.add_argument("--avg-completion-tokens", type=int, default=800, dest="avg_completion_tokens")
    ap.add_argument("--price", nargs=3, action="append", metavar=("NAME", "IN", "OUT"),
                    dest="price_overrides")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    est = estimate_cmaes_cost(
        population_size=args.population, m_cma=args.m_cma, generations=args.generations,
        worker_names=args.workers, prices=_prices(args.price_overrides),
        avg_turns=args.avg_turns, avg_prompt_tokens=args.avg_prompt_tokens,
        avg_completion_tokens=args.avg_completion_tokens,
    )
    if args.json:
        print(json.dumps(est.to_dict(), indent=2))
    else:
        print(f"[train-cost] {est.atomic_evals:,} atomic evals -> {est.worker_calls:,} "
              f"worker calls -> ~${est.total_usd:,.2f}")
        for name, usd in sorted(est.per_model_usd.items()):
            print(f"    {name}: ${usd:,.2f}")
        if est.below_receipt_floor:
            print(f"    WARNING: below the ${est.to_dict()['min_receipt_usd']:.0f} receipt "
                  "floor — this run could not produce a valid submission receipt.")
    return 1 if est.below_receipt_floor else 0


if __name__ == "__main__":
    raise SystemExit(main())
