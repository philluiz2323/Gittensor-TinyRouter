#!/usr/bin/env python3
"""TRINITY OpenRouter cost tracker.

Two modes:
  --ledger PATH   Exact: sum token usage recorded by OpenRouterPool (set
                  TRINITY_COST_LEDGER=PATH when running train/eval).
  --estimate      Approximate: estimate calls/tokens/cost from run configs, for runs
                  that happened BEFORE the ledger existed (the early pilots + the
                  currently in-flight parallel runs).

OpenRouter exposes exact token usage in every chat response, so we price from
tokens. PRICES below should track the repo's current default model pool; pass
--in/--out to override the blended rate for what-if estimates.

    python scripts/cost_report.py --estimate
    TRINITY_COST_LEDGER=~/trinity/cost_ledger.jsonl python -m trinity.train ...
    python scripts/cost_report.py --ledger ~/trinity/cost_ledger.jsonl
"""
from __future__ import annotations

import argparse
import sys

from trinity.llm.cost_ledger import read_ledger_entries, verify_ledger_chain

# ---- OpenRouter prices ($ per 1M tokens), (input, output). ----
PRICES = {
    "qwen3.5-35b-a3b": (0.14, 1.00),
    "minimax-m3":      (0.30, 1.20),
    "deepseek-v4-flash": (0.09, 0.18),
}
_DEFAULT_BLENDED_IN = sum(p[0] for p in PRICES.values()) / len(PRICES)
_DEFAULT_BLENDED_OUT = sum(p[1] for p in PRICES.values()) / len(PRICES)


def cost(prompt_tok: int, completion_tok: int, in_rate: float, out_rate: float) -> float:
    return prompt_tok / 1e6 * in_rate + completion_tok / 1e6 * out_rate


def report_ledger(path: str) -> None:
    # Verify hash-chain integrity first.
    valid, num_entries, err = verify_ledger_chain(path)
    if not valid:
        print(f"[FAIL] Cost ledger hash-chain verification failed: {err}", file=sys.stderr)
        print("       The ledger has been tampered with or corrupted. "
              "Cost totals are NOT trustworthy.", file=sys.stderr)
        sys.exit(2)

    print(f"[ OK ] Ledger hash-chain verified ({num_entries} entries, all hashes intact)\n")

    per = {}  # model -> [prompt, completion, calls]
    for entry in read_ledger_entries(path):
        acc = per.setdefault(entry.model, [0, 0, 0])
        acc[0] += entry.prompt_tokens
        acc[1] += entry.completion_tokens
        acc[2] += 1
    total = 0.0
    print(f"{'model':18s} {'calls':>8s} {'prompt_tok':>12s} {'compl_tok':>12s} {'$':>9s}")
    print("-" * 64)
    for m, (p, c, n) in sorted(per.items()):
        ir, orr = PRICES.get(m, (_DEFAULT_BLENDED_IN, _DEFAULT_BLENDED_OUT))
        d = cost(p, c, ir, orr)
        total += d
        print(f"{m:18s} {n:8d} {p:12d} {c:12d} {d:9.3f}")
    print("-" * 64)
    print(f"{'TOTAL (exact tokens, ASSUMED prices)':40s} ${total:.2f}")


# Per-run config estimates. avg_turns < max_turns due to early Verifier-ACCEPT.
# prompt grows with transcript; completion ~ fills max_tokens for reasoning models.
RUNS = [
    # name, generations, popsize, m_cma, max_turns, max_tokens, status
    ("pilot#1 (math)",       5, 6, 4, 3, 1024, "done"),
    ("pilot_crn (math)",     3, 8, 8, 3,  640, "done"),
    ("full_pilot (math)",   12, 8, 8, 3,  640, "done"),
    ("eval math500",         1, 0, 0, 0,    0, "done(special)"),
    ("eval mmlu",            1, 0, 0, 0,    0, "done(special)"),
    ("mmlu_pilot",          12, 8, 8, 3,  640, "running"),
    ("math_s0",             14, 8,10, 4,  768, "running"),
    ("math_s1",             14, 8,10, 4,  768, "running"),
    ("mmlu_s0",             14, 8,10, 4,  768, "running"),
    ("mmlu_s1",             14, 8,10, 4,  768, "running"),
]
_EVAL_CALLS = 40 * (3 * 1 + 2.5 + 2.5)  # 40 items: 3 single + TRINITY + random


def report_estimate(in_rate: float, out_rate: float, avg_turn_frac: float = 0.7,
                     avg_prompt: int = 650) -> None:
    print(f"Estimate @ blended ${in_rate:.2f}/1M in, ${out_rate:.2f}/1M out "
          f"(avg_turns={avg_turn_frac:g}*max, avg_prompt~{avg_prompt} tok, "
          f"completion~max_tokens):\n")
    print(f"{'run':22s} {'calls':>7s} {'Mtok':>7s} {'$':>8s}  status")
    print("-" * 60)
    grand_calls = grand_tok = grand_cost = 0.0
    for name, g, p, m, t, mt, status in RUNS:
        if "eval" in name:
            calls = _EVAL_CALLS
            compl = mt or 640
            ptok = calls * avg_prompt
            ctok = calls * 640
        else:
            avg_turns = max(1.0, avg_turn_frac * t)
            calls = g * p * m * avg_turns
            ptok = calls * avg_prompt
            ctok = calls * mt  # reasoning fills the budget
        d = cost(ptok, ctok, in_rate, out_rate)
        tok = (ptok + ctok) / 1e6
        grand_calls += calls; grand_tok += tok; grand_cost += d
        print(f"{name:22s} {calls:7.0f} {tok:7.2f} {d:8.2f}  {status}")
    print("-" * 60)
    print(f"{'TOTAL (when all finish)':22s} {grand_calls:7.0f} {grand_tok:7.2f} ${grand_cost:8.2f}")
    print("\nNOTE: prices should match the current OpenRouter default pool. Token/turn counts are "
          "estimated for pre-ledger runs and ignore prompt-caching discounts.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", help="path to cost_ledger.jsonl (exact)")
    ap.add_argument("--estimate", action="store_true", help="estimate from run configs")
    ap.add_argument("--in", dest="in_rate", type=float, default=_DEFAULT_BLENDED_IN)
    ap.add_argument("--out", dest="out_rate", type=float, default=_DEFAULT_BLENDED_OUT)
    args = ap.parse_args()
    if args.ledger:
        report_ledger(args.ledger)
    elif args.estimate:
        report_estimate(args.in_rate, args.out_rate)
    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
