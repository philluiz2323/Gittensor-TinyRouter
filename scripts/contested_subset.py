#!/usr/bin/env python3
"""Derive the contested (disagreement) subset from cached benchmark answers.

Reads a benchmark JSON built by ``scripts/build_benchmark.py`` (each item carries
one cached answer per model), grades those cached answers, and writes:

* an **oracle matrix** JSON, ready for ``scripts/oracle_ceiling.py --analyze``;
* the **contested question ids** — the disagreement subset that ``docs/JOURNAL.md``
  prescribes for GRPO training (on every other question the reward is constant
  across routing choices, so the within-group advantage is identically zero).

Costs nothing: it re-reads answers that were already paid for.

    python scripts/contested_subset.py --benchmark-json experiments/bench.json \\
        --out-matrix experiments/oracle_matrix_math500.json \\
        --out-contested experiments/contested_math500.json

Then, with no further API spend:

    python scripts/oracle_ceiling.py --analyze experiments/oracle_matrix_math500.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis import (  # noqa: E402
    contested_ids,
    grade_items,
    summarize,
    to_oracle_matrix,
)


def _load_items(path: Path) -> list[dict[str, Any]]:
    """Read a benchmark JSON, accepting either a bare list or an ``items`` wrapper."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return list(data)
    for key in ("items", "questions", "tasks"):
        value = data.get(key)
        if isinstance(value, list):
            return list(value)
    raise ValueError(
        f"{path}: expected a list of items or an object with an 'items' list, "
        f"got keys {sorted(data)[:6]}"
    )


def main(argv: list[str] | None = None) -> int:
    """Grade cached answers, report agreement, and export the contested subset."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark-json", required=True, type=Path,
                    help="benchmark file produced by scripts/build_benchmark.py")
    ap.add_argument("--out-matrix", type=Path, default=None,
                    help="write the oracle_ceiling --analyze matrix here")
    ap.add_argument("--out-contested", type=Path, default=None,
                    help="write the contested question ids here")
    ap.add_argument("--benchmark", default=None,
                    help="override the benchmark name stamped on the matrix")
    ap.add_argument("--no-cached-scores", action="store_true",
                    help="re-grade every answer instead of trusting item['model_scores']")
    args = ap.parse_args(argv)

    items = _load_items(args.benchmark_json)
    records = grade_items(items, use_cached_scores=not args.no_cached_scores)
    if not records:
        print("[contested] no cached model answers found; nothing to analyze.")
        return 1

    summary = summarize(records)
    contested = contested_ids(records)

    print(json.dumps(summary.to_dict(), indent=2))
    print(
        f"\n[contested] {summary.n_contested}/{summary.n_questions} questions are contested "
        f"({summary.disagreement_rate:.1%}). "
        f"{summary.n_unanimous_correct} solved by all, {summary.n_unanimous_wrong} by none — "
        f"those carry no routing gradient."
    )
    print(
        f"[contested] best single model {summary.best_single_model} at "
        f"{summary.best_single_accuracy:.3f}; any-model oracle {summary.oracle_any:.3f}; "
        f"raw headroom {summary.headroom:.3f}"
    )

    if args.out_matrix:
        matrix = to_oracle_matrix(records, benchmark=args.benchmark)
        args.out_matrix.parent.mkdir(parents=True, exist_ok=True)
        args.out_matrix.write_text(json.dumps(matrix, indent=2))
        print(f"[contested] wrote matrix -> {args.out_matrix}")
        print(f"[contested] next: python scripts/oracle_ceiling.py --analyze {args.out_matrix}")

    if args.out_contested:
        payload = {
            "benchmark": args.benchmark or (records[0].benchmark if records else None),
            "n_contested": len(contested),
            "n_questions": summary.n_questions,
            "question_ids": contested,
        }
        args.out_contested.parent.mkdir(parents=True, exist_ok=True)
        args.out_contested.write_text(json.dumps(payload, indent=2))
        print(f"[contested] wrote contested subset -> {args.out_contested}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
