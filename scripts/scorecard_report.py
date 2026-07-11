#!/usr/bin/env python3
"""Offline predicted composite-score RANGE from a built benchmark.

Grades the cached ``model_answers`` to bound the hidden (70%) term by the best
single model (floor) and the any-model oracle (ceiling), then combines each bound
with the live / efficiency / novelty inputs you provide to report the composite
score the submission would earn at the floor and at the ceiling.

    python scripts/scorecard_report.py --benchmark-json experiments/bench.json \\
        --live-acc 0.70 --avg-turns 2.4 --novelty 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.efficiency import DEFAULT_MAX_TURNS  # noqa: E402
from trinity.scorecard import scorecard  # noqa: E402


def _load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return list(data)
    for key in ("items", "questions", "tasks"):
        value = data.get(key)
        if isinstance(value, list):
            return list(value)
    raise ValueError(f"{path}: expected a list of items or an 'items' list")


def main(argv: list[str] | None = None) -> int:
    """Print the predicted composite-score range."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark-json", required=True, type=Path)
    ap.add_argument("--live-acc", type=float, default=0.0, dest="live_acc")
    ap.add_argument("--avg-turns", type=float, default=1.0, dest="avg_turns")
    ap.add_argument("--novelty", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, dest="max_turns")
    args = ap.parse_args(argv)

    card = scorecard(
        _load_items(args.benchmark_json),
        live_acc=args.live_acc, avg_turns=args.avg_turns,
        novelty=args.novelty, max_turns=args.max_turns,
    )
    print(json.dumps(card.to_dict(), indent=2))
    print(
        f"\n[scorecard] hidden accuracy in "
        f"[{card.best_single_accuracy:.3f} (best single {card.best_single_model}), "
        f"{card.oracle_accuracy:.3f} (oracle)]; routing headroom {card.headroom:.3f}"
    )
    print(
        f"[scorecard] predicted composite score in "
        f"[{card.score_floor.total:.4f}, {card.score_ceiling.total:.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
