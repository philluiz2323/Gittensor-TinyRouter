#!/usr/bin/env python3
"""Offline efficiency + composite-score report for a Conductor.

Two modes, no API cost:

Predict the composite score from its four inputs (see CONTRIBUTING.md):

    python scripts/efficiency_report.py --score \\
        --hidden-acc 0.82 --live-acc 0.70 --avg-turns 2.4 --novelty 0.1

Summarize per-answer efficiency from a JSON list of live task outcomes
(``[{"correct": true, "turns": 2, "llm_calls": 3, "cost_usd": 0.004}, ...]``):

    python scripts/efficiency_report.py --records live_outcomes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.efficiency import (  # noqa: E402
    DEFAULT_MAX_TURNS,
    TurnRecord,
    composite_score,
    summarize_efficiency,
)


def _load_records(path: Path) -> list[TurnRecord]:
    data = json.loads(path.read_text())
    rows = data["records"] if isinstance(data, dict) else data
    return [
        TurnRecord(
            correct=bool(r["correct"]),
            turns=int(r.get("turns", 0)),
            llm_calls=(int(r["llm_calls"]) if r.get("llm_calls") is not None else None),
            cost_usd=(float(r["cost_usd"]) if r.get("cost_usd") is not None else None),
        )
        for r in rows
    ]


def main(argv: list[str] | None = None) -> int:
    """Print a composite-score prediction or a per-answer efficiency summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score", action="store_true", help="predict the composite score")
    mode.add_argument("--records", type=Path, help="JSON list of live task outcomes")
    ap.add_argument("--hidden-acc", type=float, default=0.0, dest="hidden_acc")
    ap.add_argument("--live-acc", type=float, default=0.0, dest="live_acc")
    ap.add_argument("--avg-turns", type=float, default=1.0, dest="avg_turns")
    ap.add_argument("--novelty", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, dest="max_turns")
    args = ap.parse_args(argv)

    if args.score:
        breakdown = composite_score(
            hidden_acc=args.hidden_acc, live_acc=args.live_acc,
            avg_turns_used=args.avg_turns, novelty=args.novelty, max_turns=args.max_turns,
        )
        print(json.dumps(breakdown.to_dict(), indent=2))
        print(f"\n[score] composite = {breakdown.total:.4f} "
              f"(efficiency term {breakdown.efficiency:.4f})")
        return 0

    records = _load_records(args.records)
    if not records:
        print("[efficiency] no records found.")
        return 1
    summary = summarize_efficiency(records, max_turns=args.max_turns)
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"\n[efficiency] {summary.n_correct}/{summary.n_tasks} correct; "
          f"avg {summary.avg_turns:.2f} turns/task; "
          f"{summary.turns_per_correct:.2f} turns per correct answer; "
          f"efficiency term {summary.efficiency:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
