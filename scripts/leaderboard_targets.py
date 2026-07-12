#!/usr/bin/env python3
"""What score must a submission beat, per benchmark?

Reads leaderboard.json and reports, per benchmark, the score to beat (become king
only by scoring strictly above it), the current king, and how much routing
headroom is left (oracle ceiling minus the best single model) versus how much the
current best already captured.

    python scripts/leaderboard_targets.py                 # reads ./leaderboard.json
    python scripts/leaderboard_targets.py --leaderboard path/to/leaderboard.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.leaderboard import load_leaderboard, summarize_targets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the per-benchmark targets."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaderboard", type=Path, default=_REPO / "leaderboard.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    targets = summarize_targets(load_leaderboard(args.leaderboard))
    if args.json:
        print(json.dumps([t.to_dict() for t in targets], indent=2))
        return 0
    if not targets:
        print(f"[leaderboard] no benchmarks found in {args.leaderboard}")
        return 1
    for t in targets:
        king = f"{t.king_miner} (gen {t.king_generation})" if t.has_king else "no king yet"
        print(f"{t.benchmark}: beat > {t.score_to_beat:.4f} [{king}]")
        print(f"    single-model {t.best_single_model:.4f} -> oracle {t.oracle_ceiling:.4f} "
              f"(headroom {t.headroom:.4f}); best has captured {t.captured:.4f}, "
              f"{t.remaining:.4f} still on the table")
        if not t.contested:
            print("    (no headroom left above the current best — hard to improve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
