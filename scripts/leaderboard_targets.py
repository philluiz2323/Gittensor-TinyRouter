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

from trinity.leaderboard import (  # noqa: E402
    load_leaderboard,
    summarize_competition,
    summarize_targets,
)


def main(argv: list[str] | None = None) -> int:
    """Print the composite score-to-beat + the per-benchmark reference targets."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaderboard", type=Path, default=_REPO / "leaderboard.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    lb = load_leaderboard(args.leaderboard)
    targets = summarize_targets(lb)
    competition = summarize_competition(lb)
    if args.json:
        print(json.dumps({
            "competition": competition.to_dict() if competition else None,
            "benchmarks": [t.to_dict() for t in targets],
        }, indent=2))
        return 0

    # Headline: the composite score every submission is actually approved against.
    if competition is not None:
        king = (f"{competition.king_miner} (gen {competition.king_generation}, "
                f"#{competition.king_pr})" if competition.has_king else "no king yet")
        print(f"COMPOSITE: beat >= {competition.score_to_beat:.4f} "
              f"(king {competition.current_best:.4f} + margin {competition.win_margin:.4f}) "
              f"[{king}]")
        if not competition.reachable:
            print("    (score-to-beat exceeds 1.0 — the crown is effectively unbeatable)")
        if competition.best_per_benchmark:
            bd = ", ".join(f"{b} {s:.4f}" for b, s in
                           sorted(competition.best_per_benchmark.items()))
            print(f"    king per-benchmark: {bd}")
        print()

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
