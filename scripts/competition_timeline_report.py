#!/usr/bin/env python3
"""Print the competition's king-progression timeline from leaderboard.json.

Reads competition.history and reports the reign-by-reign progression: each king's
composite score, per-benchmark breakdown, and the gain over the previous king, plus the
total gain and the biggest single leap. Zero API cost.

    python scripts/competition_timeline_report.py                       # ./leaderboard.json
    python scripts/competition_timeline_report.py --leaderboard path/to/leaderboard.json
    python scripts/competition_timeline_report.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.competition_timeline import load_timeline, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the competition king-progression timeline."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaderboard", type=Path, default=_REPO / "leaderboard.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    timeline = load_timeline(args.leaderboard)
    if args.json:
        print(json.dumps(timeline.to_dict(), indent=2))
    else:
        print(render(timeline), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
