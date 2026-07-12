#!/usr/bin/env python3
"""Verify the integrity of leaderboard.json, or print a status report.

``scripts/pr_eval.py`` writes ``leaderboard.json`` (the committed competition record and
the rate-limit ``attempts`` ledger ``gates.check_rate_limit`` trusts), but nothing ever
checks those writes are consistent — even though ``docs/CI.md`` marks it a sensitive
path. This is the leaderboard analogue of ``scripts/verify_benchmark.py`` (#174).

    # Verify (CI-usable: exits non-zero on any problem):
    python scripts/verify_leaderboard.py
    python scripts/verify_leaderboard.py --file leaderboard.json

    # Status report (per-benchmark frontier; --rate-status adds weekly submission counts):
    python scripts/verify_leaderboard.py --report [--rate-status]

Read-only: it detects tampering (an inflated best_score, a score above the oracle
ceiling, best_* not matching the winning history entry, a truncated attempts ledger that
would defeat the weekly rate limit) — it never writes. Reuses the gate's own parsers so
it cannot drift from ``check_rate_limit``. Pure/offline (no torch, no network).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.submission.leaderboard import (  # noqa: E402  (needs the sys.path insert)
    leaderboard_report,
    verify_leaderboard,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify leaderboard.json integrity / print status.")
    ap.add_argument("--file", default=str(_REPO / "leaderboard.json"), help="path to leaderboard.json")
    ap.add_argument("--report", action="store_true", help="print a status report instead of verifying")
    ap.add_argument("--rate-status", action="store_true", dest="rate_status",
                    help="with --report, include per-miner weekly submission counts")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(2)
    lb = json.loads(path.read_text())

    if args.report:
        print(leaderboard_report(lb, now=time.time() if args.rate_status else None))
        return

    problems = verify_leaderboard(lb)
    if problems:
        print(f"FAIL [{path}] — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK [{path}] — leaderboard integrity verified")


if __name__ == "__main__":
    main()
