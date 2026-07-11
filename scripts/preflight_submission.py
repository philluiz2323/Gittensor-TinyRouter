#!/usr/bin/env python3
"""Run offline anti-cheat gates on a routing-head submission before opening a PR.

Mirrors ``scripts/pr_eval.py`` gates 1–7 (rate limit, weights, duplicate,
receipt plausibility, ledger/receipt cost consistency, pack schema,
theta integrity) with **no GPU** and **no OpenRouter API** calls. Use this
after ``pack_submission.py`` to catch rejections locally.

Usage::

    export TRINITY_COST_LEDGER=~/trinity/cost_ledger.jsonl   # optional, gate 5
    python scripts/preflight_submission.py --submission alice/1 --benchmark math500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.submission.preflight import PreflightRunner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline submission preflight (no GPU/API)")
    ap.add_argument(
        "--submission",
        required=True,
        help="Path under submissions/, e.g. alice/1",
    )
    ap.add_argument("--benchmark", default="math500", help="Benchmark name (math500 or mmlu)")
    ap.add_argument(
        "--ledger",
        default=None,
        help="Cost ledger path (defaults to TRINITY_COST_LEDGER env var)",
    )
    ap.add_argument(
        "--repo-root",
        default=str(_REPO),
        help="Repository root containing submissions/ and leaderboard.json",
    )
    args = ap.parse_args()

    ledger_path = args.ledger or os.environ.get("TRINITY_COST_LEDGER")
    runner = PreflightRunner(
        repo_root=Path(args.repo_root),
        benchmark=args.benchmark,
        ledger_path=ledger_path,
    )
    report = runner.run(args.submission)

    print(f"\n{'=' * 60}")
    print(f"[preflight] submissions/{args.submission} — {args.benchmark}")
    print(f"{'=' * 60}")
    for line in report.summary_lines():
        print(line)

    if not report.passed:
        failure = report.first_failure
        code = failure.reason if failure else "preflight_failed"
        print(f"\n[preflight] REJECTED: {code}", file=sys.stderr)
        sys.exit(1)

    print("\n[preflight] Ready to open a PR.")
    sys.exit(0)


if __name__ == "__main__":
    main()
