#!/usr/bin/env python3
"""Audit every registered benchmark adapter against the BenchmarkAdapter contract.

``adapters/base.py`` documents the contract (binary ``score_output``, valid ``task_type``,
non-empty ``scoring_modes``, deterministic ``load_tasks``, JSON-safe ``serialize_task``),
but only ``tests/test_benchmark_registry.py`` spot-checks the scoring invariant, and only
for 3 of the 9 adapters. This CLI audits ALL of them uniformly and exits non-zero on any
drift — a CI-usable gate for a growing adapter suite.

    # Verify (exits non-zero on any contract failure):
    python scripts/adapter_conformance.py

    # Status report (always exit 0) / machine-readable dump:
    python scripts/adapter_conformance.py --report
    python scripts/adapter_conformance.py --json conformance.json

Read-only, pure/offline: the adapters load via their toy fallback (no network / datasets /
torch). Sits beside verify_benchmark / verify_leaderboard as the adapter-contract analogue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.adapters.conformance import audit_all, render  # noqa: E402  (needs sys.path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit benchmark adapters against the contract.")
    ap.add_argument("--report", action="store_true", help="print the report; always exit 0")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    report = audit_all()
    print(render(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2))

    if args.report:
        return
    if not report.ok:
        print(f"FAIL — {len(report.failures())} adapter contract failure(s)")
        sys.exit(1)
    print(f"OK — {len(report.adapters)} adapters conform ({len(report.results)} checks)")


if __name__ == "__main__":
    main()
