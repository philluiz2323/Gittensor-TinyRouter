#!/usr/bin/env python3
"""Verify the model pool's price tables and membership lists are consistent, or report status.

The pool is defined in five places that carry "keep in sync" comments but that nothing
enforces: ``configs/models.yaml`` ``pool``, ``openrouter_pricing.OPENROUTER_POOL_PRICES``
(the self-declared single source of truth), ``oracle_ceiling.py::_DEFAULT_PRICES``,
``fugu.cost.PRICES``, and ``submission.constants.DEFAULT_POOL_MODELS`` (gate 6). Drift is
silent and costly — a lagging ``DEFAULT_POOL_MODELS`` false-rejects honest submissions at
gate 6; a lagging price makes the packed receipt cost wrong and trips the cost gates.
``config_check`` only checks the within-YAML structure, never these Python price tables.

    # Verify (CI-usable: exits non-zero on any drift):
    python scripts/verify_pool_pricing.py

    # Status report (never fails) / machine-readable dump:
    python scripts/verify_pool_pricing.py --report
    python scripts/verify_pool_pricing.py --json pool.json

Read-only, pure/offline (no torch, no network, no GPU). It can only *reject on drift*; it
never edits a config. This is the pool-pricing analogue of ``scripts/verify_benchmark.py``
and ``scripts/verify_leaderboard.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.llm.pool_consistency import (  # noqa: E402  (needs the sys.path insert)
    check_pool_consistency,
    gather_sources,
    render,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the model pool's price/membership consistency.")
    ap.add_argument("--root", default=str(_REPO), help="repo root holding configs/ and scripts/")
    ap.add_argument("--report", action="store_true", help="print a status report; always exit 0")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    sources = gather_sources(args.root)
    report = check_pool_consistency(sources)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2))

    if args.report:
        print(render(report))
        return

    if not report.ok:
        print(f"FAIL — {len(report.problems)} pool consistency problem(s):")
        for p in report.problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK — pool price/membership consistent across "
          f"{len(report.tables_checked)} price table(s) + membership ({', '.join(report.pool)})")


if __name__ == "__main__":
    main()
