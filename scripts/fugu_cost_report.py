#!/usr/bin/env python3
"""Report the Fugu Conductor cost & worker-utilization audit from baseline artifacts.

Reads the ``fugu_baseline_<bench>.json`` files ``scripts/fugu_baseline_eval.py`` writes
and reports, per run, the per-worker token/cost shares, the effective number of workers
(1/HHI), the fanout tax, $/correct, and the routing-vs-test-time-compute verdict — the
analysis docs/fugu/BASELINE_RESULTS.md currently does by hand.

    python scripts/fugu_cost_report.py experiments/final/fugu_baseline_*.json
    python scripts/fugu_cost_report.py --root experiments --json fugu_cost.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.fugu.cost_audit import analyze, render


def _files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/fugu_baseline_*.json", recursive=True))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fugu Conductor cost & worker-utilization report.")
    ap.add_argument("files", nargs="*", help="fugu_baseline_<bench>.json file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/fugu_baseline_*.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _files(args.files, args.root)
    if not files:
        print("no fugu_baseline JSONs given (pass files or --root)")
        return
    reports = []
    for f in files:
        try:
            baseline = json.loads(Path(f).read_text())
        except Exception:
            continue
        summary = analyze(baseline)
        print(render(summary))
        reports.append(summary.to_dict())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
