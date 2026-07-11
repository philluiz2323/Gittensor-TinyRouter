#!/usr/bin/env python3
"""Report training convergence + the R8 optimizer comparison from run artifacts.

Reads every ``<root>/**/summary.json`` (with its sibling ``history.json``) written by
``trinity.train`` / the baseline optimizers, and prints a per-run convergence table plus
the cross-run R8 ranking and the SPEC definition-of-done verdict ("the optimizer drives
J(θ) upward"). Nothing else consumed these artifacts.

    python scripts/convergence_report.py                 # Markdown to stdout
    python scripts/convergence_report.py --root experiments --json conv.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.convergence import RunConvergence, analyze_run, analyze_runs, render


def load_runs(root: str = "experiments") -> list[RunConvergence]:
    """Load every run under ``root`` that has a summary.json + history.json."""
    runs: list[RunConvergence] = []
    for sp in sorted(glob.glob(f"{root}/**/summary.json", recursive=True)):
        summary_path = Path(sp)
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        history_path = summary_path.with_name("history.json")
        history = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except Exception:
                history = []
        runs.append(analyze_run(summary, history, run_id=summary_path.parent.name))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Training-convergence + R8 optimizer report.")
    ap.add_argument("--root", default="experiments", help="root to glob <root>/**/summary.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a machine-readable JSON")
    args = ap.parse_args()

    runs = load_runs(args.root)
    print(render(runs))
    if args.json_out:
        payload = {"runs": [r.to_dict() for r in runs], "cross": analyze_runs(runs)}
        Path(args.json_out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
