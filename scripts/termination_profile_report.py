#!/usr/bin/env python3
"""Report how the coordinator's multi-turn loop terminates — offline.

Complements `trinity.efficiency` (which collapses turn usage into one competition-
score scalar): this reads a run's per-trajectory termination records — how many turns
each took and whether it stopped on a Verifier ACCEPT — and reports, per benchmark and
for the pooled union, the accept rate, the exhausted rate (ran to the turn budget), the
mean/median turns, the turn histogram, and a flag when the Verifier never accepts (the
accept/revise loop is inert and every question paid the full budget). Zero API cost.

Input JSON (`--records`), either record shape:

    {"math500": [[2, "accept"], [5, "max_turns"], {"turns": 3, "accepted": true}],
     "mmlu":    [{"turns": 5, "terminated_by": null}, ...]}

    python scripts/termination_profile_report.py --records termination.json

Exits non-zero when any benchmark's Verifier never accepts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.termination_profile import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of benchmark -> [records]")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the termination-profile report; exit non-zero if any loop never accepts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True, type=Path,
                    help="JSON of benchmark -> list of (turns, accepted) trajectory records")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    per_benchmark = _load(args.records)
    report = analyze_benchmarks(per_benchmark)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(per_benchmark))
    return 1 if report["any_never_accepts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
