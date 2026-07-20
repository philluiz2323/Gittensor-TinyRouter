#!/usr/bin/env python3
"""Verify SPEC R12 offline: is TRINITY far more token-efficient than the baselines?

Reads each system's ``{accuracy, tokens_per_query}`` and compares TRINITY to every
multi-agent routing baseline (MoA / Smoothie / MasRouter) on **tokens per correct answer**
(``tokens_per_query / accuracy``) -- so a baseline cannot look efficient by answering
cheaply and wrongly. Reports each speedup and the R12 verdict. Zero API cost.

Input JSON (``--systems``):

    {"TRINITY":   {"accuracy": 0.79, "tokens_per_query": 1200},
     "MoA":       {"accuracy": 0.77, "tokens_per_query": 8400},
     "Smoothie":  {"accuracy": 0.74, "tokens_per_query": 5200},
     "MasRouter": {"accuracy": 0.78, "tokens_per_query": 6100}}

    python scripts/baseline_efficiency_report.py --systems r12_systems.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.baseline_efficiency import (  # noqa: E402
    DEFAULT_FACTOR,
    TRINITY_KEY,
    analyze,
    render,
)


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of system -> "
                         "{accuracy, tokens_per_query}")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R12 report; exit non-zero when R12 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--systems", required=True, type=Path,
                    help="JSON of system -> {accuracy, tokens_per_query}")
    ap.add_argument("--trinity-key", default=TRINITY_KEY, dest="trinity_key",
                    help=f"key naming the trained router (default {TRINITY_KEY})")
    ap.add_argument("--factor", type=float, default=DEFAULT_FACTOR,
                    help=f"speedup a baseline must be beaten by (default {DEFAULT_FACTOR:g}x)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    systems = _load(args.systems)
    report = analyze(systems, trinity_key=args.trinity_key, factor=args.factor)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(systems, trinity_key=args.trinity_key, factor=args.factor))
    return 0 if report["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
