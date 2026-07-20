#!/usr/bin/env python3
"""Verify SPEC R9 offline: does removing each design component hurt accuracy?

Reads a JSON of ``{variant: accuracy}`` — the full model plus one entry per
ablation (removing SVF / Thinker / tri-role / penultimate-token) — and reports the
drop from full for each, whether every removal hurt (R9), and which components
matter most. Zero API cost.

Input JSON (``--accuracies``):

    {"full": 0.7044, "no_svf": 0.68, "no_thinker": 0.69,
     "no_trirole": 0.64, "last_token": 0.60}

    python scripts/ablations_report.py --accuracies r9.json

Exits non-zero when R9 is violated (some ablation did not hurt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.ablations import analyze, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object of variant -> accuracy")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R9 report; exit non-zero when R9 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracies", required=True, type=Path,
                    help="JSON of variant -> accuracy (must include the full model)")
    ap.add_argument("--full", default="full", help="key naming the full (un-ablated) model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    accs = _load(args.accuracies)
    report = analyze(accs, full=args.full)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(accs, full=args.full))
    return 0 if report["r9_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
