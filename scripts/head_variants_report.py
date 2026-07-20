#!/usr/bin/env python3
"""Verify SPEC R10 offline: is the linear head at least as good overall as every variant?

Reads a JSON head-variant score table -- ``{variant: {benchmark: score}}`` plus optional
parameter counts -- and reports each variant's equal-weight overall (over the shared
benchmark set), the best-overall variant, the per-benchmark exceptions where a non-linear
variant beats linear, and the R10 verdict. Zero API cost.

Input JSON (``--scores``):

    {"linear_key": "linear",
     "scores": {"linear":  {"lcb": 0.615, "math500": 0.880, "mmlu": 0.916, "rlpr": 0.401},
                "sparse":  {"lcb": 0.600, "math500": 0.870, "mmlu": 0.917, "rlpr": 0.395}},
     "params": {"linear": 40960, "sparse": 8192}}

    python scripts/head_variants_report.py --scores head_variants.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.head_variants import analyze_heads, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "scores" not in data:
        raise ValueError(f"{path}: expected an object with a 'scores' table")
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R10 report; exit non-zero when R10 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", required=True, type=Path,
                    help="JSON with a 'scores' {variant: {benchmark: score}} table")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    data = _load(args.scores)
    scores = data["scores"]
    params = data.get("params")
    linear_key = str(data.get("linear_key", "linear"))
    s = analyze_heads(scores, params=params, linear_key=linear_key)
    if args.json:
        print(json.dumps(s.to_dict(), indent=2))
    else:
        print(render(scores, params=params, linear_key=linear_key), end="")
    return 0 if s.linear_is_best else 1


if __name__ == "__main__":
    raise SystemExit(main())
