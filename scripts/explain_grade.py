#!/usr/bin/env python3
"""Explain why a candidate answer grades correct or incorrect.

Runs the same extract / normalize / compare pipeline the grader uses and prints
each step, so you can see exactly why an answer scored what it did (which
extractor fired, the normalized forms, and how they were compared). The reported
score is taken from reward.score_text itself, so it never disagrees with the real
grade.

    python scripts/explain_grade.py --benchmark math500 \\
        --candidate 'The total is $18.90.' --reference '\\$18.90'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.grading_explain import explain_grade  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the grade explanation for one (benchmark, candidate, reference)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True,
                    help="gold answer (a JSON value is parsed for code test specs)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        reference = json.loads(args.reference)
    except (ValueError, TypeError):
        reference = args.reference

    exp = explain_grade(args.benchmark, args.candidate, reference)
    if args.json:
        print(json.dumps(exp.to_dict(), indent=2))
    else:
        verdict = "CORRECT" if exp.correct else "INCORRECT"
        print(f"[grade] {args.benchmark} -> {verdict} (score {exp.score})")
        for step in exp.steps:
            print(f"  - {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
