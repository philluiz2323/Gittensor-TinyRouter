#!/usr/bin/env python3
"""Offline novelty + routing-diversity report for a head's decisions.

Novelty is 5% of the competition score ("different routing choices from other
miners"), defined by the hidden scorer as ``1 - agreement_rate`` against the
current king. Feed this the per-question decisions you already have.

Input JSON (``--decisions``): a head's decisions, and optionally a reference
head's, as aligned lists. Each decision is ``[agent, role]`` (or any value):

    {"head": [[0, "WORKER"], [1, "WORKER"]],
     "reference": [[0, "WORKER"], [0, "WORKER"]]}

    python scripts/novelty_report.py --decisions head_vs_king.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.novelty import novelty_report, selection_diversity  # noqa: E402


def _as_decisions(rows):
    # JSON lists become tuples so they hash and compare like real decisions.
    return [tuple(r) if isinstance(r, list) else r for r in rows]


def main(argv: list[str] | None = None) -> int:
    """Print novelty vs a reference head plus the head's own routing diversity."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decisions", required=True, type=Path,
                    help="JSON with a 'head' list and optional 'reference' list")
    args = ap.parse_args(argv)

    data = json.loads(args.decisions.read_text())
    head = _as_decisions(data["head"])
    reference = _as_decisions(data["reference"]) if data.get("reference") is not None else None

    report = novelty_report(head, reference)
    diversity = selection_diversity(head)

    print(json.dumps({"novelty": report.to_dict(), "diversity": diversity.to_dict()}, indent=2))
    if reference is None:
        print(f"\n[novelty] no reference head -> neutral novelty {report.novelty:.4f}")
    else:
        print(f"\n[novelty] {report.novelty:.4f} "
              f"({report.n_questions - report.n_agree}/{report.n_questions} questions differ)")
    print(f"[diversity] {diversity.n_distinct} distinct choices; "
          f"top choice {diversity.top_choice} at {diversity.top_share:.0%}; "
          f"normalized entropy {diversity.normalized_entropy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
