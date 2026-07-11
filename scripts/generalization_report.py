#!/usr/bin/env python3
"""Report the eval -> audit generalization (overfit) gap over run artifacts.

Pairs each ``trinity.eval`` output (``experiments/**/eval*.json``) with its sealed
``scripts/audit_eval.py`` output (``audit_result.json`` in the same run dir) and reports
the TRINITY eval->audit gap, flagged against the SAME thresholds ``scripts/pr_eval.py``
GATE 5 uses (penalty > 0.05, hard-reject > 0.10). ``results_table.py`` globs only
``eval*.json``, so this gap was never surfaced during development.

    python scripts/generalization_report.py                 # Markdown to stdout
    python scripts/generalization_report.py --root experiments --json gen.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.generalization import GeneralizationGap, analyze_pair


def _load(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def load_pairs(root: str = "experiments") -> list[GeneralizationGap]:
    """Pair each audit_result.json with an eval*.json (same run dir, else same benchmark)."""
    all_evals = sorted(glob.glob(f"{root}/**/eval*.json", recursive=True))
    gaps: list[GeneralizationGap] = []
    for ap in sorted(glob.glob(f"{root}/**/audit_result.json", recursive=True)):
        audit = _load(ap)
        if audit is None:
            continue
        adir = Path(ap).parent
        siblings = sorted(str(p) for p in adir.glob("eval*.json"))
        candidates = siblings or [e for e in all_evals
                                  if (_load(e) or {}).get("benchmark") == audit.get("benchmark")]
        eval_d = _load(candidates[0]) if candidates else None
        if eval_d is not None:
            gaps.append(analyze_pair(eval_d, audit))
    return gaps


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval->audit generalization (overfit) gap report.")
    ap.add_argument("--root", default="experiments", help="root to glob eval*.json / audit_result.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a machine-readable JSON")
    args = ap.parse_args()

    gaps = load_pairs(args.root)
    from trinity.analysis.generalization import render
    print(render(gaps))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([g.to_dict() for g in gaps], indent=2))


if __name__ == "__main__":
    main()
