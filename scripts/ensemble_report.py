#!/usr/bin/env python3
"""Report the multi-agent ensemble (plurality-vote) baseline — SPEC R3.

The eval compares TRINITY only against single models, random routing, and the oracle
upper bound; SPEC R3 asks for the best *realizable multi-agent* baseline, which no offline
tool computes. This reports a per-question plurality vote over the pool's cached answers
(clustered by the FIXED grader), giving R3 its first offline verdict.

    # items JSON: a list of benchmark items ({benchmark, correct_answer, model_answers}),
    # i.e. the decrypted cached-answer items analysis/agreement consumes.
    python scripts/ensemble_report.py --items items.json
    python scripts/ensemble_report.py --items items.json --trinity 0.86   # render R3 verdict

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from trinity.analysis.ensemble import analyze, render


def _load_items(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("items", [])
    return [it for it in data if isinstance(it, dict)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent ensemble (plurality) baseline report (R3).")
    ap.add_argument("--items", required=True, help="JSON of cached-answer benchmark items")
    ap.add_argument("--benchmark", default=None, help="override the benchmark name for all items")
    ap.add_argument("--trinity", type=float, default=None,
                    help="TRINITY accuracy, to render the R3 verdict")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    items = _load_items(args.items)
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_bench[args.benchmark or str(it.get("benchmark") or "?")].append(it)

    reports = []
    for bench, group in sorted(by_bench.items()):
        summary = analyze(group, benchmark=bench)
        print(render(summary, trinity_accuracy=args.trinity))
        reports.append(summary.to_dict())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
