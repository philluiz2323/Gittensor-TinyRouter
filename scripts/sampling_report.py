#!/usr/bin/env python3
"""Report per-model sampling stability (pass@1 / pass@K / majority@K) from oracle matrices.

Reads the ``oracle_matrix_<bench>.json`` files ``oracle_ceiling --collect`` produces and
reports, per model, the within-model K-sample metrics — pass@1, pass@K, self-consistency
majority@K, and the gain — plus whether the best single model's majority@K rivals the
routing oracle. This is the within-model axis every other oracle-matrix consumer
(complementarity, oracle_ceiling) collapses away.

    python scripts/sampling_report.py experiments/final/oracle_matrix_*.json
    python scripts/sampling_report.py --root experiments --json sampling.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.sampling import analyze, render


def _files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/oracle_matrix_*.json", recursive=True))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-model sampling-stability report.")
    ap.add_argument("files", nargs="*", help="oracle_matrix_<bench>.json file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/oracle_matrix_*.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _files(args.files, args.root)
    if not files:
        print("no oracle_matrix JSONs given (pass files or --root)")
        return
    reports = []
    for f in files:
        try:
            matrix = json.loads(Path(f).read_text())
        except Exception:
            continue
        summary = analyze(matrix)
        print(render(summary))
        reports.append(summary.to_dict())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
