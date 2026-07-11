#!/usr/bin/env python3
"""Report cross-benchmark (3-benchmark union) oracle headroom from solve matrices.

Reads the per-benchmark ``oracle_matrix_<bench>.json`` files ``oracle_ceiling --collect``
/ ``agreement.to_oracle_matrix`` produce, and reports — per benchmark AND for the
equally-weighted 3-benchmark union — the best fixed single model, the routing oracle, the
disagreement rate, the estimated headroom, and the oracle RER (ROADMAP Phase 2; SPEC §6.3
R13). Nothing else consumed these matrices across benchmarks.

    python scripts/union_oracle_report.py experiments/final/oracle_matrix_*.json
    python scripts/union_oracle_report.py --root experiments --json union.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.union_oracle import render, union_oracle


def _matrix_files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/oracle_matrix_*.json", recursive=True))
    # de-dup, keep first-seen order
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-benchmark union oracle-headroom report.")
    ap.add_argument("files", nargs="*", help="oracle_matrix_<bench>.json file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/oracle_matrix_*.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="solve threshold (p >= t)")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _matrix_files(args.files, args.root)
    if not files:
        print("no oracle_matrix JSONs given (pass files or --root)")
        return
    matrices = []
    for f in files:
        try:
            matrices.append(json.loads(Path(f).read_text()))
        except Exception:
            continue
    summary = union_oracle(matrices, threshold=args.threshold)
    print(render(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
