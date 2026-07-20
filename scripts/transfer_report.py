#!/usr/bin/env python3
"""Zero-shot transfer report: in-distribution (SPEC §6.1) vs held-out (SPEC §6.2).

``docs/SPEC.md`` §6.2 evaluates the trained coordinator on benchmarks it never trained on
(AIME2025, BigCodeBench, MT-Bench, GPQA-Diamond) and Table 1 claims the held-out average
still beats the best single model. Nothing in the repo encoded that split, so every report
averaged all benchmarks together — a router that wins in-distribution and collapses off it
read as a clean win everywhere.

This reads the same ``experiments/**/eval*.json`` files ``scripts/results_table.py``
consumes, partitions them into the two SPEC cohorts, and reports the TRINITY-vs-best-single
margin in each plus the transfer gap between them. Per-benchmark rows are always shown, so
a cohort mean cannot hide a held-out task that lost.

    python scripts/transfer_report.py --root experiments
    python scripts/transfer_report.py eval_math500.json eval_gpqa.json --json transfer.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from trinity.analysis.transfer import assess, render


def _eval_files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/eval*.json", recursive=True))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load_rows(files: list[str]) -> list[dict[str, Any]]:
    """``{benchmark, trinity, best_single, best_model}`` per eval JSON.

    Mirrors ``results_table.load_rows``: ``best_single`` is the highest NON-NULL
    ``single::<model>`` score, so a partially-written eval (``"single::x": null``) cannot
    crash the comparison or be read as a zero.
    """
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            d = json.loads(Path(path).read_text())
        except Exception:
            continue
        results = d.get("results") or {}
        if not isinstance(results, dict) or "TRINITY" not in results:
            continue
        singles = {k.split("::", 1)[1]: v for k, v in results.items()
                   if k.startswith("single::")}
        numeric = {m: v for m, v in singles.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not numeric:
            continue
        best_model = max(numeric, key=lambda m: numeric[m])
        rows.append({
            "benchmark": d.get("benchmark", ""),
            "trinity": results.get("TRINITY"),
            "best_single": numeric[best_model],
            "best_model": best_model,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="In-distribution vs held-out transfer report.")
    ap.add_argument("files", nargs="*", help="eval JSON file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/eval*.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _eval_files(args.files, args.root)
    if not files:
        print("no eval JSONs given (pass files or --root)")
        return
    rows = load_rows(files)
    if not rows:
        print("no eval JSON carried a TRINITY score plus a numeric single:: baseline")
        return

    summary = assess(rows)
    print(render(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
