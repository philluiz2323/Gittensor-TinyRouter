#!/usr/bin/env python3
"""Offline data-quality audit of a built benchmark.

Reports, per benchmark, the data-quality defects that silently corrupt every
downstream number: duplicate ``question_id`` (collides the per-query maps),
duplicate question text (double-weights a question), and items missing a prompt
or a reference answer. Exits non-zero if any defect is found, so it can gate a
paid eval on a broken set.

    python scripts/dataset_quality_report.py --benchmark-json experiments/bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.dataset_quality import audit_dataset  # noqa: E402


def _load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return list(data)
    for key in ("items", "questions", "tasks"):
        value = data.get(key)
        if isinstance(value, list):
            return list(value)
    raise ValueError(f"{path}: expected a list of items or an 'items' list")


def main(argv: list[str] | None = None) -> int:
    """Print the data-quality report; exit non-zero on any defect."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark-json", required=True, type=Path)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    report = audit_dataset(_load_items(args.benchmark_json))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif report.ok:
        print(f"[dataset] OK — {report.n_items} items, no data-quality defects.")
    else:
        print(f"[dataset] {report.n_problems} defect(s) across {report.n_items} items:")
        for name, q in sorted(report.per_benchmark.items()):
            if q.ok:
                continue
            print(f"  {name}: "
                  f"{len(q.duplicate_ids)} dup id(s), {q.duplicate_questions} dup question(s), "
                  f"{q.missing_prompt} missing prompt(s), {q.missing_answer} missing answer(s)")
            if q.duplicate_ids:
                print(f"    duplicate ids: {q.duplicate_ids}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
