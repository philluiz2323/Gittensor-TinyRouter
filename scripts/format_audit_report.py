#!/usr/bin/env python3
"""Offline per-model format (parse-rate) audit over a built benchmark.

Reads the cached ``model_answers`` from a benchmark JSON built by
``scripts/build_benchmark.py`` and reports, per model and per benchmark, the
fraction of answers from which the grader can extract an answer at all. This is
distinct from accuracy: a model can be right but unparseable, and that shows up
here, not in an accuracy number. Costs nothing (re-reads cached answers).

    python scripts/format_audit_report.py --benchmark-json experiments/bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.format_audit import audit_items  # noqa: E402


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
    """Print the per-model / per-benchmark parse-rate audit."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark-json", required=True, type=Path,
                    help="benchmark file produced by scripts/build_benchmark.py")
    args = ap.parse_args(argv)

    audit = audit_items(_load_items(args.benchmark_json))
    if audit.n_items == 0:
        print("[format] no cached model answers found; nothing to audit.")
        return 1

    print(json.dumps(audit.to_dict(), indent=2))
    print(f"\n[format] overall parse rate {audit.overall_parse_rate:.1%} "
          f"over {audit.n_items} questions.")
    worst = audit.worst_model()
    if worst is not None:
        s = audit.per_model[worst]
        print(f"[format] worst model: {worst} at {s.parse_rate:.1%} "
              f"({s.n_unparseable}/{s.n_answers} answers not extractable) "
              f"-- points lost to format, not reasoning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
