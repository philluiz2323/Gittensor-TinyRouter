#!/usr/bin/env python3
"""Offline structural validation of the repo's run configs.

Validates ``models.yaml`` and ``trinity.yaml`` for required keys, value ranges,
uniqueness, and the cross-field invariants the code relies on (e.g.
``head.n_a == n_models + n_roles``, unique pool names/ids, ``0 < mu <=
population_size``). Exits non-zero if any problem is found, so it can gate a run
or a CI step before a GPU or a paid API call is spent on a bad config.

    python scripts/check_config.py                 # checks ./configs
    python scripts/check_config.py --configs-dir path/to/configs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.config_check import check_config_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the config report and exit non-zero on any problem."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs-dir", type=Path, default=_REPO / "configs", dest="configs_dir")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    report = check_config_dir(args.configs_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif report.ok:
        print(f"[config] OK — {args.configs_dir} passes all checks.")
    else:
        print(f"[config] {len(report.problems)} problem(s) in {args.configs_dir}:")
        for p in report.problems:
            print(f"  - {p}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
