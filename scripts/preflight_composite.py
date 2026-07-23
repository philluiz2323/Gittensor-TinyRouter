#!/usr/bin/env python3
"""Would my per-benchmark scores win the composite crown? A pre-submission go/no-go.

Computes the composite (mean over the competition's benchmarks, the way pr_eval does),
compares it to the reigning king + win_margin, and reports the per-benchmark delta versus
the king so you can see which board to improve. Zero API cost.

    python scripts/preflight_composite.py --score math500=0.9 --score mmlu=0.9 --score livecodebench=0.75
    python scripts/preflight_composite.py --scores my_scores.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.competition_preflight import load_and_preflight, render  # noqa: E402


def _parse_scores(pairs: list[str] | None, scores_json: Path | None) -> dict[str, float]:
    scores: dict[str, float] = {}
    if scores_json is not None:
        data = json.loads(scores_json.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{scores_json}: expected a JSON object of benchmark -> score")
        scores.update({str(k): float(v) for k, v in data.items()})
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--score expects benchmark=value, got {pair!r}")
        k, v = pair.split("=", 1)
        scores[k.strip()] = float(v)
    return scores


def main(argv: list[str] | None = None) -> int:
    """Print the preflight go/no-go; exit non-zero when the submission would not win."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaderboard", type=Path, default=_REPO / "leaderboard.json")
    ap.add_argument("--score", action="append", metavar="BENCH=VALUE",
                    help="a per-benchmark score, repeatable")
    ap.add_argument("--scores", type=Path, help="JSON object of {benchmark: score}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    scores = _parse_scores(args.score, args.scores)
    if not scores:
        ap.error("provide at least one --score BENCH=VALUE or --scores FILE")
    result = load_and_preflight(args.leaderboard, scores)
    if args.json:
        print(json.dumps(result.to_dict() if result else None, indent=2))
    else:
        print(render(result), end="")
    return 0 if (result is not None and result.would_win) else 1


if __name__ == "__main__":
    raise SystemExit(main())
