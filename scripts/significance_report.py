#!/usr/bin/env python3
"""Paired-significance report for the R1/R2/R4 eval invariants.

Reads eval JSONs written by ``trinity.eval`` and, for each that carries a ``per_item``
block, reports the headline invariants with a **paired bootstrap CI + McNemar** verdict
instead of a bare point comparison — the rigor ``docs/RESULTS.md`` calls for ("verdict
read off bootstrap CIs, not point estimates") but the invariants never had.

    python scripts/significance_report.py path/to/eval.json      # one file
    python scripts/significance_report.py --root experiments      # all eval*.json
    python scripts/significance_report.py --root experiments --json report.json

Pure/offline: it consumes already-graded per-question correctness; no API calls, no GPU.
An older eval JSON without ``per_item`` is reported as "re-run eval to enable" rather
than skipped silently.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.significance import assess_invariants


def _eval_files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/eval*.json", recursive=True))
    return files


def render(reports: list[dict]) -> str:
    """Markdown for a list of ``{"file", "benchmark", "significance"|"missing"}`` rows."""
    out = ["# Eval invariants — paired significance\n"]
    out.append("Verdict is read off the 95% paired-bootstrap CI (never the point "
               "difference); a margin whose CI includes 0 is *inside the noise*.\n")
    for r in reports:
        out.append(f"## `{r['file']}` — {r.get('benchmark', '?')}\n")
        if r.get("missing"):
            out.append("_No `per_item` block — re-run `trinity.eval` to enable "
                       "significance (it now persists per-question correctness)._\n")
            continue
        sig = r["significance"]
        out.append(f"n = {sig['n_questions']} questions\n")
        out.append("| invariant | TRINITY | baseline | diff | 95% CI | McNemar p | verdict |")
        out.append("|---|---|---|---|---|---|---|")
        for c in sig["comparisons"]:
            lo, hi = c["ci_95"]
            mark = "✅" if (c["significant"] and c["diff"] > 0) else ("❌" if c["significant"] else "≈")
            out.append(
                f"| {c['name_a']} vs {c['name_b']} | {c['mean_a']:.3f} | {c['mean_b']:.3f} | "
                f"{c['diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {c['p_value']:.3f} | {mark} {c['verdict']} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def build_reports(files: list[str], *, n_boot: int, seed: int) -> list[dict]:
    reports: list[dict] = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        bench = d.get("benchmark", "?")
        per_item = d.get("per_item")
        if not per_item or "TRINITY" not in per_item:
            reports.append({"file": f, "benchmark": bench, "missing": True})
            continue
        sig = assess_invariants(per_item, benchmark=bench, n_boot=n_boot, seed=seed)
        reports.append({"file": f, "benchmark": bench, "significance": sig.to_dict()})
    return reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="eval JSON file(s)")
    ap.add_argument("--root", default=None, help="glob <root>/**/eval*.json too")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _eval_files(args.files, args.root)
    if not files:
        print("no eval JSONs given (pass files or --root)")
        return
    reports = build_reports(files, n_boot=args.n_boot, seed=args.seed)
    print(render(reports))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
