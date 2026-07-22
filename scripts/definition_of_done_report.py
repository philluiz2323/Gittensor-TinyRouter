#!/usr/bin/env python3
"""Answer the SPEC's own question: is the replication DONE?

``docs/SPEC.md`` defines done as *"R1–R4 and R8 hold on at least 2 of our chosen
in-distribution tasks; the trained coordinator runs end-to-end within the atomic-eval
budget on one H200; the optimizer drives J(θ) upward over iterations."* Every ingredient
existed but nothing computed that sentence — ``significance`` is per-eval-file,
``convergence`` covers only the optimizer clause, and ``results_table``'s multi-task
summary uses an *average*-based rule rather than "holds on at least 2 tasks".

This joins them: per-task R1–R4/R8 from the artifacts already on disk, the ≥2-task rule,
and one combined PASS / NOT MET. It reuses the canonical verdicts rather than re-deriving
them (R1/R2/R4 via ``significance.assess_invariants``, R8 + the J(θ) clause via
``convergence.analyze_runs``).

    # eval JSONs give R1/R2/R4; <root>/**/summary.json give R8 + the J(θ) clause
    python scripts/definition_of_done_report.py --root experiments
    python scripts/definition_of_done_report.py eval_math500.json eval_mmlu.json --root experiments

    # R3 (vs the plurality multi-agent baseline) needs ensemble accuracies:
    python scripts/ensemble_report.py items.json --json ens.json
    python scripts/definition_of_done_report.py --root experiments --ensemble ens.json

Evidence that is absent is reported as "not measured" and never counted as a pass.
Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from trinity.analysis.convergence import RunConvergence, analyze_run, analyze_runs
from trinity.analysis.definition_of_done import DOD_MIN_TASKS, assess, assess_task, render


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


def load_runs(root: str | None) -> list[RunConvergence]:
    """Every run under ``root`` with a summary.json (+ sibling history.json)."""
    runs: list[RunConvergence] = []
    if not root:
        return runs
    for sp in sorted(glob.glob(f"{root}/**/summary.json", recursive=True)):
        summary_path = Path(sp)
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        history: Any = []
        history_path = summary_path.with_name("history.json")
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except Exception:
                history = []
        runs.append(analyze_run(summary, history, run_id=summary_path.parent.name))
    return runs


def load_ensemble_accuracy(path: str | None) -> dict[str, float]:
    """``{benchmark: ensemble_accuracy}`` from an ensemble report JSON.

    Accepts a bare ``{bench: acc}`` mapping, a single ensemble summary, or a list of them
    (``scripts/ensemble_report.py --json`` output), so R3 can be supplied however it was
    produced. Anything unparseable yields no entries — R3 then reports "not measured".
    """
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text())
    except Exception:
        return {}

    out: dict[str, float] = {}
    rows = raw if isinstance(raw, list) else [raw]
    if isinstance(raw, dict) and "benchmark" not in raw:
        for k, v in raw.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
            elif isinstance(v, dict) and "ensemble_accuracy" in v:
                out[str(k)] = float(v["ensemble_accuracy"])
        if out:
            return out
    for row in rows:
        if isinstance(row, dict) and "benchmark" in row and "ensemble_accuracy" in row:
            out[str(row["benchmark"])] = float(row["ensemble_accuracy"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SPEC definition-of-done roll-up.")
    ap.add_argument("files", nargs="*", help="eval JSON file(s) carrying a per_item block")
    ap.add_argument("--root", default=None,
                    help="glob <root>/**/eval*.json and <root>/**/summary.json")
    ap.add_argument("--ensemble", default=None,
                    help="ensemble report JSON supplying R3 (benchmark -> ensemble_accuracy)")
    ap.add_argument("--min-tasks", type=int, default=DOD_MIN_TASKS, dest="min_tasks",
                    help=f"tasks each invariant must hold on (SPEC: {DOD_MIN_TASKS})")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _eval_files(args.files, args.root)
    if not files:
        print("no eval JSONs given (pass files or --root)")
        return

    runs = load_runs(args.root)
    cross = analyze_runs(runs) if runs else {}
    rankings = cross.get("rankings", {}) if isinstance(cross, dict) else {}
    drives_J = cross.get("dod_drives_J_upward") if isinstance(cross, dict) else None
    ens = load_ensemble_accuracy(args.ensemble)

    tasks = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        per_item = d.get("per_item")
        if not per_item or "TRINITY" not in per_item:
            continue                       # no per-question data -> nothing to assess
        bench = str(d.get("benchmark", "?"))
        tasks.append(assess_task(
            bench, per_item,
            ensemble_accuracy=ens.get(bench),
            r8_ranking=rankings.get(bench),
            n_boot=args.n_boot, seed=args.seed,
        ))

    if not tasks:
        print("no eval JSON carried a per_item block — re-run trinity.eval to enable")
        return

    verdict = assess(tasks, drives_J_upward=drives_J, min_tasks=args.min_tasks)
    print(render(verdict))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(verdict.to_dict(), indent=2))


if __name__ == "__main__":
    main()
