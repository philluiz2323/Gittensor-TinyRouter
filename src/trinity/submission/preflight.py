"""Orchestrate offline submission preflight checks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trinity.submission.gates import (
    GateResult,
    PreflightContext,
    run_offline_gates,
)
from trinity.submission.pack import SubmissionPack, load_submission_pack

__all__ = ["PreflightReport", "PreflightRunner", "load_leaderboard_json"]


@dataclass
class PreflightReport:
    """Aggregated outcome of an offline preflight run."""

    pack: SubmissionPack
    benchmark: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    @property
    def first_failure(self) -> GateResult | None:
        for result in self.results:
            if result.failed:
                return result
        return None

    def summary_lines(self) -> list[str]:
        lines = [
            f"miner={self.pack.miner} generation={self.pack.generation} benchmark={self.benchmark}",
        ]
        for result in self.results:
            status = "PASS" if result.ok else "FAIL"
            detail = "" if result.ok else f" — {result.reason}"
            lines.append(f"  [{status}] {result.gate}{detail}")
        if self.passed:
            lines.append("All offline gates passed.")
        return lines


class PreflightRunner:
    """Run the same offline gates as ``pr_eval`` on a local submission directory."""

    def __init__(
        self,
        *,
        repo_root: Path,
        benchmark: str,
        leaderboard: dict[str, Any] | None = None,
        ledger_path: str | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.benchmark = benchmark
        self.leaderboard = leaderboard if leaderboard is not None else load_leaderboard_json(repo_root)
        self.ledger_path = ledger_path
        self.submissions_root = repo_root / "submissions"

    def run(self, submission_subpath: str | Path) -> PreflightReport:
        submission_dir = self.submissions_root / str(submission_subpath)
        pack = load_submission_pack(submission_dir, submissions_root=self.submissions_root)
        if pack is None:
            import numpy as _np

            stub = SubmissionPack(
                path=submission_dir,
                miner=submission_dir.name,
                generation=0,
                head_weights=_np.array([], dtype=_np.float32),
                svf_scales=_np.array([], dtype=_np.float32),
                receipt={},
            )
            return PreflightReport(
                pack=stub,
                benchmark=self.benchmark,
                results=[GateResult("load", False, "submission_incomplete: missing weight files")],
            )

        ctx = PreflightContext(
            benchmark=self.benchmark,
            leaderboard=self.leaderboard,
            submissions_root=self.submissions_root,
            ledger_path=self.ledger_path,
        )
        # Local preflight surfaces ALL problems at once (a miner fixes their
        # submission before opening a PR); pr_eval's scoring path stays fail-fast.
        results = run_offline_gates(pack, ctx, collect_all=True)
        return PreflightReport(pack=pack, benchmark=self.benchmark, results=results)


def load_leaderboard_json(repo_root: Path) -> dict[str, Any]:
    lb_path = repo_root / "leaderboard.json"
    if not lb_path.exists():
        return {"benchmarks": {}}
    try:
        return json.loads(lb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"benchmarks": {}}
