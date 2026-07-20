"""Cross-benchmark competition standings from ``leaderboard.json``.

``trinity.leaderboard`` answers "what score must I beat on THIS benchmark" and
``trinity.submission.leaderboard`` checks integrity + prints the per-benchmark frontier —
both strictly **per-benchmark**. Neither aggregates a miner's results **across** benchmarks
into an overall ranking, so there is no answer to "who is winning the competition overall?"

This module computes that. The competition is **composite** (``pr_eval`` evaluates one
head on every benchmark and crowns a single king on the mean score), so every merged win
is written to ``competition.history`` with a ``per_benchmark`` score breakdown — the
per-benchmark ``benchmarks.*`` subtree only carries static baselines/ceilings and the
rate-limit ledger, and its ``history`` is **never** populated. For each miner this takes
their best **merged** score on each benchmark from ``competition.history[*].per_benchmark``,
then ranks miners by the **equal-weighted** mean over all benchmarks — each benchmark counts
once and a benchmark a miner never scored counts as ``0.0``, so an overall leader must be
strong on *every* benchmark, not just the top one. (Same equal-weight-per-benchmark
philosophy the union-oracle and results-table summaries use, so a specialist cannot outrank
a generalist by cherry-picking a single board.)

It reuses :func:`trinity.leaderboard.load_leaderboard` so it can't drift from the schema,
tolerates a partially-tampered record (a non-dict ``competition``, a non-list ``history``,
a non-dict entry, a non-dict ``per_benchmark``) without crashing, and degrades to an empty
ranking on the seed leaderboard (zero miners). Read-only, pure stdlib — no torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, TypeGuard

from trinity.leaderboard import load_leaderboard

__all__ = [
    "MinerStanding",
    "Standings",
    "compute_standings",
    "load_standings",
    "render",
]


_TOL = 1e-9


def _is_num(x: Any) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


@dataclass(frozen=True)
class MinerStanding:
    """One miner's cross-benchmark result."""

    miner: str
    per_benchmark: dict[str, float]   # best merged score per benchmark entered
    overall: float                    # equal-weighted mean over ALL benchmarks (missing == 0)
    n_competed: int                   # benchmarks with at least one merged win
    benchmarks_led: int               # benchmarks where this miner is the reigning king
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "miner": self.miner,
            "rank": self.rank,
            "overall": self.overall,
            "n_competed": self.n_competed,
            "benchmarks_led": self.benchmarks_led,
            "per_benchmark": dict(self.per_benchmark),
        }


@dataclass(frozen=True)
class Standings:
    """The overall competition ranking across all benchmarks."""

    benchmarks: list[str]
    miners: list[MinerStanding] = field(default_factory=list)   # sorted by rank

    @property
    def leader(self) -> str | None:
        """The rank-1 miner, or None when no one has competed yet."""
        return self.miners[0].miner if self.miners else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmarks": list(self.benchmarks),
            "n_miners": len(self.miners),
            "leader": self.leader,
            "miners": [m.to_dict() for m in self.miners],
        }


def compute_standings(leaderboard: Mapping[str, Any]) -> Standings:
    """Rank miners across benchmarks from the composite ``competition.history``.

    A miner's per-benchmark score is the max over their ``merged`` wins'
    ``per_benchmark`` breakdowns; a benchmark they never scored counts as ``0.0`` in the
    equal-weighted overall. ``benchmarks_led`` is the number of boards on which the miner
    *uniquely* holds the top score. Ranking is by ``overall`` desc, then benchmarks-led
    desc, then miner name.
    """
    comp = leaderboard.get("competition")
    if not isinstance(comp, dict):
        return Standings([], [])

    per_miner: dict[str, dict[str, float]] = {}
    seen_benches: set[str] = set()
    for h in _as_list(comp.get("history")):
        if not isinstance(h, dict) or h.get("merged") is not True:
            continue
        miner = h.get("miner")
        per_benchmark = h.get("per_benchmark")
        if miner is None or not isinstance(per_benchmark, dict):
            continue
        best = per_miner.setdefault(str(miner), {})
        for bench, score in per_benchmark.items():
            if not _is_num(score):
                continue
            b = str(bench)
            seen_benches.add(b)
            if b not in best or float(score) > best[b]:
                best[b] = float(score)

    # Benchmark set: the competition's declared benchmarks (so a board no one has scored
    # still appears as a column), unioned with any seen in history.
    declared = comp.get("benchmarks")
    if isinstance(declared, list):
        seen_benches |= {str(b) for b in declared}
    bench_names = sorted(seen_benches)

    # Per-board leader: the miner who UNIQUELY holds the top score (a tie credits no one).
    bench_leader: dict[str, str] = {}
    for b in bench_names:
        holders = [(sc[b], m) for m, sc in per_miner.items() if b in sc]
        if not holders:
            continue
        top = max(s for s, _ in holders)
        tops = [m for s, m in holders if abs(s - top) <= _TOL]
        if len(tops) == 1:
            bench_leader[b] = tops[0]

    n = len(bench_names)
    standings = []
    for miner, per in per_miner.items():
        overall = sum(per.get(b, 0.0) for b in bench_names) / n if n else 0.0
        led = sum(1 for b in bench_names if bench_leader.get(b) == miner)
        standings.append(MinerStanding(miner, dict(per), overall, len(per), led))

    standings.sort(key=lambda s: (-s.overall, -s.benchmarks_led, s.miner))
    standings = [replace(s, rank=i + 1) for i, s in enumerate(standings)]
    return Standings(bench_names, standings)


def load_standings(path: str | Path) -> Standings:
    """Load a leaderboard JSON and compute the cross-benchmark standings."""
    return compute_standings(load_leaderboard(path))


def render(standings: Standings) -> str:
    """Markdown: the overall ranking plus each miner's per-benchmark scores."""
    s = standings
    out = ["# Overall competition standings (equal-weighted across benchmarks)\n"]
    if not s.benchmarks:
        return "".join(out) + "\n_(no benchmarks)_\n"
    if not s.miners:
        return ("".join(out) + f"\nbenchmarks: {', '.join(s.benchmarks)}\n\n"
                "_(no miners have won a benchmark yet)_\n")
    header = " | ".join(s.benchmarks)
    out.append(f"| rank | miner | overall | led | {header} |")
    out.append("|---|---|---|---|" + "---|" * len(s.benchmarks))
    for m in s.miners:
        cells = " | ".join(f"{m.per_benchmark[b]:.3f}" if b in m.per_benchmark else "—"
                           for b in s.benchmarks)
        out.append(f"| {m.rank} | {m.miner} | {m.overall:.3f} | {m.benchmarks_led} | {cells} |")
    out.append(f"\n- **overall leader:** {s.leader}")
    out.append("- overall = equal-weighted mean over all benchmarks (a benchmark not "
               "entered counts as 0.000).")
    return "\n".join(out) + "\n"
