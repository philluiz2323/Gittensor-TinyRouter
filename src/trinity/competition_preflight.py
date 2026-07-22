"""Would my submission win the composite crown — and if not, where am I short?

``trinity.leaderboard.summarize_competition`` tells a miner the composite *score-to-beat*,
but not whether their own run clears it, nor **where** it falls short. The competition is
composite: ``pr_eval`` evaluates one head on every benchmark and approves only when the
**mean** of the per-benchmark scores clears the reigning ``best_composite_score`` by
``win_margin``. A miner therefore needs to turn their own per-benchmark scores into that
same composite and compare — before spending a PR.

This does exactly that: given the loaded leaderboard and a miner's ``{benchmark: score}``,
it computes the composite the *same way ``pr_eval`` does* (mean over the competition's
declared benchmarks, a benchmark not scored counting as ``0.0``), decides go/no-go against
``king + win_margin``, and — the useful part — reports the **per-benchmark delta versus the
reigning king** so the miner sees which board is dragging their composite down. Read-only,
pure stdlib — no torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trinity.leaderboard import load_leaderboard, summarize_competition

__all__ = [
    "PreflightResult",
    "preflight_submission",
    "load_and_preflight",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _clean_scores(x: Any) -> dict[str, float]:
    if not isinstance(x, Mapping):
        return {}
    out: dict[str, float] = {}
    for k, v in x.items():
        if _is_num(v):
            out[str(k)] = float(v)
    return out


@dataclass(frozen=True)
class PreflightResult:
    """Whether a candidate's per-benchmark scores would win the composite crown."""

    composite: float                    # mean over the scored benchmarks (pr_eval's rule)
    score_to_beat: float                # king composite + win_margin
    would_win: bool                     # composite >= score_to_beat (within tol)
    gap: float                          # score_to_beat - composite (>0 = shortfall)
    benchmarks: list[str]               # the competition's declared benchmarks
    my_scores: dict[str, float]         # candidate's per-benchmark scores (aligned, missing -> 0)
    missing_benchmarks: list[str]       # declared benchmarks the candidate did not score
    vs_king: dict[str, float]           # my score - king's score, per benchmark (None king -> my)
    weakest_benchmark: str | None       # the candidate's lowest-scoring declared benchmark
    king_miner: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "composite": self.composite,
            "score_to_beat": self.score_to_beat,
            "would_win": self.would_win,
            "gap": self.gap,
            "benchmarks": list(self.benchmarks),
            "my_scores": dict(self.my_scores),
            "missing_benchmarks": list(self.missing_benchmarks),
            "vs_king": dict(self.vs_king),
            "weakest_benchmark": self.weakest_benchmark,
            "king_miner": self.king_miner,
        }


def preflight_submission(
    leaderboard: Mapping[str, Any], my_scores: Mapping[str, float],
) -> PreflightResult | None:
    """Would ``my_scores`` win the composite crown? Returns None if no competition record.

    ``my_scores`` is ``{benchmark: accuracy}``. The composite is the mean over the
    competition's **declared** benchmarks (a benchmark absent from ``my_scores`` counts as
    ``0.0``, exactly as an unscored board would in ``pr_eval``); go/no-go is
    ``composite >= best_composite_score + win_margin``.
    """
    ct = summarize_competition(leaderboard)
    if ct is None:
        return None
    scores = _clean_scores(my_scores)
    # Align to the declared benchmarks; fall back to the candidate's own boards when the
    # competition declares none (older record) so the composite is still meaningful.
    benchmarks = list(ct.benchmarks) if ct.benchmarks else sorted(scores)
    aligned = {b: scores.get(b, 0.0) for b in benchmarks}
    missing = [b for b in benchmarks if b not in scores]

    composite = sum(aligned.values()) / len(aligned) if aligned else 0.0
    gap = ct.score_to_beat - composite
    king_pb = ct.best_per_benchmark
    vs_king = {b: aligned[b] - king_pb.get(b, 0.0) for b in benchmarks}
    weakest = min(benchmarks, key=lambda b: aligned[b]) if benchmarks else None
    return PreflightResult(
        composite=composite,
        score_to_beat=ct.score_to_beat,
        would_win=composite + _TOL >= ct.score_to_beat,
        gap=gap,
        benchmarks=benchmarks,
        my_scores=aligned,
        missing_benchmarks=missing,
        vs_king=vs_king,
        weakest_benchmark=weakest,
        king_miner=ct.king_miner,
    )


def load_and_preflight(path: Any, my_scores: Mapping[str, float]) -> PreflightResult | None:
    """Load a leaderboard JSON and run :func:`preflight_submission`."""
    return preflight_submission(load_leaderboard(path), my_scores)


def render(result: PreflightResult | None) -> str:
    """A compact go/no-go report with the per-benchmark deltas versus the king."""
    if result is None:
        return "# Submission preflight\n\n_(no competition record to compare against)_\n"
    r = result
    verdict = "WOULD WIN ✅" if r.would_win else "would NOT win ❌"
    out = [
        "# Submission preflight\n",
        f"- composite {r.composite:.4f} vs score-to-beat {r.score_to_beat:.4f} "
        f"({'king ' + r.king_miner if r.king_miner else 'no king yet'}): **{verdict}**",
    ]
    if not r.would_win:
        out.append(f"- shortfall: {r.gap:+.4f} to close")
    if r.missing_benchmarks:
        out.append(f"- ⚠ not scored (counted as 0.0): {', '.join(r.missing_benchmarks)}")
    out.append("\n| benchmark | my score | vs king |")
    out.append("|---|---|---|")
    for b in r.benchmarks:
        out.append(f"| {b} | {r.my_scores[b]:.4f} | {r.vs_king[b]:+.4f} |")
    if r.weakest_benchmark is not None:
        out.append(f"\n- weakest board (improve here first): **{r.weakest_benchmark}** "
                   f"({r.my_scores[r.weakest_benchmark]:.4f})")
    return "\n".join(out) + "\n"
