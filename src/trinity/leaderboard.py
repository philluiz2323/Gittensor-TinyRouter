"""Read the competition leaderboard: what score must a submission beat?

Why this exists
---------------
``leaderboard.json`` records, per benchmark, the current best score, the reigning
king, and the reference points the maintainer publishes (``baseline_random``,
``best_single_model``, ``oracle_ceiling``). ``scripts/pr_eval.py`` reads it
internally to compute novelty, but a *miner* has no tool that answers the two
questions they actually care about before spending on a training run:

* **what score do I have to beat** to become king on this benchmark, and
* **is it even reachable** — how much routing headroom is left between the best
  single model and the any-model oracle, and how much of it the current king has
  already captured.

This module turns the leaderboard into those targets. It is read-only and pure;
it never touches the scoring path.

Pure / deterministic / no network / no GPU / no torch. Only the stdlib.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["BenchmarkTarget", "load_leaderboard", "summarize_targets"]


def _as_float(x: Any, default: float = 0.0) -> float:
    """Coerce a leaderboard number to float, tolerating strings and ``None``."""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BenchmarkTarget:
    """What a submission must beat on one benchmark, and how much room is left.

    ``score_to_beat`` is ``best_score`` — a submission becomes king only by scoring
    strictly above it. ``headroom`` is ``oracle_ceiling - best_single_model``: the
    accuracy a perfect router could add over the strongest single model.
    ``captured`` is how much of that headroom the current best already realized;
    ``remaining`` is what is still on the table.
    """

    benchmark: str
    score_to_beat: float
    best_single_model: float
    oracle_ceiling: float
    baseline_random: float
    king_miner: str | None
    king_generation: int
    king_pr: Any

    @property
    def has_king(self) -> bool:
        """True iff someone already holds this benchmark."""
        return self.king_miner is not None

    @property
    def headroom(self) -> float:
        """Total routing headroom: ``oracle_ceiling - best_single_model`` (>= 0)."""
        return max(0.0, self.oracle_ceiling - self.best_single_model)

    @property
    def captured(self) -> float:
        """How far the best score is above the single-model floor (>= 0)."""
        return max(0.0, self.score_to_beat - self.best_single_model)

    @property
    def remaining(self) -> float:
        """Headroom still unclaimed above the current best: ``oracle - best`` (>= 0)."""
        return max(0.0, self.oracle_ceiling - self.score_to_beat)

    @property
    def contested(self) -> bool:
        """True iff there is measurable headroom left to win (``remaining > 0``)."""
        return self.remaining > 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "score_to_beat": self.score_to_beat,
            "best_single_model": self.best_single_model,
            "oracle_ceiling": self.oracle_ceiling,
            "baseline_random": self.baseline_random,
            "king_miner": self.king_miner,
            "king_generation": self.king_generation,
            "king_pr": self.king_pr,
            "has_king": self.has_king,
            "headroom": self.headroom,
            "captured": self.captured,
            "remaining": self.remaining,
            "contested": self.contested,
        }


def load_leaderboard(path: str | Path) -> dict[str, Any]:
    """Load a leaderboard JSON, returning an empty ``{"benchmarks": {}}`` on failure.

    Best-effort: a missing or unparseable file yields the empty shape rather than
    raising, so a caller can report "no leaderboard yet" without special-casing.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"benchmarks": {}}
    if not isinstance(data, dict) or not isinstance(data.get("benchmarks"), dict):
        return {"benchmarks": {}}
    return data


def _target_from_entry(benchmark: str, entry: Mapping[str, Any]) -> BenchmarkTarget:
    king = entry.get("best_miner")
    try:
        king_gen = int(entry.get("best_generation", 0) or 0)
    except (TypeError, ValueError):
        king_gen = 0
    return BenchmarkTarget(
        benchmark=benchmark,
        score_to_beat=_as_float(entry.get("best_score")),
        best_single_model=_as_float(entry.get("best_single_model")),
        oracle_ceiling=_as_float(entry.get("oracle_ceiling")),
        baseline_random=_as_float(entry.get("baseline_random")),
        king_miner=str(king) if king else None,
        king_generation=king_gen,
        king_pr=entry.get("best_pr"),
    )


def summarize_targets(leaderboard: Mapping[str, Any]) -> list[BenchmarkTarget]:
    """Turn a loaded leaderboard into per-benchmark :class:`BenchmarkTarget`\\ s.

    Benchmarks are returned in sorted name order. A malformed per-benchmark entry
    (not a mapping) is skipped rather than crashing the whole summary.
    """
    benches = leaderboard.get("benchmarks", {})
    if not isinstance(benches, Mapping):
        return []
    out: list[BenchmarkTarget] = []
    for name in sorted(benches):
        entry = benches[name]
        if isinstance(entry, Mapping):
            out.append(_target_from_entry(str(name), entry))
    return out
