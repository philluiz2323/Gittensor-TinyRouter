"""The competition's king-progression timeline from ``leaderboard.json``.

``trinity.leaderboard.summarize_competition`` reports the *current* composite king and the
score-to-beat, and ``trinity.standings`` ranks miners — but neither tells the **story** of
the competition: who held the composite crown, in what order, at what score, and how much
each new king had to gain to take it. That history is exactly what ``competition.history``
records (``pr_eval._update_leaderboard`` appends one entry per crowning, in order).

This reads that ledger and reports the reign-by-reign progression: each king's composite
score, per-benchmark breakdown, and the **gain over the previous king** (how hard-won the
crown was — the delta that had to clear ``win_margin``), plus the total gain from the first
crown to the current one and the single biggest leap. Read-only, pure stdlib — no torch,
no network. Empty (no crownings yet) on the seed leaderboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trinity.leaderboard import load_leaderboard

__all__ = [
    "Reign",
    "Timeline",
    "build_timeline",
    "load_timeline",
    "render",
]


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_float(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _clean_per_benchmark(x: Any) -> dict[str, float]:
    if not isinstance(x, Mapping):
        return {}
    out: dict[str, float] = {}
    for k, v in x.items():
        if _is_num(v):
            out[str(k)] = float(v)
    return out


@dataclass(frozen=True)
class Reign:
    """One crowning in the competition timeline."""

    order: int                      # 1-based crowning order
    miner: str
    generation: int
    pr: Any
    composite: float
    per_benchmark: dict[str, float]
    gain_over_prev: float           # composite - the previous king's composite (0-seed for the 1st)
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "order": self.order,
            "miner": self.miner,
            "generation": self.generation,
            "pr": self.pr,
            "composite": self.composite,
            "per_benchmark": dict(self.per_benchmark),
            "gain_over_prev": self.gain_over_prev,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class Timeline:
    """The full king-progression of the composite competition."""

    reigns: list[Reign]
    n_crownings: int
    current_king: str | None
    current_composite: float
    total_gain: float               # current composite - the first king's composite
    biggest_leap: Reign | None      # the crowning with the largest gain_over_prev

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_crownings": self.n_crownings,
            "current_king": self.current_king,
            "current_composite": self.current_composite,
            "total_gain": self.total_gain,
            "biggest_leap": self.biggest_leap.to_dict() if self.biggest_leap else None,
            "reigns": [r.to_dict() for r in self.reigns],
        }


_EMPTY = Timeline([], 0, None, 0.0, 0.0, None)


def build_timeline(leaderboard: Mapping[str, Any]) -> Timeline:
    """Build the reign-by-reign competition timeline from ``competition.history``.

    ``competition.history`` is appended in crowning order, so it is read in order. Each
    reign's ``gain_over_prev`` is its composite minus the previous king's composite (the
    first crown is measured against the ``0.0`` seed floor). Entries missing a numeric
    ``score`` or a ``miner`` are skipped. Returns an empty timeline when there is no
    ``competition`` record or no crownings yet.
    """
    comp = leaderboard.get("competition")
    if not isinstance(comp, Mapping):
        return _EMPTY
    raw_history = comp.get("history")
    history = raw_history if isinstance(raw_history, list) else []

    reigns: list[Reign] = []
    prev_composite = 0.0
    order = 0
    for h in history:
        if not isinstance(h, Mapping):
            continue
        miner = h.get("miner")
        composite = _as_float(h.get("score"))
        if miner is None or composite is None:
            continue
        order += 1
        try:
            gen = int(h.get("generation", 0) or 0)
        except (TypeError, ValueError):
            gen = 0
        ts = h.get("timestamp")
        reigns.append(Reign(
            order=order, miner=str(miner), generation=gen, pr=h.get("pr"),
            composite=composite, per_benchmark=_clean_per_benchmark(h.get("per_benchmark")),
            gain_over_prev=composite - prev_composite,
            timestamp=str(ts) if ts is not None else None,
        ))
        prev_composite = composite

    if not reigns:
        return _EMPTY
    biggest = max(reigns, key=lambda r: r.gain_over_prev)
    return Timeline(
        reigns=reigns,
        n_crownings=len(reigns),
        current_king=reigns[-1].miner,
        current_composite=reigns[-1].composite,
        total_gain=reigns[-1].composite - reigns[0].composite,
        biggest_leap=biggest,
    )


def load_timeline(path: Any) -> Timeline:
    """Load a leaderboard JSON and build its competition timeline."""
    return build_timeline(load_leaderboard(path))


def render(timeline: Timeline) -> str:
    """Markdown: the reign-by-reign king progression + the current-king summary."""
    t = timeline
    out = ["# Competition king progression\n"]
    if t.n_crownings == 0:
        return "".join(out) + "\n_(no one has been crowned yet)_\n"

    out.append("| # | king | composite | gain over prev | pr | when |")
    out.append("|---|---|---|---|---|---|")
    for r in t.reigns:
        when = r.timestamp or "—"
        pr = f"#{r.pr}" if r.pr is not None else "—"
        out.append(f"| {r.order} | {r.miner} (gen {r.generation}) | {r.composite:.4f} | "
                   f"{r.gain_over_prev:+.4f} | {pr} | {when} |")

    out.append(f"\n- **current king:** {t.current_king} ({t.current_composite:.4f}) "
               f"after {t.n_crownings} crowning(s)")
    out.append(f"- **total gain** (first crown → now): {t.total_gain:+.4f}")
    if t.biggest_leap is not None:
        b = t.biggest_leap
        out.append(f"- **biggest leap:** {b.miner} at crowning #{b.order} "
                   f"({b.gain_over_prev:+.4f})")
    return "\n".join(out) + "\n"
