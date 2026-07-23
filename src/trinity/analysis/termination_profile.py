"""Offline termination-profile diagnostic: does the accept/revise loop converge?

``trinity.efficiency`` collapses turn usage into one scalar for the competition
score (``(max_turns - avg_turns)/(max_turns-1) * live_acc``). That number hides *how*
the multi-turn loop terminates. The loop only earns its keep when the Verifier
actually ACCEPTs and stops early (SPEC §0.3.5 termination rule ``tau = min{k : R_k =
Verifier AND ACCEPT}``); a head whose Verifier never accepts just burns the full
``max_turns`` budget on every question — a broken loop the scalar cannot reveal.

This reads a run's per-trajectory termination records — how many turns each
trajectory took and whether it stopped on a Verifier ACCEPT — and reports, per
benchmark and for the pooled union:

* the **accept rate** (fraction that stopped on ACCEPT rather than exhausting the
  budget) and the **exhausted rate** (fraction that ran to ``max_turns``),
* the mean / median turns taken and the full **turn histogram**, and
* a **never_accepts** flag when the accept rate is 0 — the loop is inert and every
  question paid the full budget.

Pure stdlib over plain records -- no torch, no network, no GPU.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable, Mapping

__all__ = [
    "TerminationProfile",
    "analyze",
    "analyze_benchmarks",
    "render",
]

#: Terminated-by values that count as a Verifier ACCEPT (early stop).
_ACCEPT_MARKERS = frozenset({"accept", "accepted", "verifier_accept", "true", "1"})


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _record(rec: Any) -> tuple[int, bool] | None:
    """Coerce one trajectory record to ``(turns, accepted)``.

    Accepts a mapping with ``turns`` (or ``n_turns`` / ``turns_used``) and
    ``terminated_by`` / ``accepted`` / ``accept``; or a ``(turns, accepted)`` pair.
    ``accepted`` may be a bool or a ``terminated_by`` string (``"accept"`` => True,
    anything else — ``None``, ``"max_turns"`` — => False). Returns ``None`` for an
    unusable record (no positive integer turn count).
    """
    turns: Any
    acc: Any = None
    if isinstance(rec, Mapping):
        turns = rec.get("turns", rec.get("n_turns", rec.get("turns_used")))
        acc = rec.get("terminated_by", rec.get("accepted", rec.get("accept")))
    elif isinstance(rec, (tuple, list)) and len(rec) >= 1:
        turns = rec[0]
        acc = rec[1] if len(rec) > 1 else None
    else:
        return None
    if not _is_int(turns) or turns <= 0:
        return None
    if isinstance(acc, bool):
        accepted = acc
    elif acc is None:
        accepted = False
    else:
        accepted = str(acc).strip().lower() in _ACCEPT_MARKERS
    return int(turns), accepted


@dataclass(frozen=True)
class TerminationProfile:
    """How a run's trajectories terminated on one benchmark."""

    benchmark: str
    n_trajectories: int
    accept_rate: float
    exhausted_rate: float
    mean_turns: float
    median_turns: float
    max_turns_observed: int
    turn_histogram: dict[int, int]
    never_accepts: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_trajectories": self.n_trajectories,
            "accept_rate": self.accept_rate,
            "exhausted_rate": self.exhausted_rate,
            "mean_turns": self.mean_turns,
            "median_turns": self.median_turns,
            "max_turns_observed": self.max_turns_observed,
            # JSON object keys must be strings.
            "turn_histogram": {str(k): v for k, v in sorted(self.turn_histogram.items())},
            "never_accepts": self.never_accepts,
        }


def analyze(records: Iterable[Any], *, benchmark: str = "?") -> TerminationProfile:
    """Compute the termination profile of one benchmark's trajectories.

    Args:
        records: per-trajectory ``(turns, accepted)`` records (see :func:`_record`
            for accepted shapes). A trajectory is "exhausted" when it did not stop on
            a Verifier ACCEPT, i.e. it ran to the turn budget.
        benchmark: Name for the report row.

    Returns:
        A :class:`TerminationProfile`. An empty / all-unusable input yields a
        zeroed profile with ``never_accepts=False`` (nothing ran, so nothing to flag).
    """
    parsed = [p for p in (_record(r) for r in records) if p is not None]
    n = len(parsed)
    if n == 0:
        return TerminationProfile(benchmark, 0, 0.0, 0.0, 0.0, 0.0, 0, {}, False)
    turns = [t for t, _ in parsed]
    n_accept = sum(1 for _, a in parsed if a)
    hist = dict(Counter(turns))
    return TerminationProfile(
        benchmark=benchmark,
        n_trajectories=n,
        accept_rate=n_accept / n,
        exhausted_rate=(n - n_accept) / n,
        mean_turns=float(mean(turns)),
        median_turns=float(median(turns)),
        max_turns_observed=max(turns),
        turn_histogram=hist,
        never_accepts=n_accept == 0,
    )


def analyze_benchmarks(per_benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Per-benchmark termination profiles plus the pooled-union profile.

    Args:
        per_benchmark: ``{benchmark: records}``.

    Returns:
        ``{"per_benchmark": [TerminationProfile.to_dict, ...], "union": <the same
        over all trajectories pooled>, "any_never_accepts": bool,
        "never_accepts_benchmarks": [name, ...]}``.
    """
    results = [analyze(recs, benchmark=str(b)) for b, recs in sorted(per_benchmark.items())]
    pooled: list[Any] = []
    for recs in per_benchmark.values():
        pooled.extend(recs)
    union = analyze(pooled, benchmark="union")
    never = [r.benchmark for r in results if r.never_accepts]
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "union": union.to_dict(),
        "any_never_accepts": bool(never),
        "never_accepts_benchmarks": never,
    }


def _fmt_hist(hist: Mapping[str, int]) -> str:
    return " ".join(f"{k}t:{v}" for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))) or "-"


def render(per_benchmark: Mapping[str, Any]) -> str:
    """A compact text report of the per-benchmark termination profile."""
    report = analyze_benchmarks(per_benchmark)
    lines = ["| benchmark | n | accept rate | exhausted | mean turns | histogram |",
             "|---|---|---|---|---|---|"]
    for r in report["per_benchmark"] + [report["union"]]:
        lines.append(
            f"| {r['benchmark']} | {r['n_trajectories']} | {r['accept_rate']:.2f} | "
            f"{r['exhausted_rate']:.2f} | {r['mean_turns']:.2f} | "
            f"{_fmt_hist(r['turn_histogram'])} |"
        )
    lines.append("")
    if report["any_never_accepts"]:
        lines.append("Verifier NEVER accepts (loop inert, full budget every question) on: "
                     + ", ".join(report["never_accepts_benchmarks"]))
    return "\n".join(lines)
