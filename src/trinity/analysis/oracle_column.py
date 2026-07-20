"""Oracle/headroom/gap-closed columns for the headline results table.

Why this exists
---------------
``docs/ORACLE_CEILING_DIAGNOSTIC.md`` §7 lists the diagnostic's deliverables. Item 3 is:

    **``scripts/results_table.py``** gains an oracle column (oracle, headroom, gap_closed
    with CIs).

It was never built — ``results_table.py`` reports TRINITY against the best single model and
random routing, with **no ceiling context at all**. That is exactly the ambiguity the oracle
diagnostic exists to remove: a "✅ TRINITY > best single" tells you the router won, but not
whether it won by capturing nearly all of the achievable headroom (so further router tuning
is pointless and the lever is the **pool**) or by capturing a sliver of it (so there is real
room left). ``oracle_ceiling.py`` already computes those numbers and writes
``experiments/**/oracle_report_<bench>.json``; nothing joined them back into the table.

This module is the pure half: it parses an ``oracle_ceiling`` report into the three
mandated columns, preferring the **bootstrap CI** block over the bare point estimates so the
table shows the interval the diagnostic insists verdicts be read from ("the verdict is read
off the CIs, never the point estimates", §4). ``scripts/results_table.py`` does the globbing
and joins by benchmark.

``router_gap_closed`` is reported as-is, including values **above 100%**: a TRINITY accuracy
above the cross-fit oracle point estimate is a real, informative outcome (the cross-fit
estimator debiases the winner's curse and can land below an actual router), so it is
surfaced rather than clamped — clamping would hide the very disagreement worth looking at.

Pure / offline — stdlib only, over JSON already on disk. No torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "Estimate",
    "OracleColumns",
    "from_oracle_report",
    "index_by_benchmark",
]


def _num(x: Any) -> Optional[float]:
    """``float(x)`` for a real number, else ``None`` (booleans are not numbers here)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


@dataclass(frozen=True)
class Estimate:
    """A point estimate with an optional 95% interval."""

    point: float
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None

    @property
    def has_ci(self) -> bool:
        return self.ci_lo is not None and self.ci_hi is not None

    def render(self, digits: int = 3) -> str:
        """``0.855 [0.801, 0.903]``, or just the point when no interval is available."""
        base = f"{self.point:.{digits}f}"
        if not self.has_ci:
            return base
        return f"{base} [{self.ci_lo:.{digits}f}, {self.ci_hi:.{digits}f}]"

    def to_dict(self) -> dict[str, Any]:
        return {"point": self.point, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi}


@dataclass(frozen=True)
class OracleColumns:
    """The §7 item-3 columns for one benchmark."""

    benchmark: str
    oracle: Optional[Estimate] = None
    headroom: Optional[Estimate] = None
    gap_closed: Optional[float] = None
    verdict_label: Optional[str] = None

    @property
    def empty(self) -> bool:
        """True when the report carried none of the three columns (nothing to show)."""
        return self.oracle is None and self.headroom is None and self.gap_closed is None

    def render_gap_closed(self) -> str:
        """``router_gap_closed`` as a percentage; ``>100%`` is kept, never clamped."""
        if self.gap_closed is None:
            return "—"
        return f"{self.gap_closed * 100:.0f}%"

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "oracle": self.oracle.to_dict() if self.oracle else None,
            "headroom": self.headroom.to_dict() if self.headroom else None,
            "gap_closed": self.gap_closed,
            "verdict_label": self.verdict_label,
        }


def _estimate(report: Mapping[str, Any], key: str) -> Optional[Estimate]:
    """Pull ``key`` preferring the bootstrap-CI block, falling back to the point estimate.

    §4 of the diagnostic requires verdicts be read off the CIs, so the interval wins whenever
    ``oracle_ceiling`` recorded one; an older report that only has ``point_estimates`` still
    renders (without an interval) rather than dropping the column.
    """
    boot = report.get("bootstrap_ci_95")
    if isinstance(boot, Mapping):
        entry = boot.get(key)
        if isinstance(entry, Mapping):
            point = _num(entry.get("point"))
            if point is not None:
                return Estimate(point, _num(entry.get("ci_lo")), _num(entry.get("ci_hi")))

    points = report.get("point_estimates")
    if isinstance(points, Mapping):
        point = _num(points.get(key))
        if point is not None:
            return Estimate(point)
    return None


def from_oracle_report(report: Mapping[str, Any]) -> Optional[OracleColumns]:
    """Parse one ``oracle_report_<bench>.json`` into :class:`OracleColumns`.

    Args:
        report: The parsed report written by ``scripts/oracle_ceiling.py``.

    Returns:
        The columns, or ``None`` when the mapping carries no benchmark name (so a stray
        JSON in the tree can never be joined onto the wrong row).
    """
    if not isinstance(report, Mapping):
        return None
    benchmark = report.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        return None

    gap_closed = None
    label = None
    verdict = report.get("verdict")
    if isinstance(verdict, Mapping):
        gap_closed = _num(verdict.get("router_gap_closed"))
        raw_label = verdict.get("label")
        label = raw_label if isinstance(raw_label, str) else None
    if gap_closed is None:
        trinity = report.get("trinity")
        if isinstance(trinity, Mapping):
            gap_closed = _num(trinity.get("router_gap_closed"))

    return OracleColumns(
        benchmark=benchmark,
        oracle=_estimate(report, "routing_oracle"),
        headroom=_estimate(report, "routing_headroom"),
        gap_closed=gap_closed,
        verdict_label=label,
    )


def index_by_benchmark(reports: Iterable[Mapping[str, Any]]) -> dict[str, OracleColumns]:
    """``{benchmark: OracleColumns}``, skipping unparseable or column-less reports.

    First report wins for a benchmark, so a deterministic (sorted) scan yields a
    deterministic table even when a tree holds several reports for the same benchmark.
    """
    out: dict[str, OracleColumns] = {}
    for report in reports:
        cols = from_oracle_report(report)
        if cols is None or cols.empty or cols.benchmark in out:
            continue
        out[cols.benchmark] = cols
    return out
