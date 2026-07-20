"""Zero-shot transfer: does the router still win on benchmarks it never trained on?

Why this exists
---------------
``docs/SPEC.md`` splits the benchmark suite in two:

* **§6.1 in-distribution (train + eval)** — MATH500, MMLU, RLPR, LiveCodeBench;
* **§6.2 held-out / zero-shot transfer (no retraining)** — AIME2025, BigCodeBench,
  MT-Bench(-101), GPQA-Diamond.

and Table 1 (§7) carries the headline transfer result: held-out **Avg 54.21**, which
"beats best single Gemini 52.34". That is the SPEC's strongest generalization claim — a
coordinator that only wins where it was trained has learned the tasks, not routing.

**Nothing in the repo encodes that split.** Grepping for an in-distribution / held-out
partition finds no machinery at all, so every existing report treats all benchmarks alike:

* ``scripts/results_table.py`` averages every benchmark into one multi-task summary;
* :mod:`trinity.analysis.significance` gives per-file R1/R2/R4 CIs with no cohort notion;
* :mod:`trinity.analysis.generalization` measures the **eval-vs-audit** overfit gap — a
  different axis entirely (same benchmark, different split), not train-vs-unseen-benchmark.

So a router that beats the best single model in-distribution and collapses off it reads as a
clean win everywhere. This module separates the two cohorts and reports the margin in each,
plus the **transfer gap** between them.

The margin per cohort is TRINITY minus the best single model, averaged over that cohort's
benchmarks. The verdict is deliberately conservative: transfer only "holds" when the
held-out margin is itself positive — a shrinking-but-positive margin is reported as
*degraded*, and a negative one as *failed*, because a mean over a cohort can stay positive
while individual held-out tasks lose. Per-benchmark rows are always kept so a cohort mean
can never hide a task-level loss.

Benchmarks the SPEC does not list are ``unknown`` and are counted but excluded from both
cohorts, rather than silently folded into one.

Pure / offline — stdlib over eval JSONs already on disk. No torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, TypeGuard

from trinity.orchestration.reward import resolve_benchmark

__all__ = [
    "IN_DISTRIBUTION",
    "HELD_OUT",
    "classify",
    "BenchmarkMargin",
    "CohortSummary",
    "TransferSummary",
    "assess",
    "render",
]

#: SPEC §6.1 — trained AND evaluated on these.
IN_DISTRIBUTION: frozenset[str] = frozenset({"math500", "mmlu", "rlpr", "livecodebench"})

#: SPEC §6.2 — evaluated with NO retraining. ``gpqa`` is the repo's name for GPQA-Diamond.
HELD_OUT: frozenset[str] = frozenset({"aime", "aime2025", "bigcodebench", "mtbench",
                                      "mt_bench", "gpqa"})

IN_DIST = "in_distribution"
HELD = "held_out"
UNKNOWN = "unknown"


def classify(benchmark: str) -> str:
    """Cohort for ``benchmark``: ``in_distribution`` / ``held_out`` / ``unknown``.

    Names are resolved through the grader's own alias table first (so ``livecodebench_v6``
    lands with ``livecodebench``), then matched against the SPEC lists. An unlisted
    benchmark is ``unknown`` — never guessed into a cohort, since misfiling one would move
    a task across the very boundary this module measures.
    """
    raw = (benchmark or "").strip().lower()
    if not raw:
        return UNKNOWN
    try:
        key = resolve_benchmark(raw)
    except Exception:
        key = raw
    for candidate in (key, raw):
        if candidate in IN_DISTRIBUTION:
            return IN_DIST
        if candidate in HELD_OUT:
            return HELD
    return UNKNOWN


@dataclass(frozen=True)
class BenchmarkMargin:
    """TRINITY vs the best single model on one benchmark."""

    benchmark: str
    cohort: str
    trinity: float
    best_single: float
    best_model: Optional[str] = None

    @property
    def margin(self) -> float:
        return self.trinity - self.best_single

    @property
    def wins(self) -> bool:
        return self.margin > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "cohort": self.cohort,
            "trinity": self.trinity,
            "best_single": self.best_single,
            "best_model": self.best_model,
            "margin": self.margin,
            "wins": self.wins,
        }


@dataclass(frozen=True)
class CohortSummary:
    """Mean TRINITY / best-single / margin over one cohort's benchmarks."""

    cohort: str
    benchmarks: list[BenchmarkMargin] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.benchmarks)

    @property
    def trinity(self) -> Optional[float]:
        return sum(b.trinity for b in self.benchmarks) / self.n if self.n else None

    @property
    def best_single(self) -> Optional[float]:
        return sum(b.best_single for b in self.benchmarks) / self.n if self.n else None

    @property
    def margin(self) -> Optional[float]:
        return sum(b.margin for b in self.benchmarks) / self.n if self.n else None

    @property
    def losses(self) -> list[str]:
        """Benchmarks in this cohort where TRINITY does NOT beat the best single."""
        return [b.benchmark for b in self.benchmarks if not b.wins]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "n": self.n,
            "trinity": self.trinity,
            "best_single": self.best_single,
            "margin": self.margin,
            "losses": self.losses,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


@dataclass(frozen=True)
class TransferSummary:
    """The SPEC §6.2 zero-shot transfer verdict."""

    in_distribution: CohortSummary
    held_out: CohortSummary
    unknown: list[str] = field(default_factory=list)

    @property
    def transfer_gap(self) -> Optional[float]:
        """held-out margin minus in-distribution margin (negative = margin shrinks)."""
        a, b = self.held_out.margin, self.in_distribution.margin
        return None if a is None or b is None else a - b

    @property
    def verdict(self) -> str:
        """``holds`` / ``degraded`` / ``failed`` / ``insufficient_evidence``.

        ``holds`` requires the held-out margin to be POSITIVE — a router whose advantage
        survives only in-distribution has not demonstrated transfer. ``degraded`` marks a
        positive held-out margin that is nonetheless smaller than the in-distribution one.
        """
        held = self.held_out.margin
        if held is None:
            return "insufficient_evidence"
        if held <= 0.0:
            return "failed"
        gap = self.transfer_gap
        if gap is not None and gap < 0.0:
            return "degraded"
        return "holds"

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_distribution": self.in_distribution.to_dict(),
            "held_out": self.held_out.to_dict(),
            "unknown": list(self.unknown),
            "transfer_gap": self.transfer_gap,
            "verdict": self.verdict,
        }


def assess(rows: Iterable[dict[str, Any]]) -> TransferSummary:
    """Partition ``rows`` into SPEC cohorts and measure the transfer margin.

    Args:
        rows: Dicts carrying ``benchmark``, ``trinity`` and ``best_single`` (the shape
            ``scripts/results_table.py``'s ``load_rows`` already produces), optionally
            ``best_model``. Rows missing either score are skipped — an absent number is
            never read as a zero, which would fabricate a loss.

    Returns:
        The :class:`TransferSummary`. When a benchmark appears more than once (several
        coordinators), the BEST TRINITY score for it is used, mirroring the per-task-best
        reduction ``results_table``'s multi-task summary applies.
    """
    best_by_bench: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []

    for row in rows:
        bench = str(row.get("benchmark") or "").strip()
        tri, single = row.get("trinity"), row.get("best_single")
        if not bench or not _is_num(tri) or not _is_num(single):
            continue
        cohort = classify(bench)
        if cohort == UNKNOWN:
            if bench not in unknown:
                unknown.append(bench)
            continue
        prev = best_by_bench.get(bench)
        if prev is None or float(tri) > float(prev["trinity"]):
            best_by_bench[bench] = {
                "trinity": float(tri), "best_single": float(single),
                "best_model": row.get("best_model"), "cohort": cohort,
            }

    cohorts: dict[str, list[BenchmarkMargin]] = {IN_DIST: [], HELD: []}
    for bench in sorted(best_by_bench):
        v = best_by_bench[bench]
        cohorts[v["cohort"]].append(BenchmarkMargin(
            benchmark=bench, cohort=v["cohort"], trinity=v["trinity"],
            best_single=v["best_single"], best_model=v["best_model"],
        ))

    return TransferSummary(
        in_distribution=CohortSummary(IN_DIST, cohorts[IN_DIST]),
        held_out=CohortSummary(HELD, cohorts[HELD]),
        unknown=unknown,
    )


def _is_num(x: Any) -> TypeGuard[float]:
    """A real number (booleans excluded), narrowed for the caller."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def render(summary: TransferSummary) -> str:
    """Markdown: per-benchmark margins by cohort, the transfer gap, and the verdict."""
    out = ["# Zero-shot transfer (SPEC §6.1 in-distribution vs §6.2 held-out)\n"]
    out.append("_Does the router still beat the best single model on benchmarks it never "
               "trained on?_\n")

    out.append("| cohort | benchmark | TRINITY | best single | margin |")
    out.append("|---|---|---|---|---|")
    for cohort in (summary.in_distribution, summary.held_out):
        for b in cohort.benchmarks:
            mark = "✅" if b.wins else "❌"
            out.append(f"| {cohort.cohort} | {b.benchmark} | {b.trinity:.3f} | "
                       f"{b.best_single:.3f} ({b.best_model or '—'}) | {b.margin:+.3f} {mark} |")
    if not summary.in_distribution.n and not summary.held_out.n:
        return "".join(out[:2]) + "\n_(no benchmarks classified)_\n"

    for cohort in (summary.in_distribution, summary.held_out):
        out.append(f"| **{cohort.cohort} mean (n={cohort.n})** | | "
                   f"**{_fmt(cohort.trinity)}** | **{_fmt(cohort.best_single)}** | "
                   f"**{_fmt(cohort.margin)}** |")

    out.append(f"\n- **transfer gap** (held-out margin − in-dist margin): "
               f"{_fmt(summary.transfer_gap)}")
    if summary.held_out.losses:
        out.append(f"- held-out benchmarks TRINITY does NOT win: "
                   f"{', '.join(summary.held_out.losses)}")
    if summary.unknown:
        out.append(f"- not in either SPEC list (excluded): {', '.join(summary.unknown)}")

    messages = {
        "holds": "transfer HOLDS — the routing advantage survives off-distribution",
        "degraded": ("transfer DEGRADED — the router still wins held-out, but by a smaller "
                     "margin than in-distribution"),
        "failed": ("transfer FAILED — the routing advantage does not survive off-"
                   "distribution; the router may have learned the training tasks"),
        "insufficient_evidence": ("insufficient evidence — no held-out benchmark was "
                                  "evaluated, so §6.2 is untested"),
    }
    out.append(f"\n**Verdict:** {messages[summary.verdict]}.")
    return "\n".join(out) + "\n"
