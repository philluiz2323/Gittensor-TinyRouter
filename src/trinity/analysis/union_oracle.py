"""Cross-benchmark (3-benchmark *union*) oracle-headroom analysis.

``scripts/oracle_ceiling.py`` measures the routing ceiling one benchmark at a time, but
``ROADMAP.md`` Phase 2 mandates the same measurement **for the 3-benchmark union** —

    Per benchmark and for the 3-benchmark union:
    - best single model
    - routing oracle
    - disagreement rate
    - estimated headroom            (ROADMAP.md §"Phase 2", lines 74-79)
    ...
    - equal benchmark weighting in the composite score   (line 53)

— which exists nowhere in ``src/`` or ``scripts/``. This is the number behind the
RESULTS.md thesis that *the win is cross-task, not within-task*: a **fixed** single model
must use the same model on every benchmark (so it is dragged down by the benchmark it is
weakest on), whereas a per-query router can pick the specialist per task. The **union
headroom** = ``routing_oracle - best_fixed_single`` over the equally-weighted pool
quantifies exactly that gap.

It also computes the **Relative Error Reduction** metric (SPEC §6.3 / R13,
``RER = (Z − S*)/(1 − S*)``) — separately mandated and previously uncomputed — for the
oracle ceiling.

Read-only additive analysis: it **reuses** :func:`trinity.analysis.complementarity
.solve_matrix_from_matrix` (merged #160) to decode each per-benchmark solve matrix, then
does pure-numpy pooling with **equal benchmark weighting**. No torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from trinity.analysis.complementarity import solve_matrix_from_matrix

__all__ = [
    "BenchmarkOracle",
    "UnionOracleSummary",
    "relative_error_reduction",
    "oracle_from_matrix",
    "union_oracle",
    "render",
]

_TOL = 1e-9


def relative_error_reduction(z: float, s_star: float) -> float | None:
    """SPEC §6.3 RER = ``(Z − S*) / (1 − S*)`` — fraction of the residual error closed.

    ``Z`` is the coordinated/ceiling score, ``S*`` the best single model. Returns None
    when ``S* >= 1`` (no residual error to reduce), matching the ceiling diagnostic's
    "NaN on non-positive denominator" convention.
    """
    denom = 1.0 - s_star
    return (z - s_star) / denom if denom > _TOL else None


@dataclass(frozen=True)
class BenchmarkOracle:
    """Single-benchmark oracle summary (the per-benchmark row of the union report)."""

    benchmark: str
    n_queries: int
    models: list[str]
    per_model_accuracy: dict[str, float]
    best_single_model: str | None
    best_single: float
    routing_oracle: float
    headroom: float
    disagreement_rate: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_queries": self.n_queries,
            "models": list(self.models),
            "per_model_accuracy": dict(self.per_model_accuracy),
            "best_single_model": self.best_single_model,
            "best_single": self.best_single,
            "routing_oracle": self.routing_oracle,
            "headroom": self.headroom,
            "disagreement_rate": self.disagreement_rate,
        }


@dataclass(frozen=True)
class UnionOracleSummary:
    """Per-benchmark oracles + the equally-weighted 3-benchmark union aggregation."""

    benchmarks: list[BenchmarkOracle]
    n_benchmarks: int
    models: list[str]
    equal_weight_per_model_accuracy: dict[str, float]
    best_single_model: str | None
    best_single: float
    routing_oracle: float
    union_headroom: float
    disagreement_rate: float
    oracle_rer: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_benchmarks": self.n_benchmarks,
            "models": list(self.models),
            "equal_weight_per_model_accuracy": dict(self.equal_weight_per_model_accuracy),
            "best_single_model": self.best_single_model,
            "best_single": self.best_single,
            "routing_oracle": self.routing_oracle,
            "union_headroom": self.union_headroom,
            "disagreement_rate": self.disagreement_rate,
            "oracle_rer": self.oracle_rer,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


def oracle_from_matrix(matrix: dict, *, threshold: float = 0.5) -> BenchmarkOracle:
    """Best-single / routing-oracle / headroom / disagreement for one benchmark matrix.

    Decodes the ``oracle_matrix`` schema (``{"benchmark", "tasks":[{"id","per_model":
    {m:[0/1..K]}}]}``) via the merged ``complementarity.solve_matrix_from_matrix`` into a
    hard-solve matrix, then computes the routing ceiling with pure numpy.
    """
    B, models = solve_matrix_from_matrix(matrix, threshold=threshold)
    bench = str(matrix.get("benchmark", "?"))
    Q, M = (B.shape[0], B.shape[1]) if B.ndim == 2 else (0, 0)
    if Q == 0 or M == 0:
        return BenchmarkOracle(bench, Q, list(models), {}, None, 0.0, 0.0, 0.0, 0.0)

    acc = B.mean(axis=0)
    best_idx = int(np.argmax(acc))
    routing_oracle = float(B.max(axis=1).mean())      # per-query: pick a solver if any
    best_single = float(acc[best_idx])
    row_sums = B.sum(axis=1)
    disagreement = float(((row_sums > 0) & (row_sums < M)).mean()) if M > 1 else 0.0
    return BenchmarkOracle(
        benchmark=bench,
        n_queries=Q,
        models=list(models),
        per_model_accuracy={m: float(acc[i]) for i, m in enumerate(models)},
        best_single_model=models[best_idx],
        best_single=best_single,
        routing_oracle=routing_oracle,
        headroom=routing_oracle - best_single,
        disagreement_rate=disagreement,
    )


def union_oracle(matrices: Sequence[dict], *, threshold: float = 0.5) -> UnionOracleSummary:
    """Aggregate per-benchmark oracles into the equally-weighted 3-benchmark union.

    Each benchmark contributes equally (not query-count-weighted, which would over-weight
    a larger split — ROADMAP line 53). The union **best fixed single** is the model with
    the highest equal-weight mean accuracy across benchmarks; the union **routing oracle**
    is the equal-weight mean of the per-benchmark per-query oracles; ``union_headroom`` is
    their difference — the cross-task routing gain a fixed single model cannot capture.

    Raises:
        ValueError: If the benchmarks do not share one model set (a ragged pool would
            make the equal-weight per-model average meaningless).
    """
    oracles = [oracle_from_matrix(m, threshold=threshold) for m in matrices]
    covered = [o for o in oracles if o.n_queries > 0]
    if not covered:
        return UnionOracleSummary([], 0, [], {}, None, 0.0, 0.0, 0.0, 0.0, None)

    models = covered[0].models
    for o in covered:
        if o.models != models:
            raise ValueError(f"benchmark {o.benchmark!r} has models {o.models}, expected {models}")

    n = len(covered)
    ew_acc = {m: sum(o.per_model_accuracy[m] for o in covered) / n for m in models}
    best_model = max(models, key=lambda m: ew_acc[m])
    best_single = ew_acc[best_model]
    routing_oracle = sum(o.routing_oracle for o in covered) / n
    disagreement = sum(o.disagreement_rate for o in covered) / n
    return UnionOracleSummary(
        benchmarks=oracles,
        n_benchmarks=n,
        models=list(models),
        equal_weight_per_model_accuracy=ew_acc,
        best_single_model=best_model,
        best_single=best_single,
        routing_oracle=routing_oracle,
        union_headroom=routing_oracle - best_single,
        disagreement_rate=disagreement,
        oracle_rer=relative_error_reduction(routing_oracle, best_single),
    )


def render(summary: UnionOracleSummary) -> str:
    """Markdown: per-benchmark rows + the equally-weighted union row + verdict."""
    out = ["# Cross-benchmark union oracle headroom\n"]
    if summary.n_benchmarks == 0:
        return "".join(out) + "\n_(no benchmark matrices)_\n"

    out.append("| scope | best fixed single | routing oracle | headroom | disagreement |")
    out.append("|---|---|---|---|---|")
    for b in summary.benchmarks:
        if b.n_queries == 0:
            continue
        out.append(f"| {b.benchmark} | {b.best_single:.3f} ({b.best_single_model}) | "
                   f"{b.routing_oracle:.3f} | {b.headroom:+.3f} | {b.disagreement_rate:.3f} |")
    out.append(f"| **UNION (equal-weight, n={summary.n_benchmarks})** | "
               f"**{summary.best_single:.3f} ({summary.best_single_model})** | "
               f"**{summary.routing_oracle:.3f}** | **{summary.union_headroom:+.3f}** | "
               f"{summary.disagreement_rate:.3f} |")

    rer = summary.oracle_rer
    rer_s = f"{rer:.1%}" if rer is not None else "N/A (best single ≥ 1)"
    out.append(f"\n- **union headroom** (routing beats the best fixed single by): "
               f"{summary.union_headroom:+.3f}")
    out.append(f"- **oracle RER** (SPEC §6.3, max relative-error reduction of the ceiling): {rer_s}")
    verdict = ("routing headroom is REAL cross-task" if summary.union_headroom > 0.02
               else "little cross-task headroom — the lever is the pool, not the router")
    out.append(f"\n**Verdict:** {verdict}.")
    return "\n".join(out) + "\n"
