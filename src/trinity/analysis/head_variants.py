"""Offline R10 check: is the linear head at least as good overall as every variant?

``docs/SPEC.md`` §1.3 invariant **R10** — *"linear head ≥ all other head variants
overall"* (Table 3) — is a replication requirement, yet nothing in ``src/`` or
``scripts/`` verifies it. SPEC §Table 3 (line 385): *"linear best overall
(0.615/0.880/0.916/0.401); sparse edges MMLU only (0.917); block-diag-10 retains much
at 1,024 params."*

R10 is the head-architecture justification: the coordinator ships a plain **linear**
routing head, and R10 says none of the fancier parameterizations (sparse-edge,
block-diagonal-2, block-diagonal-10, …) beats it **overall** — a variant may edge it
out on a single benchmark (sparse edges on MMLU) without winning the equal-weight
average. If some variant did win overall, the linear default would be leaving accuracy
on the table.

Given a ``{variant: {benchmark: score}}`` table, this reports each variant's overall
(equal-weight mean over the benchmarks shared by all variants), the best-overall
variant, whether the linear head is (weakly) the best overall — the R10 verdict — the
margin to the strongest challenger, and the per-benchmark exceptions where a non-linear
variant actually beats linear. Optional parameter counts add the efficiency context
Table 3 highlights (block-diag-10 retaining much at 1,024 params).

Pure stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "HeadVariant",
    "HeadVariantsSummary",
    "analyze_heads",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _clean_scores(scores: Any) -> dict[str, dict[str, float]]:
    """Coerce ``{variant: {benchmark: score}}`` to a clean nested float dict."""
    out: dict[str, dict[str, float]] = {}
    if not isinstance(scores, Mapping):
        return out
    for variant, row in scores.items():
        if not isinstance(row, Mapping):
            continue
        clean = {str(b): float(s) for b, s in row.items() if _is_num(s)}
        if clean:
            out[str(variant)] = clean
    return out


@dataclass(frozen=True)
class HeadVariant:
    """One head variant's per-benchmark scores + its equal-weight overall."""

    name: str
    per_benchmark: dict[str, float]
    overall: float               # equal-weight mean over the shared benchmark set
    n_benchmarks: int
    params: int | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "name": self.name,
            "per_benchmark": dict(self.per_benchmark),
            "overall": self.overall,
            "n_benchmarks": self.n_benchmarks,
            "params": self.params,
        }


@dataclass(frozen=True)
class HeadVariantsSummary:
    """R10 head-variant comparison over the shared benchmark set."""

    linear_key: str
    benchmarks: list[str]        # the shared benchmark set the overall is averaged over
    variants: list[HeadVariant]  # sorted by overall desc
    linear_overall: float
    best_variant: str | None
    best_overall: float
    margin: float                # linear_overall - best NON-linear overall (>0 => linear wins)
    per_benchmark_winner: dict[str, str]
    linear_exceptions: list[str]  # benchmarks where a non-linear variant beats linear
    linear_is_best: bool          # R10 verdict: linear is (weakly) the best overall

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "linear_key": self.linear_key,
            "benchmarks": list(self.benchmarks),
            "variants": [v.to_dict() for v in self.variants],
            "linear_overall": self.linear_overall,
            "best_variant": self.best_variant,
            "best_overall": self.best_overall,
            "margin": self.margin,
            "per_benchmark_winner": dict(self.per_benchmark_winner),
            "linear_exceptions": list(self.linear_exceptions),
            "linear_is_best": self.linear_is_best,
        }


def _empty_summary(linear_key: str) -> HeadVariantsSummary:
    return HeadVariantsSummary(
        linear_key=linear_key, benchmarks=[], variants=[], linear_overall=0.0,
        best_variant=None, best_overall=0.0, margin=0.0, per_benchmark_winner={},
        linear_exceptions=[], linear_is_best=False,
    )


def analyze_heads(
    scores: Any,
    *,
    params: Mapping[str, int] | None = None,
    linear_key: str = "linear",
    tol: float = _TOL,
) -> HeadVariantsSummary:
    """Compute the R10 head-variant comparison from a ``{variant: {benchmark: score}}`` table.

    Overall is the equal-weight mean over the benchmarks **shared by all variants** (a
    ragged table would make the average an unfair apples-to-oranges comparison, so only
    the intersection is scored — mirroring the composite's equal benchmark weighting).

    Args:
        scores: ``{variant: {benchmark: score}}``.
        params: optional ``{variant: parameter_count}`` for the efficiency context.
        linear_key: the name of the linear head in ``scores`` (default ``"linear"``).
        tol: a variant only *beats* linear overall if it exceeds it by more than ``tol``
            (so a tie keeps R10 holding — "linear ≥ all others").

    Returns:
        A :class:`HeadVariantsSummary`. ``linear_is_best`` is False when the linear key is
        absent or fewer than two variants share a benchmark (nothing to compare).
    """
    clean = _clean_scores(scores)
    params = dict(params) if isinstance(params, Mapping) else {}
    if linear_key not in clean:
        return _empty_summary(linear_key)

    shared = set.intersection(*(set(row) for row in clean.values())) if clean else set()
    benchmarks = sorted(shared)
    if not benchmarks:
        return _empty_summary(linear_key)

    variants: list[HeadVariant] = []
    for name, row in clean.items():
        overall = sum(row[b] for b in benchmarks) / len(benchmarks)
        variants.append(HeadVariant(
            name=name, per_benchmark={b: row[b] for b in benchmarks},
            overall=overall, n_benchmarks=len(benchmarks), params=params.get(name),
        ))
    variants.sort(key=lambda v: v.overall, reverse=True)

    linear = next(v for v in variants if v.name == linear_key)
    best = variants[0]
    non_linear = [v for v in variants if v.name != linear_key]
    best_non_linear = max((v.overall for v in non_linear), default=float("-inf"))
    margin = linear.overall - best_non_linear

    per_benchmark_winner = {
        b: max(variants, key=lambda v: v.per_benchmark[b]).name for b in benchmarks
    }
    linear_exceptions = [
        b for b in benchmarks
        if any(v.per_benchmark[b] > linear.per_benchmark[b] + tol for v in non_linear)
    ]
    # R10: linear is (weakly) the best overall -> no other variant exceeds it by > tol.
    linear_is_best = bool(non_linear) and all(
        linear.overall + tol >= v.overall for v in non_linear
    )
    return HeadVariantsSummary(
        linear_key=linear_key, benchmarks=benchmarks, variants=variants,
        linear_overall=linear.overall, best_variant=best.name, best_overall=best.overall,
        margin=margin, per_benchmark_winner=per_benchmark_winner,
        linear_exceptions=linear_exceptions, linear_is_best=linear_is_best,
    )


def render(
    scores: Any,
    *,
    params: Mapping[str, int] | None = None,
    linear_key: str = "linear",
    tol: float = _TOL,
) -> str:
    """Markdown: per-variant overall table + the R10 verdict and per-benchmark exceptions."""
    s = analyze_heads(scores, params=params, linear_key=linear_key, tol=tol)
    out = ["# R10 head-variant comparison\n"]
    if not s.variants:
        return "".join(out) + "\n_(no comparable head-variant scores)_\n"

    head = " | ".join(f"acc[{b}]" for b in s.benchmarks)
    out.append(f"| head | overall | {head} | params |")
    out.append("|---|---|" + "---|" * len(s.benchmarks) + "---|")
    for v in s.variants:
        cells = " | ".join(f"{v.per_benchmark[b]:.3f}" for b in s.benchmarks)
        star = " ⭐" if v.name == s.linear_key else ""
        params_s = f"{v.params:,}" if v.params is not None else "—"
        out.append(f"| {v.name}{star} | {v.overall:.3f} | {cells} | {params_s} |")

    out.append(f"\n- best overall: **{s.best_variant}** ({s.best_overall:.3f}); "
               f"linear overall {s.linear_overall:.3f} (margin to best challenger "
               f"{s.margin:+.3f})")
    if s.linear_exceptions:
        wins = ", ".join(f"{b}→{s.per_benchmark_winner[b]}" for b in s.linear_exceptions)
        out.append(f"- per-benchmark exceptions (a variant beats linear): {wins}")
    verdict = "HOLDS" if s.linear_is_best else "VIOLATED"
    out.append(f"\n**R10** (linear head ≥ all other head variants overall): {verdict}.")
    return "\n".join(out) + "\n"
