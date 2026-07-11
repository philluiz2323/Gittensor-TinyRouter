"""Paired significance for the R1/R2/R4 eval invariants.

``trinity.eval`` reports the headline invariants as **bare point comparisons**
(``s_trinity > best_single``), and ``docs/RESULTS.md`` repeatedly notes the margins
are *"inside the noise"* (e.g. math R1/R2 ``0.792 vs 0.794``). SPEC §6.4 mandates
reporting uncertainty ("report mean ± std"), and ``docs/ORACLE_CEILING_DIAGNOSTIC.md``
§4 already reads *the oracle* verdict "off bootstrap CIs, never the point estimates" —
but the same rigor was never applied to the headline TRINITY-vs-baseline invariants.

This module supplies exactly that: given per-question correctness vectors for TRINITY
and each baseline (all scored on the SAME questions), it computes a **paired bootstrap
CI** of the mean accuracy difference and a **paired McNemar** discordant-pairs test, and
turns each invariant into a CI-gated verdict (significant hold / inside-the-noise /
significant miss) instead of a knife-edge boolean.

It is a read-only statistics layer over already-graded correctness — it changes no
scoring or fitness math, and reuses the same paired-comparison idea already shipped in
``scripts/oracle_ceiling.py``. Pure numpy + stdlib (no scipy), no torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, erfc, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "PairedComparison",
    "InvariantSignificance",
    "mcnemar",
    "paired_bootstrap_ci",
    "paired_diff_test",
    "assess_invariants",
]

# 0/1 or float correctness vectors, as lists or numpy arrays.
_ArrayLike = Sequence[float] | np.ndarray


def mcnemar(correct_a: _ArrayLike, correct_b: _ArrayLike) -> dict[str, Any]:
    """Paired McNemar test on two 0/1 correctness vectors over the SAME questions.

    Only the discordant pairs carry signal: ``a_right_b_wrong`` and ``b_right_a_wrong``.
    Under H0 (the systems are equally likely to win a discordant pair) the split is
    Binomial(n, 0.5). The p-value is the exact two-sided binomial tail for
    ``n <= 1000`` (fast and precise at eval sizes), and a continuity-corrected normal
    approximation via :func:`math.erfc` beyond that — so no scipy dependency.

    Args:
        correct_a: 0/1 correctness of system A per question.
        correct_b: 0/1 correctness of system B, aligned to A.

    Returns:
        ``{a_right_b_wrong, b_right_a_wrong, n_discordant, statistic, p_value}``.

    Raises:
        ValueError: If the two vectors differ in length.
    """
    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    a_win = int(((a == 1) & (b == 0)).sum())
    b_win = int(((a == 0) & (b == 1)).sum())
    n = a_win + b_win
    if n == 0:
        return {"a_right_b_wrong": a_win, "b_right_a_wrong": b_win, "n_discordant": 0,
                "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(a_win - b_win) - 1.0) ** 2 / n   # chi-square w/ continuity corr.
    k = min(a_win, b_win)
    if n <= 1000:
        tail = sum(comb(n, i) for i in range(k + 1))
        p = min(1.0, 2.0 * tail / (2.0 ** n))
    else:
        z = (abs(a_win - b_win) - 1.0) / sqrt(n)
        p = min(1.0, erfc(z / sqrt(2.0)))
    return {"a_right_b_wrong": a_win, "b_right_a_wrong": b_win, "n_discordant": n,
            "statistic": float(statistic), "p_value": float(p)}


def paired_bootstrap_ci(
    a: _ArrayLike,
    b: _ArrayLike,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for the PAIRED mean difference ``mean(a - b)``.

    Resamples question indices with replacement (the SAME indices for both systems, so
    the pairing is preserved) and takes the ``alpha/2`` and ``1-alpha/2`` quantiles of
    the resampled mean difference. Deterministic for a given ``seed``.

    Returns:
        ``{point, ci_lo, ci_hi}`` for ``mean(a) - mean(b)`` (empty input -> all 0.0).
    """
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape:
        raise ValueError(f"length mismatch: {av.shape} vs {bv.shape}")
    n = av.shape[0]
    if n == 0:
        return {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    diff = av - bv
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    return {
        "point": float(diff.mean()),
        "ci_lo": float(np.quantile(boot, alpha / 2.0)),
        "ci_hi": float(np.quantile(boot, 1.0 - alpha / 2.0)),
    }


@dataclass(frozen=True)
class PairedComparison:
    """One invariant's paired significance result (system A vs baseline B)."""

    name_a: str
    name_b: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    significant: bool
    verdict: str
    mcnemar: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "diff": self.diff,
            "ci_95": [self.ci_lo, self.ci_hi],
            "p_value": self.p_value,
            "significant": self.significant,
            "verdict": self.verdict,
            "mcnemar": self.mcnemar,
        }


def paired_diff_test(
    correct_a: _ArrayLike,
    correct_b: _ArrayLike,
    *,
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedComparison:
    """Compare A vs B on the same questions: bootstrap CI + McNemar + CI-gated verdict.

    A verdict is **significant** only when the ``1-alpha`` CI of ``mean(A) - mean(B)``
    excludes 0 — the "read it off the CI, never the point estimate" rule. A margin whose
    CI straddles 0 is reported as *inside the noise* however large the point difference.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    ci = paired_bootstrap_ci(a, b, n_boot=n_boot, seed=seed, alpha=alpha)
    mc = mcnemar(a.astype(int), b.astype(int))
    lo, hi = ci["ci_lo"], ci["ci_hi"]
    pct = int(round((1.0 - alpha) * 100))
    if lo > 0.0:
        significant, verdict = True, (
            f"SIGNIFICANT: {name_a} > {name_b} ({pct}% CI [{lo:.3f}, {hi:.3f}] excludes 0, "
            f"McNemar p={mc['p_value']:.3f})"
        )
    elif hi < 0.0:
        significant, verdict = True, (
            f"SIGNIFICANT MISS: {name_a} < {name_b} ({pct}% CI [{lo:.3f}, {hi:.3f}] excludes 0, "
            f"McNemar p={mc['p_value']:.3f})"
        )
    else:
        significant, verdict = False, (
            f"NOT SIGNIFICANT: inside the noise ({pct}% CI [{lo:.3f}, {hi:.3f}] includes 0, "
            f"McNemar p={mc['p_value']:.3f})"
        )
    return PairedComparison(
        name_a=name_a, name_b=name_b, n=int(a.shape[0]),
        mean_a=float(a.mean()) if a.size else 0.0,
        mean_b=float(b.mean()) if b.size else 0.0,
        diff=float(ci["point"]), ci_lo=lo, ci_hi=hi,
        p_value=float(mc["p_value"]), significant=significant, verdict=verdict, mcnemar=mc,
    )


@dataclass(frozen=True)
class InvariantSignificance:
    """Paired-significance verdicts for the headline eval invariants."""

    benchmark: str | None
    n_questions: int
    comparisons: list[PairedComparison]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_questions": self.n_questions,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def assess_invariants(
    per_item: Mapping[str, Any],
    *,
    benchmark: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> InvariantSignificance:
    """Assess R1/R2 (TRINITY vs best single) and R4 (TRINITY vs random) with CIs.

    Args:
        per_item: The eval ``per_item`` block: ``{"task_ids": [...], "TRINITY": [0/1..],
            "single::<model>": [...], "random_routing": [...]}``. Missing systems are
            skipped rather than raising.
        benchmark: Benchmark name for the summary.
        n_boot, seed, alpha: Bootstrap settings (paired resampling; CI level 1-alpha).

    Returns:
        The :class:`InvariantSignificance` with one comparison per available invariant.
    """
    tri = per_item.get("TRINITY")
    singles = {k[len("single::"):]: v for k, v in per_item.items() if k.startswith("single::")}
    rand = per_item.get("random_routing")
    n = len(tri) if tri is not None else 0

    comparisons: list[PairedComparison] = []
    if tri is not None and singles:
        # best single = the one with the highest mean accuracy (the R1/R2 comparator).
        best_model = max(singles, key=lambda m: float(np.mean(singles[m])) if len(singles[m]) else 0.0)
        comparisons.append(paired_diff_test(
            tri, singles[best_model], name_a="TRINITY", name_b=f"best single ({best_model})",
            n_boot=n_boot, seed=seed, alpha=alpha,
        ))
    if tri is not None and rand is not None:
        comparisons.append(paired_diff_test(
            tri, rand, name_a="TRINITY", name_b="random routing",
            n_boot=n_boot, seed=seed, alpha=alpha,
        ))
    return InvariantSignificance(benchmark=benchmark, n_questions=n, comparisons=comparisons)
