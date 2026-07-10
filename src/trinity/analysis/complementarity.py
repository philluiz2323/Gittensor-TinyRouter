"""Pool-complementarity audit: which model is redundant, and which to swap.

``scripts/oracle_ceiling.py`` is titled *"Oracle-ceiling diagnostic + pool-
complementarity audit"* and its POOL_BOUND verdict declares *"the lever is the model
pool, not the router"* — but it never says **which** model to swap. This module
supplies the missing complementarity half that the plan
(``docs/ORACLE_CEILING_DIAGNOSTIC.md`` §6) and ``docs/IMPROVEMENTS.md`` #1 call for:

  > recommend swapping the most redundant model (**highest pairwise agreement, fewest
  > unique solves**) for one correct on a disjoint slice.

From the same solve matrix the oracle ceiling already consumes, it computes, per model:

* **accuracy** — its own solve rate;
* **unique solves** — questions ONLY it solves (its irreplaceable contribution);
* **marginal oracle contribution** — how far the "solved-by-any" ceiling drops if the
  model is removed (equals the unique-solve rate for the hard-solve matrix);
* **leave-one-out oracle** — the ceiling of the pool without it;
* **mean pairwise agreement** — how redundant its correctness pattern is vs the others;

plus pairwise **agreement**, **double-fault** (both wrong together), and **Cohen's
kappa** redundancy matrices, and the resulting **most-redundant model** + swap
recommendation.

It is a read-only diagnostic over already-graded correctness — it changes no scoring
or fitness math, and reuses the exact matrix schema of
:func:`trinity.analysis.agreement.to_oracle_matrix` /
``oracle_ceiling.matrix_to_tensor``. Pure numpy, no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "PerModelComplementarity",
    "ComplementaritySummary",
    "solve_matrix_from_matrix",
    "solve_matrix_from_records",
    "analyze_tensor",
    "analyze",
]

_EPS = 1e-12


@dataclass(frozen=True)
class PerModelComplementarity:
    """Complementarity profile of one model within the pool.

    Attributes:
        model: Model name.
        accuracy: Fraction of questions the model solves.
        unique_solves: Count of questions ONLY this model solves.
        unique_solve_rate: ``unique_solves / n_questions``.
        marginal_oracle_contribution: Drop in the solved-by-any ceiling if this model
            is removed (equals ``unique_solve_rate`` for a hard-solve matrix).
        leave_one_out_oracle: Solved-by-any ceiling of the pool WITHOUT this model.
        mean_pairwise_agreement: Mean fraction-agreement of this model's correctness
            pattern with each other model (higher = more redundant).
        is_most_redundant: True for the single recommended swap candidate.
    """

    model: str
    accuracy: float
    unique_solves: int
    unique_solve_rate: float
    marginal_oracle_contribution: float
    leave_one_out_oracle: float
    mean_pairwise_agreement: float
    is_most_redundant: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "accuracy": self.accuracy,
            "unique_solves": self.unique_solves,
            "unique_solve_rate": self.unique_solve_rate,
            "marginal_oracle_contribution": self.marginal_oracle_contribution,
            "leave_one_out_oracle": self.leave_one_out_oracle,
            "mean_pairwise_agreement": self.mean_pairwise_agreement,
            "is_most_redundant": self.is_most_redundant,
        }


@dataclass(frozen=True)
class ComplementaritySummary:
    """Pool-level complementarity audit.

    ``most_redundant_model`` is the swap candidate: the model whose removal costs the
    oracle the least (fewest unique solves / lowest marginal contribution), breaking
    ties toward the highest mean pairwise agreement — exactly the plan's criterion.
    """

    n_questions: int
    models: list[str]
    best_single_model: str | None
    best_single_accuracy: float
    oracle_any: float
    per_model: list[PerModelComplementarity]
    pairwise_agreement: dict[str, dict[str, float]]
    pairwise_double_fault: dict[str, dict[str, float]]
    cohen_kappa: dict[str, dict[str, float]]
    most_redundant_model: str | None
    swap_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, for the oracle report and CLI output."""
        return {
            "n_questions": self.n_questions,
            "models": list(self.models),
            "best_single_model": self.best_single_model,
            "best_single_accuracy": self.best_single_accuracy,
            "oracle_any": self.oracle_any,
            "per_model": [p.to_dict() for p in self.per_model],
            "pairwise_agreement": self.pairwise_agreement,
            "pairwise_double_fault": self.pairwise_double_fault,
            "cohen_kappa": self.cohen_kappa,
            "most_redundant_model": self.most_redundant_model,
            "swap_recommendation": self.swap_recommendation,
        }


def solve_matrix_from_matrix(matrix: dict, *, threshold: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """Decode an ``oracle_matrix`` dict into a hard-solve matrix ``B`` and model names.

    ``matrix["tasks"][i]["per_model"][model]`` is a length-K list of 0/1 samples (the
    schema of :func:`trinity.analysis.agreement.to_oracle_matrix`). Each cell is
    reduced to a solve probability ``p = mean(samples)`` and thresholded, so ``B`` has
    shape ``(n_questions, n_models)`` with entries in ``{0, 1}``.

    Raises:
        ValueError: If a ``(question, model)`` cell has a different model set than the
            first task (a ragged pool would bias every count).
    """
    tasks = matrix.get("tasks", [])
    if not tasks:
        return np.zeros((0, 0), dtype=int), []
    models = list(tasks[0]["per_model"].keys())
    Q, M = len(tasks), len(models)
    P = np.zeros((Q, M), dtype=float)
    for qi, t in enumerate(tasks):
        cells = t["per_model"]
        if list(cells.keys()) != models:
            raise ValueError(
                f"task {t.get('id', qi)!r} has models {list(cells.keys())}, expected {models}"
            )
        for mi, m in enumerate(models):
            cell = cells[m]
            P[qi, mi] = float(np.mean(cell)) if len(cell) else 0.0
    return (P >= threshold).astype(int), models


def solve_matrix_from_records(records: Sequence[Any]) -> tuple[np.ndarray, list[str]]:
    """Build a hard-solve matrix from ``QuestionAgreement``-like records.

    Each record must expose ``.models`` (sorted names) and ``.per_model_correct``
    (name -> 0/1); this is the shape emitted by
    :func:`trinity.analysis.agreement.grade_items`. Duck-typed so this module needs no
    import from ``agreement``.

    Raises:
        ValueError: If records disagree about the model set.
    """
    if not records:
        return np.zeros((0, 0), dtype=int), []
    models = list(records[0].models)
    B = np.zeros((len(records), len(models)), dtype=int)
    for qi, r in enumerate(records):
        if list(r.models) != models:
            raise ValueError(f"record {qi} has models {list(r.models)}, expected {models}")
        for mi, m in enumerate(models):
            B[qi, mi] = int(r.per_model_correct[m])
    return B, models


def _nested(mat: np.ndarray, models: list[str]) -> dict[str, dict[str, float]]:
    """A square numpy matrix -> ``{model_i: {model_j: value}}`` for JSON output."""
    return {mi: {mj: float(mat[i, j]) for j, mj in enumerate(models)} for i, mi in enumerate(models)}


def analyze_tensor(
    S: np.ndarray,
    models: Sequence[str],
    *,
    threshold: float = 0.5,
) -> ComplementaritySummary:
    """Compute the pool-complementarity audit from a solve tensor or matrix.

    Args:
        S: Either the ``(n_questions, n_models, K)`` solve tensor produced by
            ``oracle_ceiling.matrix_to_tensor`` (reduced to a per-cell solve
            probability via the K-mean) or an already-reduced ``(n_questions,
            n_models)`` probability/0-1 matrix.
        models: Model names, column order matching ``S``.
        threshold: A cell counts as solved when its solve probability is
            ``>= threshold`` (default 0.5, matching the oracle's hard-solve view).

    Returns:
        The :class:`ComplementaritySummary`.
    """
    arr = np.asarray(S, dtype=float)
    if arr.ndim == 3:
        p = arr.mean(axis=2)
    elif arr.ndim == 2:
        p = arr
    else:
        raise ValueError(f"S must be 2-D (Q,M) or 3-D (Q,M,K), got shape {arr.shape}")
    models = list(models)
    M = len(models)
    B = (p >= threshold).astype(int)
    Q = B.shape[0]

    if Q == 0 or M == 0:
        return ComplementaritySummary(
            n_questions=Q, models=models, best_single_model=None, best_single_accuracy=0.0,
            oracle_any=0.0, per_model=[], pairwise_agreement={}, pairwise_double_fault={},
            cohen_kappa={}, most_redundant_model=None,
            swap_recommendation="empty pool or no questions; nothing to audit",
        )

    accuracy = B.mean(axis=0)                        # (M,)
    row_sums = B.sum(axis=1)                          # (Q,) models solving each q
    oracle_any = float((row_sums >= 1).mean())

    # unique solves: questions where exactly one model solves, credited to that model.
    unique = ((B == 1) & (row_sums[:, None] == 1)).sum(axis=0)        # (M,)
    # leave-one-out oracle: solved-by-any of the pool without model m.
    loo_oracle = (row_sums[:, None] - B >= 1).mean(axis=0)            # (M,)
    marginal = oracle_any - loo_oracle                                # == unique / Q

    # pairwise agreement / double-fault / Cohen's kappa (M x M).
    agree = (B[:, :, None] == B[:, None, :]).mean(axis=0)             # (M,M)
    wrong = B == 0
    double_fault = (wrong[:, :, None] & wrong[:, None, :]).mean(axis=0)
    pe = np.outer(accuracy, accuracy) + np.outer(1.0 - accuracy, 1.0 - accuracy)
    denom = 1.0 - pe
    kappa = np.where(denom > _EPS, (agree - pe) / np.where(denom > _EPS, denom, 1.0), 0.0)
    np.fill_diagonal(kappa, 1.0)

    if M > 1:
        mean_agree = (agree.sum(axis=1) - 1.0) / (M - 1)             # exclude self (=1)
    else:
        mean_agree = np.zeros(M)

    # best single model (max accuracy, first on ties for determinism).
    best_idx = int(np.argmax(accuracy))
    # most redundant: fewest unique solves (lowest marginal), tie -> highest agreement.
    most_idx = min(range(M), key=lambda m: (round(float(marginal[m]), 12), -float(mean_agree[m])))

    per_model = [
        PerModelComplementarity(
            model=models[m],
            accuracy=float(accuracy[m]),
            unique_solves=int(unique[m]),
            unique_solve_rate=float(unique[m] / Q),
            marginal_oracle_contribution=float(marginal[m]),
            leave_one_out_oracle=float(loo_oracle[m]),
            mean_pairwise_agreement=float(mean_agree[m]),
            is_most_redundant=(m == most_idx and M > 1),
        )
        for m in range(M)
    ]

    if M > 1:
        rm = models[most_idx]
        rec = (
            f"Most redundant model: {rm} — lowest marginal oracle contribution "
            f"({marginal[most_idx]:.3f}: removing it drops the solved-by-any ceiling from "
            f"{oracle_any:.3f} to {loo_oracle[most_idx]:.3f}), {int(unique[most_idx])} unique "
            f"solve(s), mean pairwise agreement {mean_agree[most_idx]:.3f}. If the oracle verdict "
            f"is POOL_BOUND, swap {rm} for a model correct on a disjoint slice "
            f"(docs/ORACLE_CEILING_DIAGNOSTIC.md §6)."
        )
    else:
        rec = "single-model pool: no complementarity to audit"

    return ComplementaritySummary(
        n_questions=Q,
        models=models,
        best_single_model=models[best_idx],
        best_single_accuracy=float(accuracy[best_idx]),
        oracle_any=oracle_any,
        per_model=per_model,
        pairwise_agreement=_nested(agree, models),
        pairwise_double_fault=_nested(double_fault, models),
        cohen_kappa=_nested(kappa, models),
        most_redundant_model=(models[most_idx] if M > 1 else None),
        swap_recommendation=rec,
    )


def analyze(matrix: dict, *, threshold: float = 0.5) -> ComplementaritySummary:
    """Audit a pool from an ``oracle_matrix`` dict (see :func:`solve_matrix_from_matrix`)."""
    B, models = solve_matrix_from_matrix(matrix, threshold=threshold)
    return analyze_tensor(B, models, threshold=threshold)
