"""Per-model sampling-stability diagnostic: pass@1 / pass@K / majority@K.

Every consumer of ``oracle_matrix_<bench>.json`` collapses the K per-(query, model)
samples to a **cross-model** signal: ``complementarity`` / ``oracle_ceiling`` take
``p_hat = mean over K`` and then compare models. Nobody measures the **within-model,
across-K-samples** axis — for each model, how much does cheaply re-sampling it improve
accuracy (self-consistency / pass@k), and does the best single model's majority@K rival
the *routing* oracle?

That axis is decision-relevant. On the real ``oracle_matrix_math500.json`` (K=5) the best
single model's **majority@5 (~0.825) beats the trained router (0.792)** and closes about a
third of the routing headroom *for free, no routing* — directly informing the
ROUTER_BOUND-vs-sampling question ``docs/RESULTS.md`` §8/§9 debates but never quantifies.

Distinct from **#139 HERO** (a per-TURN training-fitness self-consistency term) and
**#238 ensemble** (a cross-MODEL answer plurality baseline): this is the per-model
over-K-samples axis, read-only over the 0/1 correctness matrix. Pure numpy over JSON — no
torch, no network, no GPU. (Meaningful only for K > 1; a K=1 matrix has pass@1 == pass@K
== majority@K and zero self-consistency gain.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ModelSampling", "SamplingSummary", "solve_counts", "analyze", "render"]


def solve_counts(matrix: dict) -> tuple[np.ndarray, int, list[str]]:
    """Decode an ``oracle_matrix`` dict into ``(solves[Q, M], K, models)``.

    ``solves[q, m]`` is the number of the model's K samples that were correct on query
    ``q`` (the schema is ``tasks[i]["per_model"][model] = [0/1, ...K]``). K must be uniform
    across every cell (a ragged K would be a collection bug, so it raises).

    Raises:
        ValueError: On a ragged model set or a ragged K.
    """
    tasks = matrix.get("tasks", [])
    if not tasks:
        return np.zeros((0, 0), dtype=int), 0, []
    models = list(tasks[0]["per_model"].keys())
    solves = np.zeros((len(tasks), len(models)), dtype=int)
    ks: set[int] = set()
    for qi, t in enumerate(tasks):
        cells = t["per_model"]
        if list(cells.keys()) != models:
            raise ValueError(f"task {t.get('id', qi)!r} has models {list(cells.keys())}, "
                             f"expected {models}")
        for mi, m in enumerate(models):
            cell = cells[m]
            ks.add(len(cell))
            solves[qi, mi] = int(sum(1 for v in cell if v))
    if len(ks) != 1:
        raise ValueError(f"ragged K across cells: {ks}")
    return solves, ks.pop(), models


@dataclass(frozen=True)
class ModelSampling:
    """Sampling-stability profile of one model over its K samples per question."""

    model: str
    pass_at_1: float          # expected single-sample accuracy (mean over K)
    pass_at_k: float          # any-of-K correct (self-consistency upper bound)
    majority_at_k: float      # strict-majority-of-K correct (self-consistency proxy)
    self_consistency_gain: float   # majority_at_k - pass_at_1

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "majority_at_k": self.majority_at_k,
            "self_consistency_gain": self.self_consistency_gain,
        }


@dataclass(frozen=True)
class SamplingSummary:
    """Per-model sampling stability + whether majority-voting the best single rivals routing."""

    benchmark: str
    n_questions: int
    k: int
    models: list[str]
    per_model: list[ModelSampling]
    best_pass1_model: str | None
    best_pass1: float
    best_majority_model: str | None
    best_majority: float
    routing_oracle: float
    majority_rivals_oracle: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_questions": self.n_questions,
            "k": self.k,
            "models": list(self.models),
            "per_model": [p.to_dict() for p in self.per_model],
            "best_pass1_model": self.best_pass1_model,
            "best_pass1": self.best_pass1,
            "best_majority_model": self.best_majority_model,
            "best_majority": self.best_majority,
            "routing_oracle": self.routing_oracle,
            "majority_rivals_oracle": self.majority_rivals_oracle,
        }


def analyze(matrix: dict, *, benchmark: str | None = None) -> SamplingSummary:
    """Per-model pass@1 / pass@K / majority@K from an ``oracle_matrix`` dict.

    ``routing_oracle`` is the cross-model per-query oracle (``mean_q max_m p_hat``) so the
    report can say whether cheaply re-sampling the best single model rivals the routing
    ceiling. Strict majority = ``solves >= K//2 + 1``.
    """
    solves, k, models = solve_counts(matrix)
    bench = str(benchmark or matrix.get("benchmark", "?"))
    q, m = (solves.shape[0], solves.shape[1]) if solves.ndim == 2 else (0, 0)
    if q == 0 or m == 0 or k == 0:
        return SamplingSummary(bench, 0, k, list(models), [], None, 0.0, None, 0.0, 0.0, False)

    p_hat = solves / k
    pass1 = p_hat.mean(axis=0)
    passk = (solves > 0).mean(axis=0)
    majk = (solves >= (k // 2 + 1)).mean(axis=0)
    routing_oracle = float(p_hat.max(axis=1).mean())

    per_model = [
        ModelSampling(models[i], float(pass1[i]), float(passk[i]), float(majk[i]),
                      float(majk[i] - pass1[i]))
        for i in range(m)
    ]
    bp1, bmaj = int(np.argmax(pass1)), int(np.argmax(majk))
    return SamplingSummary(
        benchmark=bench, n_questions=q, k=k, models=list(models), per_model=per_model,
        best_pass1_model=models[bp1], best_pass1=float(pass1[bp1]),
        best_majority_model=models[bmaj], best_majority=float(majk[bmaj]),
        routing_oracle=routing_oracle,
        majority_rivals_oracle=bool(majk[bmaj] >= routing_oracle - 1e-9),
    )


def render(summary: SamplingSummary) -> str:
    """Markdown: per-model pass@1/pass@K/majority@K + the sampling-vs-routing verdict."""
    out = ["# Per-model sampling stability (pass@1 / pass@K / majority@K)\n"]
    if summary.n_questions == 0:
        return "".join(out) + "\n_(no matrix data)_\n"

    out.append(f"n = {summary.n_questions} questions, K = {summary.k} samples/model\n")
    out.append("| model | pass@1 | pass@K | majority@K | self-consistency gain |")
    out.append("|---|---|---|---|---|")
    for p in summary.per_model:
        out.append(f"| {p.model} | {p.pass_at_1:.3f} | {p.pass_at_k:.3f} | "
                   f"{p.majority_at_k:.3f} | {p.self_consistency_gain:+.3f} |")
    out.append(f"\n- best single **pass@1** = {summary.best_pass1:.3f} ({summary.best_pass1_model})")
    out.append(f"- best single **majority@{summary.k}** = {summary.best_majority:.3f} "
               f"({summary.best_majority_model})")
    out.append(f"- routing oracle (per-query best model) = {summary.routing_oracle:.3f}")
    if summary.majority_rivals_oracle:
        out.append("\n**Verdict:** majority-voting the best single model RIVALS the routing "
                   "oracle — self-consistency is a lever comparable to routing (cheap, no router).")
    else:
        out.append("\n**Verdict:** the routing oracle still exceeds the best single model's "
                   "majority@K — routing headroom is not closed by self-consistency alone.")
    return "\n".join(out) + "\n"
