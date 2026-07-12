"""Project the API spend of a sep-CMA-ES head-training run, before launching it.

Why this exists
---------------
SUBMITTING.md says a full training run costs "~$25-65" and the receipt gate
(``submission.gates`` / ``MIN_TRAINING_COST_USD``) rejects a submission whose
receipt shows less than $15 of spend. But nothing turns the *config* into that
dollar figure ahead of time:

* ``optim.budget.cmaes_budget`` gives the run's size in **atomic evaluations**
  (``λ·m_cma·generations``) — a count, not a cost;
* ``fugu.cost.estimate_grpo_cost`` prices the **GRPO Conductor** path, which is a
  different training loop.

This module bridges them: it takes the sep-CMA-ES config knobs (population,
m_cma, generations, turns), prices each atomic evaluation from the worker pool,
and returns the projected USD spend with a per-model breakdown — so a paid run is
never launched blind, and a contributor can see up front whether their planned
run will even clear the $15 receipt floor.

Pure / deterministic / no network / no GPU / no torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from trinity.optim.budget import cmaes_budget

__all__ = ["TrainCostEstimate", "estimate_cmaes_cost"]

# The receipt gate floor, imported so the estimate can flag a too-cheap plan.
try:
    from trinity.submission.constants import MIN_TRAINING_COST_USD
except Exception:  # pragma: no cover - constants module optional in some envs
    MIN_TRAINING_COST_USD = 15.0


@dataclass(frozen=True)
class TrainCostEstimate:
    """Projected spend of a sep-CMA-ES training run.

    ``atomic_evals`` is ``λ·m_cma·generations`` (one atomic eval == one scored
    trajectory); each trajectory makes ``avg_turns`` worker calls, so
    ``worker_calls = atomic_evals * avg_turns``. ``total_usd`` prices those calls
    at the blended worker token price.
    """

    atomic_evals: int
    worker_calls: int
    total_usd: float
    per_model_usd: dict[str, float]
    below_receipt_floor: bool
    assumptions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "atomic_evals": self.atomic_evals,
            "worker_calls": self.worker_calls,
            "total_usd": self.total_usd,
            "per_model_usd": dict(self.per_model_usd),
            "min_receipt_usd": MIN_TRAINING_COST_USD,
            "below_receipt_floor": self.below_receipt_floor,
            "assumptions": dict(self.assumptions),
        }


def estimate_cmaes_cost(
    *,
    population_size: int,
    m_cma: int,
    generations: int,
    worker_names: Sequence[str],
    prices: Mapping[str, tuple[float, float]],
    avg_turns: float = 2.5,
    avg_prompt_tokens: int = 1200,
    avg_completion_tokens: int = 800,
) -> TrainCostEstimate:
    """Project the USD spend of a ``generations``-generation sep-CMA-ES run.

    Args:
        population_size: sep-CMA-ES population ``λ``.
        m_cma: Replications per candidate (inner runs).
        generations: Number of generations ``T``.
        worker_names: The worker pool the trajectories route over; their prices are
            averaged into a blended per-call cost.
        prices: ``model -> (price_in, price_out)`` in USD per 1M tokens. An unknown
            worker prices at ``(0, 0)`` and is surfaced in the breakdown.
        avg_turns: Mean worker calls per scored trajectory.
        avg_prompt_tokens / avg_completion_tokens: Mean tokens per worker call.

    Returns:
        A :class:`TrainCostEstimate`.

    Raises:
        ValueError: If any of ``population_size`` / ``m_cma`` / ``generations`` is
            negative, or ``avg_turns`` is negative (delegated to ``cmaes_budget``
            for the counts).
    """
    if avg_turns < 0:
        raise ValueError(f"avg_turns must be >= 0; got {avg_turns}")

    atomic = cmaes_budget(population_size, m_cma, generations)
    worker_calls = int(round(atomic * avg_turns))

    names = list(worker_names) or [""]
    # Cost of ONE worker call at each model's price, then averaged over the pool
    # (a trajectory routes across the pool, so the blended price is the estimator).
    per_call: dict[str, float] = {}
    for name in names:
        pin, pout = prices.get(name, (0.0, 0.0))
        per_call[name] = (avg_prompt_tokens / 1e6 * pin
                          + avg_completion_tokens / 1e6 * pout)

    calls_per_model = worker_calls / len(names)
    per_model_usd = {name: round(calls_per_model * c, 4) for name, c in per_call.items()}
    total = round(sum(per_model_usd.values()), 2)

    return TrainCostEstimate(
        atomic_evals=atomic,
        worker_calls=worker_calls,
        total_usd=total,
        per_model_usd=per_model_usd,
        below_receipt_floor=total < MIN_TRAINING_COST_USD,
        assumptions={
            "population_size": population_size, "m_cma": m_cma,
            "generations": generations, "avg_turns": avg_turns,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "workers": names,
        },
    )
