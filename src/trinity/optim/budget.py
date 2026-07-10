"""Atomic-eval (B_env) budget planning for sep-CMA-ES training (docs/SPEC.md §5.2).

SPEC §5.2 is explicit: *"B_env counts individual Bernoulli calls. CMA cost/iteration
= m_CMA·λ; T = ⌊B_env / (m_CMA·λ)⌋."* and the definition of done requires the run to
finish *"within the atomic-eval budget"*. Yet ``trinity.train`` only **printed** the
naive forward product ``λ·m_cma·generations`` — it never planned the number of
generations from a budget, nor tracked or enforced consumption. ``configs/trinity.yaml``
even carries ``sep_cmaes.budget_b_env: 31680`` that nothing read.

This module supplies the missing planner + tracker:

* :func:`plan_generations` — ``T = ⌊B_env / (m_CMA·λ)⌋``, the generations a budget affords.
* :func:`cmaes_budget` — the inverse: atomic evals a full ``T``-generation run spends.
* :class:`AtomicEvalBudget` — a live consumption tracker (cost/generation = ``m_CMA·λ``),
  with ``remaining`` / ``fraction_used`` / ``exhausted`` so the loop can stop on budget.

Pure arithmetic — no torch, no numpy, CPU only. ``trinity.train`` uses it opt-in via
``--budget``; default off, so existing runs are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def plan_generations(b_env: int, m_cma: int, popsize: int) -> int:
    """Generations affordable under an atomic-eval budget: ``T = ⌊B_env / (m_cma·λ)⌋``.

    Args:
        b_env: Total atomic-eval (Bernoulli) budget.
        m_cma: Replications per candidate (inner runs).
        popsize: sep-CMA-ES population size ``λ``.

    Returns:
        The floored number of generations ``T`` (0 if the budget can't fund one).

    Raises:
        ValueError: If ``b_env < 0`` or ``m_cma``/``popsize`` is not positive.
    """
    if b_env < 0:
        raise ValueError(f"b_env must be >= 0, got {b_env}")
    if m_cma < 1 or popsize < 1:
        raise ValueError(f"m_cma and popsize must be >= 1, got m_cma={m_cma}, popsize={popsize}")
    return int(b_env) // (int(m_cma) * int(popsize))


def cmaes_budget(popsize: int, m_cma: int, generations: int) -> int:
    """Atomic evals a full ``generations``-generation sep-CMA-ES run spends: ``λ·m_cma·T``.

    The inverse of :func:`plan_generations` (up to the floor):
    ``plan_generations(cmaes_budget(λ, m, T), m, λ) == T``.

    Raises:
        ValueError: If any argument is negative.
    """
    if min(popsize, m_cma, generations) < 0:
        raise ValueError("popsize, m_cma and generations must be non-negative")
    return int(popsize) * int(m_cma) * int(generations)


@dataclass
class AtomicEvalBudget:
    """Tracks atomic-eval (Bernoulli) consumption against a fixed ``B_env`` cap.

    One sep-CMA-ES generation costs :attr:`cost_per_generation` = ``m_cma·λ`` atomic
    evals (the population evaluation; per SPEC §5.2 the val-holdout diagnostic is a
    separate cost and is not counted here). Call :meth:`record_generation` after each
    generation and stop when :attr:`exhausted`.

    Attributes:
        b_env: Total atomic-eval budget.
        m_cma: Replications per candidate.
        popsize: Population size ``λ``.
        consumed: Atomic evals spent so far.
    """

    b_env: int
    m_cma: int
    popsize: int
    consumed: int = 0

    def __post_init__(self) -> None:
        if self.b_env < 0:
            raise ValueError(f"b_env must be >= 0, got {self.b_env}")
        if self.m_cma < 1 or self.popsize < 1:
            raise ValueError(
                f"m_cma and popsize must be >= 1, got m_cma={self.m_cma}, popsize={self.popsize}"
            )
        if self.consumed < 0:
            raise ValueError(f"consumed must be >= 0, got {self.consumed}")
        self.b_env = int(self.b_env)
        self.m_cma = int(self.m_cma)
        self.popsize = int(self.popsize)
        self.consumed = int(self.consumed)

    @property
    def cost_per_generation(self) -> int:
        """Atomic evals one generation spends (``m_cma·λ``)."""
        return self.m_cma * self.popsize

    @property
    def max_generations(self) -> int:
        """Generations the full budget affords (``⌊B_env / (m_cma·λ)⌋``)."""
        return plan_generations(self.b_env, self.m_cma, self.popsize)

    @property
    def remaining(self) -> int:
        """Atomic evals left (never negative)."""
        return max(0, self.b_env - self.consumed)

    @property
    def fraction_used(self) -> float:
        """Share of the budget consumed in ``[0, 1]`` (1.0 when ``b_env`` is 0)."""
        return self.consumed / self.b_env if self.b_env > 0 else 1.0

    @property
    def can_afford_generation(self) -> bool:
        """Whether at least one more full generation fits in the remaining budget."""
        return self.remaining >= self.cost_per_generation

    @property
    def exhausted(self) -> bool:
        """True once another full generation can no longer be afforded."""
        return not self.can_afford_generation

    def record_generation(self, n: int = 1) -> int:
        """Charge ``n`` generations to the budget; return the new ``consumed`` total.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.consumed += int(n) * self.cost_per_generation
        return self.consumed

    def report(self) -> dict[str, Any]:
        """JSON-serializable snapshot, for ``summary.json``."""
        return {
            "b_env": self.b_env,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "cost_per_generation": self.cost_per_generation,
            "max_generations": self.max_generations,
            "fraction_used": self.fraction_used,
            "exhausted": self.exhausted,
        }
