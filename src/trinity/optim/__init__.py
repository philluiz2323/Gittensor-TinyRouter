"""Evolutionary training (separable CMA-ES) + baseline optimizers."""
from __future__ import annotations

from trinity.optim.baselines import (
    RandomSearchTrainer,
    budget_matched_candidates,
    run_random_search,
)
from trinity.optim.budget import AtomicEvalBudget, cmaes_budget, plan_generations
from trinity.optim.reinforce import (
    MovingBaseline,
    REINFORCETrainer,
    budget_matched_batch,
    run_reinforce,
)
from trinity.optim.sep_cmaes import SepCMAES, default_popsize, run
from trinity.optim.sft import SFTTrainer, build_teacher_targets, fit_head_sft

__all__ = [
    "SepCMAES",
    "default_popsize",
    "run",
    "RandomSearchTrainer",
    "budget_matched_candidates",
    "run_random_search",
    "SFTTrainer",
    "build_teacher_targets",
    "fit_head_sft",
    "REINFORCETrainer",
    "MovingBaseline",
    "budget_matched_batch",
    "run_reinforce",
    "AtomicEvalBudget",
    "cmaes_budget",
    "plan_generations",
]
