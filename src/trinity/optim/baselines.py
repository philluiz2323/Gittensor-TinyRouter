"""Baseline optimizers for the R8 comparison (docs/SPEC.md §5.4, milestone M4).

The headline replication claim R8 is ``sep-CMA-ES > SFT > RS > REINFORCE`` on all
four tasks (docs/SPEC.md L109, Table 4). To measure it we need the *baselines* the
coordinator is compared against, which the ``optim`` package advertises ("+ baseline
optimizers") but did not yet implement — only :mod:`trinity.optim.sep_cmaes` shipped.

This module implements the **Random Search (RS)** baseline, the self-contained,
GPU-free slice of that comparison:

    θ ~ U[−0.5, 0.5]^n , ``m_RS = 32`` trials/candidate, budget-matched to CMA-ES
    (docs/SPEC.md L420, L461; the ``random_search`` block in configs/trinity.yaml).

RS draws independent uniform parameter vectors, scores each on a freshly-sampled
minibatch, and keeps the single best — no distribution to adapt, no gradients. That
makes its core pure ``numpy`` (like :class:`~trinity.optim.sep_cmaes.SepCMAES` it
imports no torch and injects the fitness), so the sampler / budget-match / keep-best
logic is unit-testable offline exactly as ``sep_cmaes.run`` is.

SFT and REINFORCE (the other two R8 baselines, docs/SPEC.md L420) are intentionally
**not** implemented here: both are gradient methods that require torch on the head /
SLM and cannot be validated in a CPU-only environment. They are deferred to a
follow-up so this module stays fully offline-provable.

``RandomSearchTrainer`` subclasses :class:`~trinity.optim.base.BaseTrainer` and
returns the documented trainer summary; it reuses
:func:`trinity.optim.fitness.evaluate_population` so RS and sep-CMA-ES score
candidates through the exact same trajectory-evaluation path.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from trinity.optim.base import BaseTrainer
from trinity.optim.fitness import FitnessConfig, evaluate_population

# Random-Search baseline constants (docs/SPEC.md L420/L461 and the
# ``sep_cmaes.random_search`` block in configs/trinity.yaml).
RS_SAMPLE_LOW: float = -0.5
RS_SAMPLE_HIGH: float = 0.5
RS_TRIALS_PER_CANDIDATE: int = 32  # m_RS (docs/SPEC.md L453)


def budget_matched_candidates(
    popsize: int,
    m_cma: int,
    generations: int,
    trials_per_candidate: int = RS_TRIALS_PER_CANDIDATE,
) -> int:
    """Number of RS candidates whose env budget matches sep-CMA-ES.

    sep-CMA-ES spends ``popsize · m_cma · generations`` environment interactions
    (docs/SPEC.md ``budget_b_env`` = 33·16·60 = 31,680). RS spends
    ``num_candidates · trials_per_candidate``. Equating the two and flooring gives
    the budget-matched candidate count (31,680 / 32 = 990) so the R8 comparison is
    fair (docs/SPEC.md L461, "budget-matched").

    Args:
        popsize: sep-CMA-ES population size ``λ``.
        m_cma: Replications per CMA candidate.
        generations: Number of CMA generations ``T``.
        trials_per_candidate: Task instances scored per RS candidate (``m_RS``).

    Returns:
        The floored number of RS candidates that fit the same env budget.

    Raises:
        ValueError: If ``trials_per_candidate < 1`` or any count is negative.
    """
    if trials_per_candidate < 1:
        raise ValueError(f"trials_per_candidate must be >= 1, got {trials_per_candidate}")
    if min(popsize, m_cma, generations) < 0:
        raise ValueError("popsize, m_cma and generations must be non-negative")
    total_budget = int(popsize) * int(m_cma) * int(generations)
    return total_budget // int(trials_per_candidate)


def sample_candidates(
    n: int,
    num_candidates: int,
    *,
    seed: int = 0,
    low: float = RS_SAMPLE_LOW,
    high: float = RS_SAMPLE_HIGH,
) -> np.ndarray:
    """Draw ``num_candidates`` i.i.d. uniform parameter vectors ``θ ~ U[low, high]``.

    Uses ``numpy.random.default_rng(seed)`` so sampling is reproducible for every
    seed without touching global RNG state (unlike sep-CMA-ES, which must seed
    ``numpy.random`` for pycma). Two calls with the same ``seed`` return identical
    arrays; distinct seeds return distinct ones.

    Args:
        n: Parameter-vector dimension (TRINITY: ``spec.n_total`` = 13,312).
        num_candidates: Number of vectors to draw (rows).
        seed: RNG seed for reproducible sampling.
        low: Lower bound of the uniform range (docs/SPEC.md: −0.5).
        high: Upper bound of the uniform range (docs/SPEC.md: +0.5).

    Returns:
        Array of shape ``(num_candidates, n)`` with entries in ``[low, high)``.

    Raises:
        ValueError: If ``n < 1``, ``num_candidates < 1``, or ``low >= high``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
    if not low < high:
        raise ValueError(f"low ({low}) must be < high ({high})")
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(int(num_candidates), int(n)))


def select_best(candidates: np.ndarray, fitnesses: Any) -> tuple[np.ndarray, float, int]:
    """Return the highest-fitness candidate (RS keep-best over a fixed sample).

    Args:
        candidates: Array of shape ``(num_candidates, n)`` (or a stack of vectors).
        fitnesses: One scalar per candidate; larger is better.

    Returns:
        ``(best_theta, best_fitness, best_index)`` — a **copy** of the winning
        vector (so mutating it cannot corrupt ``candidates``), its fitness, and its
        row index. On ties the first (lowest-index) maximum wins.

    Raises:
        ValueError: If ``fitnesses`` is empty or lengths disagree.
    """
    cands = np.asarray(candidates, dtype=float)
    fits = np.asarray(fitnesses, dtype=float)
    if fits.size == 0:
        raise ValueError("fitnesses is empty; nothing to select")
    if cands.shape[0] != fits.shape[0]:
        raise ValueError(
            f"candidates ({cands.shape[0]}) and fitnesses ({fits.shape[0]}) length mismatch"
        )
    best_index = int(np.argmax(fits))
    return cands[best_index].copy(), float(fits[best_index]), best_index


def run_random_search(
    objective: Callable[[np.ndarray], float],
    n: int,
    *,
    num_candidates: int,
    seed: int = 0,
    low: float = RS_SAMPLE_LOW,
    high: float = RS_SAMPLE_HIGH,
    verbose: bool = False,
) -> tuple[np.ndarray, float, list[dict]]:
    """Run Random Search to **maximize** ``objective`` and log per-trial progress.

    Standalone, torch-free driver — the RS analogue of :func:`sep_cmaes.run`. Draws
    ``num_candidates`` uniform vectors and evaluates each once with ``objective``,
    tracking the best-so-far. Used by the offline smoke test / unit tests with a
    synthetic deterministic objective (no SLM, no pool).

    Args:
        objective: Callable mapping a vector of shape ``(n,)`` to a scalar fitness
            to be MAXIMIZED.
        n: Search-space dimension.
        num_candidates: Number of uniform draws to evaluate.
        seed: RNG seed for the draws (reproducible per seed).
        low: Lower bound of ``U[low, high]``.
        high: Upper bound of ``U[low, high]``.
        verbose: If True, print a one-line summary per trial.

    Returns:
        ``(best_x, best_f, history)`` where ``best_x`` is the best vector found,
        ``best_f`` its objective value, and ``history`` a list of per-trial dicts
        with keys ``{"trial", "fitness", "best_fitness"}`` (``best_fitness`` is the
        monotone non-decreasing best-so-far, suitable for a J-vs-trial curve).
    """
    candidates = sample_candidates(n, num_candidates, seed=seed, low=low, high=high)
    history: list[dict] = []
    best_x: np.ndarray | None = None
    best_f = -math.inf
    for i, x in enumerate(candidates):
        f = float(objective(x))
        if f > best_f:
            best_f = f
            best_x = np.asarray(x, dtype=float).copy()
        history.append({"trial": i, "fitness": f, "best_fitness": best_f})
        if verbose:
            print(f"[RS] trial {i + 1:4d}/{num_candidates} | f={f:+.4f} | best={best_f:+.4f}")
    assert best_x is not None  # num_candidates >= 1 guarantees at least one draw.
    return best_x, best_f, history


class RandomSearchTrainer(BaseTrainer):
    """Random-Search baseline trainer (docs/SPEC.md §5.4 R8; milestone M4).

    Samples ``num_candidates`` i.i.d. uniform parameter vectors ``θ ~ U[low, high]``,
    scores each on a freshly-sampled minibatch of ``trials_per_candidate`` tasks via
    :func:`trinity.optim.fitness.evaluate_population`, and keeps the best. Unlike
    sep-CMA-ES there is no distribution to adapt, so RS is a pure exploration
    baseline; its parameter-sampling core is offline-testable.

    RS scores candidates with the **plain binary** reward (accuracy) by design — the
    training-fitness shaping is a CMA-only aid, so a baseline must not use it. Pass an
    explicit ``fitness_cfg`` only to override this.
    """

    def __init__(
        self,
        *,
        trials_per_candidate: int = RS_TRIALS_PER_CANDIDATE,
        low: float = RS_SAMPLE_LOW,
        high: float = RS_SAMPLE_HIGH,
        seed: int = 0,
    ) -> None:
        """Configure the RS baseline.

        Args:
            trials_per_candidate: Task instances scored per candidate (``m_RS`` = 32).
            low: Lower bound of the uniform sampling range.
            high: Upper bound of the uniform sampling range.
            seed: Seed for candidate sampling and per-candidate minibatch draws.

        Raises:
            ValueError: If ``trials_per_candidate < 1`` or ``low >= high``.
        """
        if trials_per_candidate < 1:
            raise ValueError(f"trials_per_candidate must be >= 1, got {trials_per_candidate}")
        if not low < high:
            raise ValueError(f"low ({low}) must be < high ({high})")
        self.trials_per_candidate = int(trials_per_candidate)
        self.low = float(low)
        self.high = float(high)
        self.seed = int(seed)

    async def train(
        self,
        policy,
        pool,
        tasks: List[Any],
        *,
        spec: Any = None,
        pool_models: List[str] | None = None,
        num_candidates: int | None = None,
        popsize: int = 33,
        m_cma: int = 16,
        generations: int = 60,
        run_dir: str | Path | None = None,
        benchmark: str | None = None,
        fitness_cfg: FitnessConfig | None = None,
        max_turns: int = 5,
        sample: bool = True,
        on_candidate: Callable[[int, float, float], None] | None = None,
        **run_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the RS baseline and return the :class:`BaseTrainer` summary dict.

        Args:
            policy: A configured ``CoordinatorPolicy`` (encoder on GPU).
            pool: Async LLM pool exposing ``chat`` and a ``models`` mapping.
            tasks: Training tasks to minibatch-sample from.
            spec: Coordinator parameter spec; ``spec.n_total`` sets the search dim.
            pool_models: Pool model names in slot order; defaults to ``list(pool.models)``.
            num_candidates: RS draws to evaluate; defaults to the budget-matched
                count from ``popsize``/``m_cma``/``generations``.
            popsize, m_cma, generations: sep-CMA-ES budget parameters used only to
                derive the default budget-matched ``num_candidates``.
            run_dir: Directory for run artifacts (``best_theta.npy``, ``history.json``,
                ``summary.json``); created if absent.
            benchmark: Benchmark name for the summary; inferred from ``tasks`` if None.
            fitness_cfg: Reward config; ``None`` -> plain binary (correct RS default).
            max_turns: Session turn cap ``K``.
            sample: Whether the policy samples (vs. argmax) actions.
            on_candidate: Optional ``(i, fitness, elapsed_s)`` progress callback.
            **run_kwargs: Forwarded to the trajectory runner (max_tokens, reasoning,
                verifier_requires_prior_worker, ...).

        Returns:
            Summary dict with the ``BaseTrainer`` keys (``benchmark``, ``best_fitness``,
            ``best_theta_path``, ``run_dir``, ``total_cost_usd``) plus RS-specific
            metadata and the per-trial ``history``.

        Raises:
            ValueError: If ``spec`` or ``run_dir`` is missing, or the derived
                ``num_candidates`` is < 1.
        """
        if spec is None:
            raise ValueError("RandomSearchTrainer.train requires spec=<parameter spec>")
        if run_dir is None:
            raise ValueError("RandomSearchTrainer.train requires run_dir=<path>")
        # Lazy import keeps this module (and `import trinity.optim.baselines`) free of
        # any torch that the dataset/orchestration chain might pull at import time.
        from trinity.orchestration.dataset import sample_minibatch

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if pool_models is None:
            pool_models = list(pool.models)
        if benchmark is None:
            benchmark = getattr(tasks[0], "benchmark", "unknown") if tasks else "unknown"

        n = int(spec.n_total)
        if num_candidates is None:
            num_candidates = budget_matched_candidates(
                popsize, m_cma, generations, self.trials_per_candidate
            )
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")

        thetas = sample_candidates(n, num_candidates, seed=self.seed, low=self.low, high=self.high)

        history: list[dict] = []
        best_i = -1
        best_f = -math.inf

        def _on_cand(i: int, fit: float, elapsed: float) -> None:
            nonlocal best_i, best_f
            if fit > best_f:
                best_f = float(fit)
                best_i = int(i)
                np.save(run_dir / "best_theta.npy", thetas[i])
            history.append(
                {
                    "trial": int(i),
                    "fitness": float(fit),
                    "best_fitness": float(best_f),
                    "seconds": round(float(elapsed), 1),
                }
            )
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            if on_candidate is not None:
                on_candidate(i, fit, elapsed)

        def minibatch_fn(i: int) -> list:
            # Each candidate scores its own freshly-sampled minibatch (unbiased J),
            # seeded from the run seed + index so the whole run is reproducible.
            rng = random.Random(self.seed * 100000 + i)
            return sample_minibatch(tasks, self.trials_per_candidate, rng)

        fits = await evaluate_population(
            [thetas[i] for i in range(num_candidates)],
            spec,
            policy,
            pool,
            pool_models,
            minibatch_fn,
            sample=sample,
            on_candidate=_on_cand,
            fitness_cfg=fitness_cfg,
            max_turns=max_turns,
            **run_kwargs,
        )

        best_x, best_fitness, best_index = select_best(thetas, fits)
        np.save(run_dir / "best_theta.npy", best_x)

        summary: Dict[str, Any] = {
            "trainer": "random_search",
            "benchmark": benchmark,
            "pool": list(pool_models),
            "n_total": n,
            "num_candidates": int(num_candidates),
            "trials_per_candidate": self.trials_per_candidate,
            "sample_range": [self.low, self.high],
            "best_fitness": float(best_fitness),
            "best_trial": int(best_index),
            "best_theta_path": str(run_dir / "best_theta.npy"),
            "run_dir": str(run_dir),
            "seed": self.seed,
            "total_cost_usd": float(getattr(pool, "total_cost_usd", 0.0)),
            "history": history,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


if __name__ == "__main__":
    # Smoke test: RS on a synthetic deterministic objective at the real TRINITY
    # dimension. Confirms best-so-far is monotone and the budget-match arithmetic.
    _N = 13312
    _rng = np.random.default_rng(0)
    _target = _rng.standard_normal(_N) * 0.05

    def _sphere(x: np.ndarray) -> float:
        """Negative squared distance to a target (maximized at ``_target``)."""
        d = x - _target
        return -float(np.dot(d, d))

    _bx, _bf, _hist = run_random_search(_sphere, _N, num_candidates=200, seed=0)
    print(f"RS best fitness = {_bf:+.6f} over {len(_hist)} trials")
    print(
        "best_fitness monotone non-decreasing: "
        f"{all(_hist[i]['best_fitness'] <= _hist[i + 1]['best_fitness'] + 1e-12 for i in range(len(_hist) - 1))}"
    )
    print(f"budget-matched candidates (λ=33, m_cma=16, T=60, m_RS=32) = {budget_matched_candidates(33, 16, 60, 32)}")
