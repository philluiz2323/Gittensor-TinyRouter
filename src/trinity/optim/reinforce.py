"""REINFORCE baseline trainer — docs/SPEC.md L420/L462, R8, milestone M4.

SPEC §1.3 **R8** claims ``sep-CMA-ES > SFT > RS > REINFORCE on all 4 tasks`` (Table 4), and
§9 pins REINFORCE's budget: ``batch = m_CMA·λ`` with 60 iterations (``configs/trinity.yaml``
ships exactly ``reinforce: {batch_size: 528, iterations: 60}``, and 16·33 = 528). As with
SFT, the consumers of that ordering are merged — :data:`trinity.analysis.convergence.
R8_EXPECTED_ORDER` names ``"reinforce"`` — while nothing could produce a REINFORCE run.

**The learning rule.** REINFORCE is the score-function policy gradient: sample an action
from the current policy, then push its log-probability up or down in proportion to how much
better than average its reward was::

    ∇_W J  =  E[ (r − b) · ∇_W log π(a | h) ]

with ``b`` an exponential-moving-average baseline carried in from previous iterations.
Because ``b`` is a function of past batches only, it is independent of the action being
scored and so subtracts nothing in expectation — it leaves the gradient unbiased while
removing the bulk of its variance, which is what makes REINFORCE usable on a reward this
noisy at all.

**Why the one-step (contextual-bandit) reduction, stated plainly.** A faithful multi-turn
REINFORCE needs ``∇log π(a_k | h_k)`` at *every* turn, i.e. the penultimate-token hidden
state ``h_k`` that produced each decision. :class:`trinity.types.TurnRecord` records the
action taken (``agent_name``, ``role``) but **not** ``h_k``, so multi-turn credit assignment
would require threading hidden states through ``orchestration/session.py`` — a change to the
live rollout path, which this baseline has no business making. Instead this module trains on
the *same* frozen encodings and measured solve rates that :mod:`trinity.optim.sft` uses:
each query is one contextual-bandit step where the policy picks a pool model and the reward
is that model's measured solve rate. That is a genuine policy gradient (sampled actions,
reward-weighted log-probs, variance-reduction baseline) on real data, it costs nothing to
run, and — crucially for R8 — it differs from SFT in *exactly one respect*, the learning
rule, since both consume identical inputs. Extending to multi-turn credit assignment is a
follow-up gated on recording per-turn hidden states.

The contrast this makes visible is the point of R8: SFT imitates a teacher distribution
built from *all* models' solve rates, so every query informs every row of the head. REINFORCE
only ever observes the reward of the single action it sampled, so it must discover the same
signal through far noisier, higher-variance updates — which is why the SPEC expects it to
finish last.

torch is imported **lazily inside the update functions**, so ``import trinity.optim.reinforce``
stays torch-free at module scope (the invariant ``trinity/optim/baselines.py`` documents and
``tests/test_shaped_fitness.py::test_no_torch_imported`` enforces). The baseline, sampling
and budget-arithmetic layers are pure numpy and unit-test without torch at all.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from trinity.optim.base import BaseTrainer

__all__ = [
    "REINFORCE_BATCH_SIZE",
    "REINFORCE_ITERATIONS",
    "REINFORCE_LR",
    "REINFORCE_BASELINE_DECAY",
    "budget_matched_batch",
    "MovingBaseline",
    "sample_actions",
    "run_reinforce",
    "REINFORCETrainer",
]

#: Batch size, docs/SPEC.md L420/L462: ``batch = m_CMA·λ`` = 16·33 = 528. Mirrors the
#: ``baselines.reinforce.batch_size`` entry in configs/trinity.yaml.
REINFORCE_BATCH_SIZE: int = 528
#: Iterations, docs/SPEC.md L462 and ``baselines.reinforce.iterations``.
REINFORCE_ITERATIONS: int = 60
#: Adam learning rate. Not pinned by the SPEC (which fixes only batch/iterations); set well
#: above the SFT rate because a score-function gradient is far noisier than a supervised one.
REINFORCE_LR: float = 1.0e-3
#: EMA decay for the reward baseline: ``b <- decay·b + (1 - decay)·mean(batch reward)``.
REINFORCE_BASELINE_DECAY: float = 0.9


def budget_matched_batch(popsize: int, m_cma: int) -> int:
    """Environment-interactions-per-iteration that match sep-CMA-ES (SPEC L420).

    sep-CMA-ES spends ``λ · m_cma`` env interactions per generation, so REINFORCE draws the
    same number of samples per iteration and the R8 comparison is budget-fair. With the
    TRINITY defaults (λ=33, m_cma=16) this is 528 — the value pinned in trinity.yaml.

    Args:
        popsize: sep-CMA-ES population size ``λ``.
        m_cma: Replications per CMA candidate.

    Returns:
        ``popsize * m_cma``.

    Raises:
        ValueError: If either argument is < 1.
    """
    if popsize < 1:
        raise ValueError(f"popsize must be >= 1, got {popsize}")
    if m_cma < 1:
        raise ValueError(f"m_cma must be >= 1, got {m_cma}")
    return int(popsize) * int(m_cma)


class MovingBaseline:
    """Exponential-moving-average reward baseline for policy-gradient variance reduction.

    Tracks ``b <- decay·b + (1 - decay)·x``, initialized on the first observation so the
    estimate is not dragged from an arbitrary zero (a cold start at 0 would make every early
    advantage look large and positive on a non-negative reward). Pure numpy/stdlib.

    Attributes:
        decay: EMA decay in ``[0, 1)``; larger = smoother, slower to adapt.
        value: Current baseline estimate; ``0.0`` before the first :meth:`update`.
        count: Number of updates applied so far.
    """

    def __init__(self, decay: float = REINFORCE_BASELINE_DECAY) -> None:
        """Create a baseline.

        Args:
            decay: EMA decay, must satisfy ``0 <= decay < 1``.

        Raises:
            ValueError: If ``decay`` is outside ``[0, 1)``.
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.value = 0.0
        self.count = 0

    def update(self, reward: float) -> float:
        """Fold one reward observation into the baseline and return the new estimate.

        Args:
            reward: Scalar reward (typically a batch mean).

        Returns:
            The updated baseline value.
        """
        r = float(reward)
        self.value = r if self.count == 0 else self.decay * self.value + (1.0 - self.decay) * r
        self.count += 1
        return self.value


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one categorical action per row of ``probs``.

    Vectorized inverse-CDF sampling: a single uniform per row against the row's cumulative
    distribution. Rows are renormalized first so minor float drift cannot make the last
    category unreachable.

    Args:
        probs: ``(N, n_actions)`` row-stochastic probabilities.
        rng: NumPy generator supplying the uniforms (seeded by the caller).

    Returns:
        int64 array of shape ``(N,)`` with entries in ``[0, n_actions)``.

    Raises:
        ValueError: If ``probs`` is not 2-D or has a non-positive row sum.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, n_actions); got shape {p.shape}")
    row_sum = p.sum(axis=1, keepdims=True)
    if not np.all(row_sum > 0):
        raise ValueError("every row of probs must sum to a positive value")
    cdf = np.cumsum(p / row_sum, axis=1)
    u = rng.random((p.shape[0], 1))
    # clip guards the case where floating-point cumsum ends just below the drawn uniform.
    return np.clip((u > cdf).sum(axis=1), 0, p.shape[1] - 1).astype(np.int64)


def run_reinforce(
    features: np.ndarray,
    solve_prob: np.ndarray,
    *,
    batch_size: int = REINFORCE_BATCH_SIZE,
    iterations: int = REINFORCE_ITERATIONS,
    lr: float = REINFORCE_LR,
    baseline_decay: float = REINFORCE_BASELINE_DECAY,
    seed: int = 0,
    device: str = "cpu",
    return_history: bool = True,
):
    """Train agent-selection head rows by REINFORCE on the one-step routing bandit.

    Each iteration draws ``batch_size`` queries with replacement, samples one pool model per
    query from the head's current agent softmax, collects the measured solve rate as reward,
    and applies the score-function update ``-(r − b)·log π(a|h)`` through Adam. Drawing
    **with** replacement is deliberate: it keeps the per-iteration env spend exactly
    ``batch_size`` regardless of how many labelled queries exist, which is what makes the
    budget match against sep-CMA-ES (:func:`budget_matched_batch`) exact.

    Args:
        features: ``(N, d_h)`` frozen query encodings.
        solve_prob: ``(N, n_models)`` measured solve rate per (query, model), the reward table.
        batch_size: Samples per iteration (SPEC: 528).
        iterations: Number of gradient iterations (SPEC: 60).
        lr: Adam learning rate.
        baseline_decay: EMA decay for the reward baseline.
        seed: Seed for query draws, action sampling and init.
        device: torch device string.
        return_history: Return per-iteration records alongside the weights.

    Returns:
        ``(W_agent, history)`` when ``return_history`` else ``W_agent``. ``W_agent`` is
        ``(n_models, d_h)`` float64; history entries are
        ``{"iteration", "mean_reward", "baseline", "mean_advantage", "loss"}``.

    Raises:
        ValueError: On shape disagreement or non-positive hyperparameters.
    """
    # Lazy import: keeps `import trinity.optim.reinforce` torch-free at module scope.
    import torch

    H = np.ascontiguousarray(features, dtype=np.float64)
    R = np.ascontiguousarray(solve_prob, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"features must be 2-D (N, d_h); got shape {H.shape}")
    if R.ndim != 2:
        raise ValueError(f"solve_prob must be 2-D (N, n_models); got shape {R.shape}")
    if H.shape[0] != R.shape[0]:
        raise ValueError(f"features has {H.shape[0]} rows but solve_prob has {R.shape[0]}")
    if H.shape[0] == 0:
        raise ValueError("features is empty; nothing to train on")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    n_rows, d_h = H.shape
    n_models = R.shape[1]

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    feats = torch.tensor(H, dtype=torch.float64, device=device, requires_grad=False)
    W = torch.nn.Parameter(
        torch.randn(n_models, d_h, dtype=torch.float64, device=device) * 1e-3
    )
    opt = torch.optim.Adam([W], lr=float(lr))
    baseline = MovingBaseline(baseline_decay)

    history: list[dict] = []
    for it in range(int(iterations)):
        idx = rng.integers(0, n_rows, size=int(batch_size))
        sel = torch.as_tensor(idx, dtype=torch.long, device=device)
        batch_feats = feats.index_select(0, sel)

        logits = batch_feats @ W.t()
        log_probs = torch.log_softmax(logits, dim=-1)
        # Actions are sampled from the CURRENT policy, detached: the score-function
        # estimator differentiates log π(a|h) w.r.t. W, never the sampling operation.
        probs_np = torch.softmax(logits, dim=-1).detach().to("cpu").numpy()
        actions = sample_actions(probs_np, rng)
        rewards = R[idx, actions]

        mean_reward = float(np.mean(rewards))
        # Subtract the baseline carried in from PREVIOUS iterations, then fold this batch
        # in afterwards. Using the post-update value would leak this batch's sampled
        # actions into its own baseline, making the "b is action-independent" property
        # (and hence unbiasedness) hold only up to an O(1/batch) term. Iteration 0 has no
        # history, so it falls back to the batch mean — a zero-mean, neutral first step.
        b = baseline.value if baseline.count > 0 else mean_reward
        advantages = rewards - b
        baseline.update(mean_reward)

        act_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float64, device=device)
        chosen_log_probs = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
        # Negated because Adam minimizes while REINFORCE ascends the reward gradient.
        loss = -(adv_t * chosen_log_probs).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if return_history:
            history.append(
                {
                    "iteration": it,
                    "mean_reward": mean_reward,
                    "baseline": float(b),
                    "mean_advantage": float(np.mean(advantages)),
                    "loss": float(loss.detach()),
                }
            )

    W_out = W.detach().to("cpu").numpy().astype(np.float64)
    return (W_out, history) if return_history else W_out


class REINFORCETrainer(BaseTrainer):
    """REINFORCE baseline trainer (docs/SPEC.md L420/L462; R8, milestone M4).

    Trains the head's agent rows by policy gradient on the one-step routing bandit built
    from an oracle-ceiling matrix, then packs the result into a full-length θ via
    :func:`trinity.coordinator.warmstart.pack_warmstart_theta`.

    Like :class:`~trinity.optim.sft.SFTTrainer` this consumes previously-measured labels, so
    a run makes no API calls and ``total_cost_usd`` is 0.0. It differs from SFT only in the
    learning rule, which is precisely the comparison R8 asks for.
    """

    def __init__(
        self,
        *,
        batch_size: int = REINFORCE_BATCH_SIZE,
        iterations: int = REINFORCE_ITERATIONS,
        lr: float = REINFORCE_LR,
        baseline_decay: float = REINFORCE_BASELINE_DECAY,
        seed: int = 0,
    ) -> None:
        """Configure the REINFORCE baseline.

        Args:
            batch_size: Samples per iteration (SPEC default 528 = λ·m_cma).
            iterations: Gradient iterations (SPEC default 60).
            lr: Adam learning rate.
            baseline_decay: EMA decay for the reward baseline.
            seed: Seed for draws, sampling and init.

        Raises:
            ValueError: If any hyperparameter is out of range.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if not lr > 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= baseline_decay < 1.0:
            raise ValueError(f"baseline_decay must be in [0, 1), got {baseline_decay}")
        self.batch_size = int(batch_size)
        self.iterations = int(iterations)
        self.lr = float(lr)
        self.baseline_decay = float(baseline_decay)
        self.seed = int(seed)

    async def train(
        self,
        policy,
        pool,
        tasks: List[Any],
        *,
        spec: Any = None,
        features: Any = None,
        matrix_path: str | Path | None = None,
        solve_prob: Any = None,
        pool_models: List[str] | None = None,
        run_dir: str | Path | None = None,
        benchmark: str | None = None,
        device: str = "cpu",
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """Run the REINFORCE baseline and return the :class:`BaseTrainer` summary dict.

        Args:
            policy: Unused — training happens on frozen precomputed features, so the
                trainer never routes. Accepted to satisfy the :class:`BaseTrainer` signature.
            pool: Unused for the same reason; read only for ``total_cost_usd``.
            tasks: Unused; rewards come from ``solve_prob`` / ``matrix_path``.
            spec: Coordinator parameter spec (``spec.n_total`` sets the θ length).
            features: ``(N, d_h)`` frozen encodings, aligned row-wise with the labels.
            matrix_path: ``oracle_matrix_<bench>.json`` to read rewards from, via
                :func:`trinity.coordinator.warmstart.load_labels`.
            solve_prob: ``(N, n_models)`` solve rates, as an alternative to ``matrix_path``.
            pool_models: Model names in slot order; defaults to the matrix's model order.
            run_dir: Directory for ``best_theta.npy`` / ``history.json`` / ``summary.json``.
            benchmark: Benchmark label for the summary.
            device: torch device for the updates.

        Returns:
            Summary dict with the :class:`BaseTrainer` keys plus REINFORCE metadata and the
            per-iteration ``history``.

        Raises:
            ValueError: If ``spec``/``run_dir``/``features`` is missing, if neither
                ``solve_prob`` nor ``matrix_path`` is given, or if shapes disagree.
        """
        if spec is None:
            raise ValueError("REINFORCETrainer.train requires spec=<parameter spec>")
        if run_dir is None:
            raise ValueError("REINFORCETrainer.train requires run_dir=<path>")
        if features is None:
            raise ValueError(
                "REINFORCETrainer.train requires features=<(N, d_h) frozen encodings>; the "
                "SLM is frozen, so encode once with coordinator.warmstart.encode_queries "
                "and cache it"
            )
        # Reuse the canonical loader so this cannot drift from the on-disk matrix schema.
        from trinity.coordinator.warmstart import load_labels, pack_warmstart_theta

        if solve_prob is None:
            if matrix_path is None:
                raise ValueError(
                    "REINFORCETrainer.train requires either solve_prob=<(N, n_models)> or "
                    "matrix_path=<oracle_matrix_*.json>"
                )
            _qids, solve_prob, matrix_models = load_labels(str(matrix_path))
            if pool_models is None:
                pool_models = list(matrix_models)

        sp = np.asarray(solve_prob, dtype=float)
        H = np.asarray(features, dtype=float)
        if sp.ndim != 2:
            raise ValueError(f"solve_prob must be 2-D (N, n_models); got shape {sp.shape}")
        if H.ndim != 2:
            raise ValueError(f"features must be 2-D (N, d_h); got shape {H.shape}")
        if H.shape[0] != sp.shape[0]:
            raise ValueError(
                f"features has {H.shape[0]} rows but solve_prob has {sp.shape[0]}; the "
                "encodings and labels must be aligned row-wise (same query order)"
            )

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if pool_models is None:
            pool_models = [f"model_{i}" for i in range(sp.shape[1])]
        if benchmark is None:
            benchmark = getattr(tasks[0], "benchmark", "unknown") if tasks else "unknown"

        W_agent, history = run_reinforce(
            H,
            sp,
            batch_size=self.batch_size,
            iterations=self.iterations,
            lr=self.lr,
            baseline_decay=self.baseline_decay,
            seed=self.seed,
            device=device,
            return_history=True,
        )
        theta = pack_warmstart_theta(W_agent, spec)
        np.save(run_dir / "best_theta.npy", theta)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        # The achieved objective is the sampled routing reward, which IS a fitness on the
        # same 0-1 scale as the other R8 optimizers -- unlike SFT's cross-entropy. Report the
        # final iteration's mean reward rather than the best-ever, so a lucky early batch
        # cannot flatter the baseline.
        final_reward = float(history[-1]["mean_reward"]) if history else math.nan
        summary: Dict[str, Any] = {
            "trainer": "reinforce",
            "benchmark": benchmark,
            "pool": list(pool_models),
            "n_total": int(spec.n_total),
            "n_labelled": int(sp.shape[0]),
            "optimizer": "adam",
            "lr": self.lr,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "baseline_decay": self.baseline_decay,
            "env_interactions": self.batch_size * self.iterations,
            "final_mean_reward": final_reward,
            "best_fitness": final_reward,
            "objective": "sampled_routing_reward",
            "best_theta_path": str(run_dir / "best_theta.npy"),
            "run_dir": str(run_dir),
            "seed": self.seed,
            "total_cost_usd": float(getattr(pool, "total_cost_usd", 0.0)),
            "history": history,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
