"""Fitness evaluation for sep-CMA-ES candidates.

Fitness(θ) = mean reward R(τ) over a minibatch of `m_cma` task instances, each
run through the coordination loop with the policy configured by θ.

By default the reward is the single terminal binary correctness signal and the
candidate fitness is its plain mean — exactly the original behavior. A
``fitness:`` block in ``configs/trinity.yaml`` can optionally turn on two
TRAINING-ONLY shaping mechanisms (see :class:`FitnessConfig`):

  * **shaped_reward** — a denser per-trajectory scalar that keeps correctness as
    the dominant term but adds a small format bonus (answer is parseable) and a
    small turn penalty (fewer turns preferred), so the optimizer gets gradient
    on near-misses instead of a flat 0 reward.
  * **variance_reweight** — a per-task weighting of the candidate's mean that
    up-weights tasks on which the population disagrees (high reward variance),
    focusing selection pressure where it can actually rank candidates.

INVARIANT: this shaping affects the CMA-ES *training* fitness ONLY. The
evaluation path (``trinity.eval`` / :func:`trinity.orchestration.reward.score`)
stays pure binary correct/incorrect — these functions never touch it, and the
default config (``enable_reweight=False``, zero bonuses) reproduces the original
mean-binary fitness exactly.

These shaping cores (:func:`shaped_reward`, :func:`variance_reweight`,
:class:`FitnessConfig`) are pure numpy/python with NO torch dependency, so they
are unit-testable on a box without a GPU.

Concurrency model (see docs/SPEC.md §0.3.7, §5.2):
  - Candidates are evaluated SEQUENTIALLY because SVF mutates the single shared SLM's
    weights in place; two candidates cannot be live on the GPU at once.
  - Within one candidate, the `m_cma` trajectories share θ, so their (fast, serialized)
    SLM forwards interleave while the (slow) hosted-model calls run concurrently.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import mean

import numpy as np

from ..orchestration import reward as _reward
from ..orchestration.session import run_trajectory

__all__ = [
    "FitnessConfig",
    "shaped_reward",
    "variance_reweight",
    "hero_quality",
    "hero_bucket_bonus",
    "evaluate_candidate",
    "evaluate_population",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitnessConfig:
    """Training-only fitness-shaping knobs (read from the ``fitness:`` block).

    All defaults reproduce the original mean-binary fitness exactly:
    ``enable_reweight=False`` makes :func:`variance_reweight` a no-op (uniform
    task weights), and a zero ``format_bonus`` + zero ``turn_penalty`` +
    ``hero_dense=False`` make :func:`shaped_reward` collapse to plain binary
    ``correct``.

    Attributes:
        enable_reweight: If ``True``, weight the per-task means by
            :func:`variance_reweight` before averaging into the candidate
            fitness. If ``False`` (default), tasks are weighted uniformly.
        format_bonus: Reward added when the trajectory produced a parseable
            answer (``reward.has_answer``), regardless of correctness. Default
            ``0.05``.
        turn_penalty: Reward subtracted in proportion to how many turns past the
            first the trajectory used (normalized to ``[0, 1]``). Default
            ``0.05``.
        hero_dense: If ``True``, add the HERO dense quality proxy (improvement #3,
            Stage A): a per-trajectory self-consistency score (:func:`hero_quality`)
            min-max normalized *within* the correct and incorrect buckets and added
            as a small bonus (:func:`hero_bucket_bonus`), so CMA-ES gets gradient
            *within* each bucket while correctness stays the dominant term. Default
            ``False`` so behavior is unchanged.
        hero_bonus: Magnitude of the HERO in-bucket bonus (only used when
            ``hero_dense`` is ``True``). Kept at the same ``0.05`` scale as the
            other shaping terms so it never flips the correct-vs-wrong ordering.
    """

    enable_reweight: bool = False
    format_bonus: float = 0.05
    turn_penalty: float = 0.05
    hero_dense: bool = False
    hero_bonus: float = 0.05

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "FitnessConfig":
        """Build from a parsed ``fitness:`` block; ``None``/``{}`` -> defaults."""
        cfg = cfg or {}
        return cls(
            enable_reweight=bool(cfg.get("enable_reweight", False)),
            format_bonus=float(cfg.get("format_bonus", 0.05)),
            turn_penalty=float(cfg.get("turn_penalty", 0.05)),
            hero_dense=bool(cfg.get("hero_dense", False)),
            hero_bonus=float(cfg.get("hero_bonus", 0.05)),
        )

    @property
    def shaping_active(self) -> bool:
        """True iff per-trajectory shaping would change a binary reward."""
        return self.format_bonus != 0.0 or self.turn_penalty != 0.0 or self.hero_dense


# ---------------------------------------------------------------------------
# Pure shaping cores (torch-free, unit-testable)
# ---------------------------------------------------------------------------
def shaped_reward(
    correct: int,
    has_answer: bool,
    num_turns: int,
    max_turns: int,
    cfg: FitnessConfig,
) -> float:
    """Dense per-trajectory training reward.

    ``shaped = correct
               + cfg.format_bonus * has_answer
               - cfg.turn_penalty * (num_turns - 1) / max(1, max_turns - 1)``

    The correctness term (``0`` or ``1``) dominates: with the default bonuses the
    format bonus and turn penalty each have magnitude ``0.05``, so a correct
    trajectory (>= 1.0 - 0.05 = 0.95) always outranks a wrong one
    (<= 0.0 + 0.05 = 0.05). When ``format_bonus`` and ``turn_penalty`` are both
    ``0`` (and ``hero_dense`` is ``False``), this returns exactly ``float(correct)``.

    Args:
        correct: ``1`` if the final answer was judged correct, else ``0``.
        has_answer: Whether a parseable answer was produced (format validity).
        num_turns: Number of turns the trajectory used (>= 1).
        max_turns: The turn budget ``K`` for the run (>= 1).
        cfg: The active :class:`FitnessConfig`.

    Returns:
        The shaped scalar reward.
    """
    r = float(correct)
    if cfg.format_bonus:
        r += cfg.format_bonus * (1.0 if has_answer else 0.0)
    if cfg.turn_penalty:
        denom = max(1, int(max_turns) - 1)
        frac = (max(1, int(num_turns)) - 1) / denom
        # Clamp in case a trajectory somehow reports more turns than the budget.
        frac = min(1.0, max(0.0, frac))
        r -= cfg.turn_penalty * frac
    return r


def variance_reweight(reward_matrix: np.ndarray, cfg: FitnessConfig) -> np.ndarray:
    """Per-task weights that emphasize tasks the population disagrees on.

    For each task (column of ``reward_matrix`` shaped ``[n_candidates, n_tasks]``)
    compute the reward standard deviation across candidates, ``sigma_task``. Tasks
    with above-average ``sigma`` are up-weighted::

        w_j = 0.5 + 1.5 * sigmoid(5 * (sigma_j - mean_sigma))

    so a flat task (everyone scores the same, ``sigma_j == 0`` and below the mean)
    gets ~0.5 and a high-variance task gets up to ~2.0. When ``cfg.enable_reweight``
    is ``False``, or when every task has identical variance (e.g. all-equal
    rewards -> all ``sigma == 0``), the result is uniform weights (all ``1.0``),
    making the downstream weighted mean identical to a plain mean.

    Args:
        reward_matrix: ``[n_candidates, n_tasks]`` array of per-(candidate, task)
            rewards.
        cfg: The active :class:`FitnessConfig`.

    Returns:
        A ``[n_tasks]`` float array of non-negative weights. Uniform ``1.0`` when
        reweighting is disabled or all task variances are equal.
    """
    m = np.asarray(reward_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"reward_matrix must be 2D [n_candidates, n_tasks]; got {m.shape}")
    n_tasks = m.shape[1]
    if not cfg.enable_reweight or n_tasks == 0:
        return np.ones(n_tasks, dtype=float)

    sigma = m.std(axis=0)
    mean_sigma = float(sigma.mean())
    # All tasks equally variable (incl. the all-equal-rewards case) -> uniform.
    if np.allclose(sigma, mean_sigma):
        return np.ones(n_tasks, dtype=float)
    w = 0.5 + 1.5 * _sigmoid(5.0 * (sigma - mean_sigma))
    return w


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _candidate_fitness(per_task: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean of a candidate's per-task rewards (plain mean if uniform)."""
    per_task = np.asarray(per_task, dtype=float)
    if per_task.size == 0:
        return 0.0
    weights = np.asarray(weights, dtype=float)
    wsum = float(weights.sum())
    if wsum <= 0.0:
        return float(per_task.mean())
    return float(np.dot(per_task, weights) / wsum)


# ---------------------------------------------------------------------------
# HERO dense self-consistency proxy (improvement #3, Stage A)
# ---------------------------------------------------------------------------
def _answers_agree(benchmark: str, a: str, b: str) -> bool:
    """Whether two turn outputs express the same answer, per benchmark family.

    Reuses the eval scorer's own extractors so "agreement" here means exactly what
    "correct" means there: a choice letter, a math value (symbolic/numeric), or the
    extracted code string. Unknown benchmarks fall back to a stripped-text match.
    """
    key = (benchmark or "").strip().lower()
    if key in _reward.CHOICE_BENCHMARKS:
        la = _reward.extract_choice_letter(a)
        return la is not None and la == _reward.extract_choice_letter(b)
    if key in _reward.MATH_BENCHMARKS:
        na = _reward.extract_boxed(a) or _reward.extract_last_number(a)
        nb = _reward.extract_boxed(b) or _reward.extract_last_number(b)
        return na is not None and nb is not None and _reward.math_equal(na, nb)
    if key in _reward.CODE_BENCHMARKS:
        ca, cb = _reward.extract_code(a).strip(), _reward.extract_code(b).strip()
        return bool(ca) and ca == cb
    return bool(a.strip()) and a.strip() == b.strip()


def hero_quality(traj) -> float:
    """HERO self-consistency proxy in ``[0, 1]`` for one trajectory.

    Of the trajectory's *non-verifier* answer-bearing turns
    (:func:`reward.answerful_non_verifier_outputs`), the fraction whose answer
    agrees with the committed answer (:func:`reward.committed_answer`). A
    trajectory whose solver turns all reach the same answer is high quality
    (``1.0``); one whose turns disagree is lower; one that never produced a
    parseable answer is ``0.0``. Pure and torch-free.

    Verifier turns are excluded from *both* the vote population and the committed
    reference (see :func:`reward._committed_answer`), so the signal reflects what
    the *solver* committed, never the checker's critique — the concern that closed
    the first cut of this reward. This shapes only TRAINING fitness (via
    :func:`hero_bucket_bonus`); the eval scorer never sees it.
    """
    benchmark = (traj.task.benchmark or "").strip().lower()
    committed = _reward.committed_answer(benchmark, traj)
    outs = _reward.answerful_non_verifier_outputs(benchmark, getattr(traj, "turns", None))
    if not outs:
        return 0.0
    agree = sum(1 for txt in outs if _answers_agree(benchmark, txt, committed))
    return agree / len(outs)


def hero_bucket_bonus(qualities, corrects, cfg: FitnessConfig) -> np.ndarray:
    """Min-max normalize the HERO quality WITHIN the correct/incorrect buckets.

    Given per-trajectory ``qualities`` (from :func:`hero_quality`) and binary
    ``corrects``, normalize the quality to ``[0, 1]`` separately inside the correct
    bucket and the incorrect bucket, then scale by ``cfg.hero_bonus``. The result is
    a non-negative per-trajectory bonus in ``[0, cfg.hero_bonus]``.

    Because it is added on top of the binary correctness anchor, this gives CMA-ES
    gradient *within* each bucket while a correct trajectory always outranks a wrong
    one (the bonus magnitude is ``<= cfg.hero_bonus`` for both). A bucket whose
    members share one quality value (or has a single member) gets a neutral ``0.5``
    normalized score, so a degenerate bucket adds a constant and cannot spuriously
    reorder candidates.

    Args:
        qualities: Per-trajectory HERO quality in ``[0, 1]``.
        corrects: Per-trajectory binary correctness (``0``/``1``), same length.
        cfg: The active :class:`FitnessConfig` (uses ``hero_bonus``).

    Returns:
        A ``[n]`` float array of bonuses, all ``0.0`` when ``hero_bonus == 0``.
    """
    q = np.asarray(qualities, dtype=float)
    c = np.asarray(corrects, dtype=float)
    bonus = np.zeros(q.shape[0], dtype=float)
    if q.size == 0 or cfg.hero_bonus == 0.0:
        return bonus
    for bucket_mask in (c >= 0.5, c < 0.5):
        idx = np.where(bucket_mask)[0]
        if idx.size == 0:
            continue
        vals = q[idx]
        lo, hi = float(vals.min()), float(vals.max())
        norm = np.full(idx.size, 0.5) if hi <= lo else (vals - lo) / (hi - lo)
        bonus[idx] = cfg.hero_bonus * norm
    return bonus


# ---------------------------------------------------------------------------
# Trajectory -> per-task reward
# ---------------------------------------------------------------------------
def _task_reward(traj, cfg: FitnessConfig, max_turns: int) -> float:
    """Per-task TRAINING reward for one completed trajectory.

    Always computes the binary correctness via :func:`reward.score` (and stores
    it on ``traj.reward`` so callers/eval see the unshaped value). When shaping is
    active, returns :func:`shaped_reward`; otherwise returns the plain binary.
    """
    correct = float(_reward.score(traj))
    traj.reward = correct  # keep the binary signal on the trajectory (eval-facing)
    if not cfg.shaping_active:
        return correct
    benchmark = (traj.task.benchmark or "").strip().lower()
    # Score the format bonus on the SAME text ``reward.score`` grades — the
    # committed (non-verifier) answer — so "has an answer" stays consistent with
    # "can be scored" (see ``reward.has_answer``). Reading ``final_answer`` alone
    # denied the bonus to a run whose committed answer lived in an earlier turn.
    committed = _reward.committed_answer(benchmark, traj)
    has_ans = _reward.has_answer(benchmark, committed)
    return shaped_reward(
        int(round(correct)),
        has_ans,
        num_turns=int(getattr(traj, "n_turns", 0) or len(getattr(traj, "turns", []) or [])),
        max_turns=int(max_turns),
        cfg=cfg,
    )


async def evaluate_candidate(
    theta,
    spec,
    policy,
    pool,
    pool_models: list[str],
    minibatch: list,
    *,
    sample: bool = True,
    client=None,
    return_trajectories: bool = False,
    return_per_task: bool = False,
    fitness_cfg: FitnessConfig | None = None,
    max_turns: int = 5,
    **run_kwargs,
) -> tuple:
    """Configure the policy with θ and score it over ``minibatch``.

    Returns ``(fitness, trajectories)`` by default. ``fitness`` is the candidate's
    scalar fitness (plain mean of per-task rewards here; population-level variance
    reweighting is applied in :func:`evaluate_population`). With
    ``return_per_task=True`` the third element is the ``[n_tasks]`` array of
    per-task rewards (failed trajectories contribute ``0.0``), which
    :func:`evaluate_population` needs to build the reward matrix.

    ``fitness_cfg`` selects binary vs shaped per-task reward; ``None`` -> defaults
    (plain binary, original behavior).
    """
    # ``None`` means "no shaping": zero the bonuses rather than taking
    # FitnessConfig()'s field defaults, which are the *configured* shaping values
    # (format_bonus=turn_penalty=0.05) and would silently make the documented
    # plain-binary path shaped. Training always passes an explicit cfg.
    cfg = fitness_cfg if fitness_cfg is not None else FitnessConfig(
        format_bonus=0.0, turn_penalty=0.0
    )
    policy.configure(theta, spec)

    own_client = False
    if client is None:
        try:
            import httpx

            client = httpx.AsyncClient()
            own_client = True
        except Exception:
            client = None
    try:
        # return_exceptions=True so one trajectory that exhausts retries (e.g. a
        # persistent timeout) degrades to reward 0 instead of crashing the whole
        # training run. The optimizer treats it as a (slightly pessimistic) sample.
        trajs = await asyncio.gather(
            *[
                run_trajectory(
                    task, policy, pool, pool_models, sample=sample, client=client,
                    max_turns=max_turns, **run_kwargs,
                )
                for task in minibatch
            ],
            return_exceptions=True,
        )
    finally:
        if own_client and client is not None:
            await client.aclose()

    per_task: list[float] = []
    good_trajs = []
    good_pos: list[int] = []
    n_failed = 0
    for t in trajs:
        if isinstance(t, BaseException):
            n_failed += 1
            per_task.append(0.0)
            continue
        good_pos.append(len(per_task))
        per_task.append(_task_reward(t, cfg, max_turns))
        good_trajs.append(t)
    if n_failed:
        print(f"      [warn] {n_failed}/{len(trajs)} trajectories failed (counted as reward 0)",
              flush=True)

    # HERO dense reward (improvement #3, Stage A): add the self-consistency quality,
    # min-max normalized within the correct/incorrect buckets, as a small in-bucket
    # bonus. Batch-level because the bucket normalization needs the whole minibatch.
    # Failed trajectories (reward 0, no object) are excluded. TRAINING fitness only.
    if cfg.hero_dense and good_trajs:
        corrects = [float(getattr(t, "reward", 0.0) or 0.0) for t in good_trajs]
        qualities = [hero_quality(t) for t in good_trajs]
        bonuses = hero_bucket_bonus(qualities, corrects, cfg)
        for pos, b in zip(good_pos, bonuses):
            per_task[pos] += float(b)

    fit = float(mean(per_task)) if per_task else 0.0
    trajs_out = good_trajs if return_trajectories else []
    if return_per_task:
        return fit, trajs_out, np.asarray(per_task, dtype=float)
    return fit, trajs_out


async def evaluate_population(
    thetas: list,
    spec,
    policy,
    pool,
    pool_models: list[str],
    minibatch_fn,
    *,
    sample: bool = True,
    on_candidate=None,
    fitness_cfg: FitnessConfig | None = None,
    max_turns: int = 5,
    **run_kwargs,
) -> list[float]:
    """Evaluate λ candidates sequentially (GPU constraint). `minibatch_fn(i)->tasks`
    yields the per-candidate minibatch (re-sampled each iteration for an unbiased J).

    `on_candidate(i, fit, elapsed_s)` is called after each candidate for progress
    logging (with the as-evaluated per-task mean; if variance reweighting is on,
    the returned fitness is re-weighted afterward).

    When ``fitness_cfg.enable_reweight`` is on, the per-task reward matrix across
    all candidates is collected and :func:`variance_reweight` is applied so each
    candidate's fitness becomes a variance-weighted task mean. This requires that
    every candidate share a comparable task layout — which holds under the common-
    random-numbers regime (all candidates in a generation score the SAME minibatch,
    so columns align). Default config keeps weights uniform -> plain means.
    """
    import time

    # See evaluate_candidate: ``None`` -> plain binary, not FitnessConfig()'s
    # configured shaping defaults.
    cfg = fitness_cfg if fitness_cfg is not None else FitnessConfig(
        format_bonus=0.0, turn_penalty=0.0
    )

    fits: list[float] = []
    per_task_rows: list[np.ndarray] = []
    client = None
    try:
        import httpx

        client = httpx.AsyncClient()
    except Exception:
        client = None
    try:
        for i, theta in enumerate(thetas):
            t0 = time.time()
            mb = minibatch_fn(i)
            fit, _, per_task = await evaluate_candidate(
                theta, spec, policy, pool, pool_models, mb,
                sample=sample, client=client,
                return_per_task=True, fitness_cfg=cfg, max_turns=max_turns,
                **run_kwargs,
            )
            fits.append(fit)
            per_task_rows.append(per_task)
            if on_candidate is not None:
                on_candidate(i, fit, time.time() - t0)
    finally:
        if client is not None:
            await client.aclose()

    if cfg.enable_reweight and per_task_rows:
        # Only candidates that scored the same number of tasks can share a column
        # layout (true under common random numbers); otherwise fall back to means.
        widths = {row.shape[0] for row in per_task_rows}
        if len(widths) == 1 and per_task_rows[0].shape[0] > 0:
            matrix = np.vstack(per_task_rows)
            weights = variance_reweight(matrix, cfg)
            fits = [_candidate_fitness(row, weights) for row in per_task_rows]

    return fits
