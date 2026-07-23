"""Supervised fine-tuning (SFT) baseline trainer — docs/SPEC.md L420/L464, R8, milestone M4.

SPEC §1.3 **R8** claims ``sep-CMA-ES > SFT > RS > REINFORCE on all 4 tasks`` (Table 4), and
§9 pins the SFT recipe exactly: **Adam, lr 1e-6, batch 64, frozen SLM, head-only**. Two
consumers of that ordering are already merged — :data:`trinity.analysis.convergence.
R8_EXPECTED_ORDER` names ``"sft"``, and ``configs/trinity.yaml`` ships an ``imitation_sft``
entry plus an ``sft:`` hyperparameter block — but nothing in ``src/`` could ever *produce*
an SFT run, so the ordering was unverifiable at its source. This module is that producer.

**What SFT is here.** The SLM is frozen, so the only trainable tensor is the head weight
``W ∈ R^{n_a × d_h}`` (coordinator/head.py, ``z = W·h``). With frozen features the routing
head is a multinomial logistic regression over precomputed hidden states, and SFT is
imitation learning: for each query, build a teacher categorical over pool models from the
measured per-(query, model) solve rates, then minimize softmax cross-entropy between the
head's agent distribution and that teacher. Labels come from the oracle-ceiling matrices
already on disk — **no new API calls, zero marginal cost**.

**Relationship to coordinator/warmstart.py — deliberately not the same thing.**
``warmstart.fit_agent_head`` also fits agent rows by cross-entropy, but it is a *pure-numpy
full-batch GD helper whose only job is to produce a CMA-ES initial mean* ``x0``: bespoke
lr 0.5, L2, disagreement reweighting, no optimizer state, no trainer summary. This module is
an *R8 competitor* and must therefore reproduce the SPEC recipe verbatim (torch Adam, lr
1e-6, minibatch 64) and implement :class:`~trinity.optim.base.BaseTrainer` so its result is
directly comparable with sep-CMA-ES and Random Search. Sharing a loss family is expected —
they are the same statistical problem — but a warm-start and a baseline are different
experiments and collapsing them would make R8 compare CMA-ES against its own initializer.
The label loader :func:`trinity.coordinator.warmstart.load_labels` IS reused rather than
reimplemented, so the two cannot drift on the on-disk matrix schema.

**Scope, stated honestly.** Only the head's *agent* rows are fit: the oracle matrices carry
per-model correctness and nothing about roles, so there is no role supervision to learn
from. Role rows stay 0 (uniform) and SVF scales stay 1.0 (identity), matching
:func:`trinity.coordinator.warmstart.pack_warmstart_theta`, which is reused to build the
full-length θ. A run therefore produces a θ of exactly ``spec.n_total`` that the same
evaluation path scores, so R8 compares like with like.

torch is imported **lazily, inside the fitting functions**, so ``import trinity.optim.sft``
(and hence ``import trinity.optim``) stays torch-free at module scope — the invariant
``trinity/optim/baselines.py`` documents and ``tests/test_shaped_fitness.py::
test_no_torch_imported`` enforces. The target/batching layer is pure numpy and unit-tests
without torch at all.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np

from trinity.optim.base import BaseTrainer

__all__ = [
    "SFT_OPTIMIZER",
    "SFT_LR",
    "SFT_BATCH_SIZE",
    "SFT_EPOCHS",
    "SFT_TARGET_TEMP",
    "build_teacher_targets",
    "iter_minibatches",
    "fit_head_sft",
    "SFTTrainer",
]

#: Optimizer mandated by docs/SPEC.md L464 (§9 row "SFT").
SFT_OPTIMIZER: str = "adam"
#: Learning rate, docs/SPEC.md L420/L464 and the ``baselines.sft`` block in trinity.yaml.
SFT_LR: float = 1.0e-6
#: Minibatch size, docs/SPEC.md L420/L464.
SFT_BATCH_SIZE: int = 64
#: Passes over the labelled set. Not pinned by the SPEC (which fixes lr/batch only);
#: chosen so the default recipe is a complete, runnable configuration.
SFT_EPOCHS: int = 20
#: Softmax temperature used to peak the teacher categorical built from solve rates.
#: Smaller -> routes harder toward the single best model. 1.0 = use solve rates as-is.
SFT_TARGET_TEMP: float = 0.5


def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, max-shifted for numerical stability."""
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def build_teacher_targets(
    solve_prob: np.ndarray,
    *,
    temperature: float = SFT_TARGET_TEMP,
) -> np.ndarray:
    """Build the per-query teacher categorical over pool models from solve rates.

    SFT needs a distribution to imitate. ``solve_prob[i, m]`` — model ``m``'s measured
    solve rate on query ``i`` — is not itself a distribution (rows need not sum to 1, and
    a row of all-zeros carries no preference), so it is converted:

    * rows with at least one non-zero solve rate become ``softmax(solve_prob / temperature)``,
      which is peaked toward the best model(s) as ``temperature`` shrinks;
    * rows **no** model solved become exactly uniform. Those queries are unsolvable by this
      pool, so any routing target would be fabricated supervision; uniform contributes no
      gradient preference between models.

    Args:
        solve_prob: ``(N, n_models)`` solve rates, clipped into ``[0, 1]``.
        temperature: Softmax temperature. Must be > 0.

    Returns:
        ``(N, n_models)`` float64 array whose rows each sum to 1.

    Raises:
        ValueError: If ``solve_prob`` is not 2-D or ``temperature <= 0``.
    """
    sp = np.asarray(solve_prob, dtype=float)
    if sp.ndim != 2:
        raise ValueError(f"solve_prob must be 2-D (N, n_models); got shape {sp.shape}")
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    sp = np.clip(sp, 0.0, 1.0)
    n_rows, n_models = sp.shape
    target = np.full((n_rows, n_models), 1.0 / n_models, dtype=float)
    if n_rows == 0:
        return target
    has_solver = sp.sum(axis=1) > 0
    if has_solver.any():
        target[has_solver] = _softmax_rows(sp[has_solver] / temperature)
    return target


def iter_minibatches(
    n_rows: int,
    batch_size: int = SFT_BATCH_SIZE,
    *,
    epochs: int = SFT_EPOCHS,
    seed: int = 0,
) -> Iterator[np.ndarray]:
    """Yield shuffled index minibatches, ``epochs`` passes over ``range(n_rows)``.

    Each epoch is an independent permutation drawn from ``default_rng(seed + epoch)``, so a
    run is reproducible from ``seed`` alone without touching global RNG state. The final
    batch of an epoch is short when ``batch_size`` does not divide ``n_rows`` (kept rather
    than dropped — with ~120 labelled queries, dropping would discard a large fraction).

    Args:
        n_rows: Number of labelled examples.
        batch_size: Rows per batch (SPEC: 64).
        epochs: Number of full passes.
        seed: Base RNG seed.

    Yields:
        int64 index arrays, each of length ``<= batch_size``.

    Raises:
        ValueError: If ``n_rows < 1``, ``batch_size < 1`` or ``epochs < 1``.
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    for epoch in range(int(epochs)):
        order = np.random.default_rng(int(seed) + epoch).permutation(int(n_rows))
        for start in range(0, int(n_rows), int(batch_size)):
            yield order[start : start + int(batch_size)]


def fit_head_sft(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    lr: float = SFT_LR,
    batch_size: int = SFT_BATCH_SIZE,
    epochs: int = SFT_EPOCHS,
    seed: int = 0,
    device: str = "cpu",
    return_history: bool = False,
):
    """Fit agent-selection head rows by Adam on softmax cross-entropy (SPEC §9 SFT row).

    Implements the SPEC recipe literally: ``torch.optim.Adam`` at ``lr`` over minibatches of
    ``batch_size``, with the SLM frozen — ``features`` are precomputed hidden states carried
    as a **non-differentiable constant tensor**, so the only parameter receiving gradient is
    the returned head block ("head-only"). The loss is the soft-label cross-entropy
    ``-Σ target · log softmax(W·h)``, averaged over the batch.

    Note the SPEC learning rate (1e-6) is calibrated for a real run over many steps; on a
    small synthetic problem it moves ``W`` very little per step, so tests that assert
    *convergence* legitimately pass a larger ``lr`` while separate tests assert the module
    **defaults** equal the SPEC values. Both facts matter and neither substitutes for the other.

    Args:
        features: ``(N, d_h)`` frozen query encodings (L2-normalized SLM hidden states).
        targets: ``(N, n_models)`` teacher distribution from :func:`build_teacher_targets`;
            rows should sum to 1.
        lr: Adam learning rate.
        batch_size: Minibatch size.
        epochs: Passes over the labelled set.
        seed: Seed for batch shuffling and torch parameter init.
        device: torch device string; ``"cpu"`` everywhere in CI.
        return_history: Also return the per-step loss records.

    Returns:
        ``W_agent`` of shape ``(n_models, d_h)`` as float64 numpy; or
        ``(W_agent, history)`` when ``return_history``, where history is a list of
        ``{"step", "epoch", "loss"}`` dicts.

    Raises:
        ValueError: On shape disagreement, or non-positive ``lr``.
    """
    # Lazy import: keeps `import trinity.optim.sft` torch-free at module scope.
    import torch

    H = np.ascontiguousarray(features, dtype=np.float64)
    T = np.ascontiguousarray(targets, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"features must be 2-D (N, d_h); got shape {H.shape}")
    if T.ndim != 2:
        raise ValueError(f"targets must be 2-D (N, n_models); got shape {T.shape}")
    if H.shape[0] != T.shape[0]:
        raise ValueError(f"features has {H.shape[0]} rows but targets has {T.shape[0]}")
    if H.shape[0] == 0:
        raise ValueError("features is empty; nothing to fit")
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    n_rows, d_h = H.shape
    n_models = T.shape[1]

    torch.manual_seed(int(seed))
    # requires_grad=False on the features IS the frozen-SLM contract (SPEC §9 "frozen SLM").
    feats = torch.tensor(H, dtype=torch.float64, device=device, requires_grad=False)
    tgts = torch.tensor(T, dtype=torch.float64, device=device, requires_grad=False)
    # Tiny non-zero init breaks the symmetry a zero head would have across models.
    W = torch.nn.Parameter(
        torch.randn(n_models, d_h, dtype=torch.float64, device=device) * 1e-3
    )
    opt = torch.optim.Adam([W], lr=float(lr))

    history: list[dict] = []
    step = 0
    for epoch in range(int(epochs)):
        for idx in iter_minibatches(n_rows, batch_size, epochs=1, seed=int(seed) + epoch):
            sel = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device)
            logits = feats.index_select(0, sel) @ W.t()
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(tgts.index_select(0, sel) * log_probs).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append({"step": step, "epoch": epoch, "loss": float(loss.detach())})
            step += 1

    W_out = W.detach().to("cpu").numpy().astype(np.float64)
    return (W_out, history) if return_history else W_out


class SFTTrainer(BaseTrainer):
    """SFT baseline trainer (docs/SPEC.md L420/L464; R8, milestone M4).

    Fits the head's agent rows by Adam on softmax cross-entropy against teacher
    distributions derived from an oracle-ceiling matrix, then packs the result into a
    full-length θ via :func:`trinity.coordinator.warmstart.pack_warmstart_theta`.

    Unlike :class:`~trinity.optim.baselines.RandomSearchTrainer` and sep-CMA-ES, SFT is
    **offline**: it consumes previously-measured labels rather than rolling out sessions
    against the pool, so a run makes no API calls and ``total_cost_usd`` is 0.0. That is
    inherent to imitation learning, not a shortcut — and it is exactly why R8 expects SFT to
    beat RS while losing to sep-CMA-ES, which optimizes the deployed objective directly.
    """

    def __init__(
        self,
        *,
        lr: float = SFT_LR,
        batch_size: int = SFT_BATCH_SIZE,
        epochs: int = SFT_EPOCHS,
        target_temp: float = SFT_TARGET_TEMP,
        seed: int = 0,
    ) -> None:
        """Configure the SFT baseline.

        Args:
            lr: Adam learning rate (SPEC default 1e-6).
            batch_size: Minibatch size (SPEC default 64).
            epochs: Passes over the labelled set.
            target_temp: Teacher-softmax temperature.
            seed: Seed for shuffling and init.

        Raises:
            ValueError: If any hyperparameter is out of range.
        """
        if not lr > 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if not target_temp > 0:
            raise ValueError(f"target_temp must be > 0, got {target_temp}")
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.target_temp = float(target_temp)
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
        """Run the SFT baseline and return the :class:`BaseTrainer` summary dict.

        Args:
            policy: Unused — SFT trains on frozen precomputed features, so it never
                routes. Accepted to satisfy the :class:`BaseTrainer` signature.
            pool: Unused for the same reason; read only for ``total_cost_usd``.
            tasks: Unused; labels come from ``solve_prob`` / ``matrix_path``.
            spec: Coordinator parameter spec (``spec.n_total`` sets the θ length).
            features: ``(N, d_h)`` frozen encodings, aligned row-wise with the labels.
            matrix_path: ``oracle_matrix_<bench>.json`` to read labels from, via
                :func:`trinity.coordinator.warmstart.load_labels`.
            solve_prob: ``(N, n_models)`` solve rates, as an alternative to ``matrix_path``.
            pool_models: Model names in slot order; defaults to the matrix's model order.
            run_dir: Directory for ``best_theta.npy`` / ``history.json`` / ``summary.json``.
            benchmark: Benchmark label for the summary.
            device: torch device for the fit.

        Returns:
            Summary dict with the :class:`BaseTrainer` keys plus SFT metadata and the
            per-step ``history``.

        Raises:
            ValueError: If ``spec``/``run_dir``/``features`` is missing, if neither
                ``solve_prob`` nor ``matrix_path`` is given, or if shapes disagree.
        """
        if spec is None:
            raise ValueError("SFTTrainer.train requires spec=<parameter spec>")
        if run_dir is None:
            raise ValueError("SFTTrainer.train requires run_dir=<path>")
        if features is None:
            raise ValueError(
                "SFTTrainer.train requires features=<(N, d_h) frozen encodings>; the SLM is "
                "frozen, so encode once with coordinator.warmstart.encode_queries and cache it"
            )
        # Reuse the canonical loader so this cannot drift from the on-disk matrix schema.
        from trinity.coordinator.warmstart import load_labels, pack_warmstart_theta

        if solve_prob is None:
            if matrix_path is None:
                raise ValueError(
                    "SFTTrainer.train requires either solve_prob=<(N, n_models)> or "
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

        targets = build_teacher_targets(sp, temperature=self.target_temp)
        W_agent, history = fit_head_sft(
            H,
            targets,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            seed=self.seed,
            device=device,
            return_history=True,
        )
        theta = pack_warmstart_theta(W_agent, spec)
        np.save(run_dir / "best_theta.npy", theta)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        final_loss = float(history[-1]["loss"]) if history else math.nan
        summary: Dict[str, Any] = {
            "trainer": "sft",
            "benchmark": benchmark,
            "pool": list(pool_models),
            "n_total": int(spec.n_total),
            "n_labelled": int(sp.shape[0]),
            "optimizer": SFT_OPTIMIZER,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "target_temp": self.target_temp,
            "steps": len(history),
            "final_loss": final_loss,
            # SFT's objective is cross-entropy, not routing reward. Reporting a reward-like
            # "best_fitness" it never measured would be fabricated, so the BaseTrainer key
            # carries the negated final loss (higher = better, as the key requires) and the
            # raw loss is kept alongside. R8 must score this θ through the same evaluation
            # path as the other optimizers to get a comparable fitness.
            "best_fitness": (-final_loss if history else math.nan),
            "objective": "softmax_cross_entropy",
            "best_theta_path": str(run_dir / "best_theta.npy"),
            "run_dir": str(run_dir),
            "seed": self.seed,
            "total_cost_usd": float(getattr(pool, "total_cost_usd", 0.0)),
            "history": history,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
