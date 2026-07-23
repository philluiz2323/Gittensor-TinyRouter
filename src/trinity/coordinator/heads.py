"""Alternative coordinator heads (SPEC §3.5, Appendix A.4 ablation parity).

SPEC §1.3 invariant **R10** — *"linear head ≥ all other head variants overall"* (Table 3)
— is the justification for shipping a plain linear routing head. The *verifier* for R10 is
already merged (:mod:`trinity.analysis.head_variants`, PR #352): give it a
``{variant: {benchmark: score}}`` table and it reports the overall ranking and the R10
verdict. But nothing in ``src/`` could ever *produce* a non-linear head, so the table it
grades had to be typed in by hand from the paper. This module is that producer.

Every head here maps ``h ∈ R^{d_h} → z ∈ R^{n_a}`` and splits ``z`` into the same two
logit groups the linear head uses (``z[:n_models]`` agent, ``z[n_models:]`` role, one
softmax each — SPEC §3.3), so a variant is a drop-in replacement for
:class:`~trinity.coordinator.head.LinearHead` at the decision boundary.

SPEC §3.5 Table 3, at the paper's ``d_h=1024, n_a=10``:

===================  ================================================  ==============
Head                 Equation                                          Params (n_a=10)
===================  ================================================  ==============
linear (default)     ``z=Wh``, no bias                                 10,240
low-rank             ``u=ELU(Uh, α=0.1); z=Vu·σ``; ``r=14``, σ fixed   20,680
sparse               ``z=W(h⊙α)``; Gumbel top-k, hard top-k at eval    11,266
block-diagonal-2     B=2 proportional blocks                           5,120
block-diagonal-10    B=10, one block per logit, **argmax** output      1,024
===================  ================================================  ==============

**[SPEC INCONSISTENCY — low-rank]** The stated rank ``r=14`` does not reproduce the
tabled parameter count. A low-rank head is ``U ∈ R^{r×d_h}`` plus ``V ∈ R^{n_a×r}``
(σ is fixed, so it is not a parameter), i.e. ``r·d_h + n_a·r = r·(d_h + n_a)``. At
``d_h=1024, n_a=10`` that is ``14 × 1034 = 14,476``, not 20,680; the tabled number needs
``r=20`` (``20 × 1034 = 20,680``). The two cannot both hold. This module keeps the
**stated hyperparameter** ``r=14`` as the default — a hyperparameter is the more direct
claim, and silently switching to r=20 to make a derived total match would hide the
conflict — and records the discrepancy in :data:`SPEC_TABLE3_PARAMS` and in
``tests/test_torch_coordinator_heads.py::test_lowrank_rank14_contradicts_spec_table3``
so it is documented rather than papered over. Pass ``rank=20`` to reproduce the table.

**Our ``n_a=6``, not the paper's 10.** Per the §3.4 replication delta this build has
``L=3 → n_a=6``, so the tabled counts do not apply directly; use
:func:`param_counts` for the counts at *this* build's shape.
``block-diagonal-10`` is defined by SPEC as *"one block per logit"*, which at the paper's
``n_a=10`` happens to mean 10 blocks. This module implements the **semantics** (one block
per logit → ``B = n_a``), which preserves the "exactly ``d_h`` parameters" property at any
``n_a`` — 1,024 at both ``n_a=10`` and ``n_a=6``. A literal ``B=10`` would simply be
invalid at ``n_a=6`` (10 does not divide 6).

**Deliberately not wired into the submission path.** :mod:`trinity.coordinator.policy`
still constructs :class:`~trinity.coordinator.head.LinearHead` directly and
:func:`trinity.coordinator.params.make_spec` still computes ``n_head = n_a · d_h``, which
is linear-only and feeds the frozen submission dimension ``n_total = 13,312``. Wiring a
variant into that path would change ``n_total`` and break submission. This module is
therefore **purely additive**: it introduces the variants and a factory so the R10
ablation can be *run*, and changes no default.

:func:`from_config` is the first **reader** of the ``coordinator.head`` config block.
Four of its keys — ``type``, ``hidden_dim``, ``include_stop_action`` and ``factorize`` —
had no consumer anywhere in ``src/``, so editing them silently did nothing; they are now
either honoured or rejected with an explicit error. Feeding it the shipped
``configs/trinity.yaml`` reproduces exactly today's linear head, so the default path is
unchanged (``test_from_config_builds_the_shipped_config_unchanged``).

torch is imported at module scope, matching ``coordinator/head.py`` (the sibling this
module parallels). Tests are named ``test_torch_*`` so alphabetical collection keeps them
after ``test_shaped_fitness.py::test_no_torch_imported``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from trinity.types import ROLE_ORDER, Role

__all__ = [
    "BlockDiagonalHead",
    "HEAD_VARIANTS",
    "LowRankHead",
    "SPEC_TABLE3_PARAMS",
    "SparseHead",
    "VariantHead",
    "from_config",
    "make_head",
    "param_counts",
    "spec_param_count",
]


#: SPEC §3.5 Table 3 parameter counts, verbatim, at the paper's ``d_h=1024, n_a=10``.
#: Kept as data so tests can assert which entries this module reproduces and which one
#: (``low_rank``) is internally inconsistent with its own stated ``r=14``.
SPEC_TABLE3_PARAMS: dict[str, int] = {
    "linear": 10_240,
    "low_rank": 20_680,
    "sparse": 11_266,
    "block_diag_2": 5_120,
    "block_diag_10": 1_024,
}

#: SPEC §3.5: low-rank uses ELU with ``α=0.1`` and a *fixed* output scale ``σ``.
LOW_RANK_ELU_ALPHA = 0.1
LOW_RANK_SIGMA = 1.0
LOW_RANK_DEFAULT_RANK = 14

#: SPEC §3.5: sparse uses a Gumbel top-k relaxation with ``τ ∈ [1.0, 20.0]``.
SPARSE_TAU_MIN = 1.0
SPARSE_TAU_MAX = 20.0
SPARSE_DEFAULT_TAU = 1.0
#: ``ρ`` sets the *dropped* fraction via ``k = max(1, ⌊d_h·(1−σ(ρ))⌋)``. ρ=0 → keep half.
SPARSE_DEFAULT_RHO = 0.0


class VariantHead(torch.nn.Module):
    """Base class for the SPEC §3.5 alternative heads.

    Subclasses implement :meth:`forward` (returning the full ``(..., n_a)`` logit vector
    split into the two groups) and :meth:`n_params`. The group split and the
    sample/argmax decision rule are implemented once here so every variant agrees with
    the linear head on everything except the ``h → z`` map itself.

    Notes
    -----
    :class:`~trinity.coordinator.head.LinearHead` keeps its own ``select``: this module
    deliberately does not refactor the submission path (see the module docstring). The
    two implementations are pinned to agree by
    ``tests/test_torch_coordinator_heads.py::test_registry_linear_matches_head_py``.
    """

    #: SPEC §Table 3 caption: block-diag-10 converts its output with **argmax**, not
    #: softmax. Variants that prefer argmax advertise it here; the decision rule itself
    #: still honours the caller's ``sample`` flag so training can stay stochastic.
    prefers_argmax: bool = False

    def __init__(self, n_a: int = 6, d_h: int = 1024, n_models: int = 3) -> None:
        super().__init__()
        n_roles = n_a - n_models
        if n_roles != len(ROLE_ORDER):
            raise ValueError(
                f"n_a - n_models = {n_roles} role logits, but ROLE_ORDER has "
                f"{len(ROLE_ORDER)} roles ({[r.value for r in ROLE_ORDER]})"
            )
        self.n_a = n_a
        self.d_h = d_h
        self.n_models = n_models
        self.n_roles = n_roles

    def logits(self, h: Tensor) -> Tensor:
        """Map ``h`` to the full logit vector ``z`` of shape ``(..., n_a)``."""
        raise NotImplementedError

    def n_params(self) -> int:
        """Number of trainable parameters in this head."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the two logit groups.

        Parameters
        ----------
        h:
            Hidden state of shape ``(d_h,)`` or ``(..., d_h)``. Expected to be
            L2-normalized upstream (SPEC §0.3.2); no head normalizes it.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(agent_logits, role_logits)``, preserving leading batch dimensions.
        """
        z = self.logits(h)
        return z[..., : self.n_models], z[..., self.n_models :]

    @torch.no_grad()
    def select(
        self,
        h: Tensor,
        *,
        sample: bool,
        rng: "torch.Generator | None" = None,
    ) -> tuple[int, Role, dict[str, Any]]:
        """Pick ``(agent_idx, role)`` from a single hidden state.

        Separate softmax per group, then sample (training fitness) or argmax
        (deterministic eval) — SPEC §4.3. Mirrors
        :meth:`trinity.coordinator.head.LinearHead.select` exactly.
        """
        h = h.squeeze(0) if h.dim() == 2 and h.shape[0] == 1 else h
        if h.dim() != 1:
            raise ValueError(
                f"select expects a single hidden state of shape (d_h,), got {tuple(h.shape)}"
            )

        agent_logits, role_logits = self.forward(h)
        agent_probs = torch.softmax(agent_logits, dim=-1)
        role_probs = torch.softmax(role_logits, dim=-1)

        if sample:
            agent_idx = int(torch.multinomial(agent_probs, 1, generator=rng).item())
            role_pos = int(torch.multinomial(role_probs, 1, generator=rng).item())
        else:
            agent_idx = int(torch.argmax(agent_logits, dim=-1).item())
            role_pos = int(torch.argmax(role_logits, dim=-1).item())

        role = ROLE_ORDER[role_pos]
        logits_debug: dict[str, Any] = {
            "agent_logits": agent_logits.detach().to("cpu").float().numpy(),
            "role_logits": role_logits.detach().to("cpu").float().numpy(),
            "agent_probs": agent_probs.detach().to("cpu").float().numpy(),
            "role_probs": role_probs.detach().to("cpu").float().numpy(),
            "agent_idx": agent_idx,
            "role_pos": role_pos,
            "role": role,
            "sampled": sample,
        }
        return agent_idx, role, logits_debug


class LowRankHead(VariantHead):
    """Low-rank head: ``u = ELU(U·h, α=0.1)``, ``z = (V·u)·σ`` (SPEC §3.5).

    ``σ`` is a **fixed** output scale, not a parameter, so the trainable count is
    ``r·(d_h + n_a)``. Both factors use the Xavier-uniform bounds SPEC states verbatim:
    ``U ~ U[±√(6/(d_h+r))]`` and ``V ~ U[±√(18/(r+n_a))]``.

    Parameters
    ----------
    rank:
        Bottleneck width ``r``. Defaults to SPEC's stated ``r=14``; pass ``20`` to
        reproduce the Table 3 count of 20,680 (see the module docstring on why these
        two cannot both hold).
    """

    def __init__(
        self,
        n_a: int = 6,
        d_h: int = 1024,
        n_models: int = 3,
        *,
        rank: int = LOW_RANK_DEFAULT_RANK,
        sigma: float = LOW_RANK_SIGMA,
        generator: "torch.Generator | None" = None,
    ) -> None:
        super().__init__(n_a=n_a, d_h=d_h, n_models=n_models)
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.rank = rank
        self.sigma = float(sigma)
        self.elu_alpha = LOW_RANK_ELU_ALPHA

        bound_u = math.sqrt(6.0 / (d_h + rank))
        bound_v = math.sqrt(18.0 / (rank + n_a))
        self.U = torch.nn.Parameter(
            _uniform((rank, d_h), bound_u, generator=generator)
        )
        self.V = torch.nn.Parameter(
            _uniform((n_a, rank), bound_v, generator=generator)
        )

    def logits(self, h: Tensor) -> Tensor:
        u = torch.nn.functional.elu(
            torch.matmul(h, self.U.t()), alpha=self.elu_alpha
        )
        return torch.matmul(u, self.V.t()) * self.sigma


class SparseHead(VariantHead):
    """Sparse-edge head: ``z = W·(h ⊙ α)`` with a learned Gumbel top-k gate (SPEC §3.5).

    The gate keeps ``k = max(1, ⌊d_h·(1−σ(ρ))⌋)`` input dimensions. During training the
    top-k is relaxed: Gumbel noise perturbs the gate scores and the mask is a temperature
    ``τ`` sigmoid around the k-th largest perturbed score, so the selection is
    differentiable. At inference the mask is a **hard** top-k of ``α`` with no noise —
    SPEC's *"hard top-k at inference"* — which also makes eval deterministic.

    Trainable parameters are ``W`` (``n_a·d_h``), the per-dimension gate ``α`` (``d_h``),
    and the two scalars ``ρ`` and ``τ`` — matching SPEC's stated ``d_h·n_a + d_h + 2``.

    **``ρ`` carries no gradient, by construction.** SPEC defines the kept-edge count as
    ``k = max(1, ⌊d_h·(1−σ(ρ))⌋)``, and a floor to an integer has zero derivative almost
    everywhere, so ``∂L/∂ρ`` is structurally zero however the mask is relaxed — the
    relaxation makes *which* edges survive differentiable, not *how many*. ``ρ`` is kept
    as a parameter anyway because SPEC's ``+2`` counts it, and because it stays settable
    (and searchable by the gradient-free optimizers this repo actually uses for the head
    — sep-CMA-ES, Random Search). This is asserted, not hidden, by
    ``test_sparse_rho_has_no_gradient_path``. Only ``τ`` is gradient-trainable of the two.
    """

    def __init__(
        self,
        n_a: int = 6,
        d_h: int = 1024,
        n_models: int = 3,
        *,
        rho: float = SPARSE_DEFAULT_RHO,
        tau: float = SPARSE_DEFAULT_TAU,
        generator: "torch.Generator | None" = None,
    ) -> None:
        super().__init__(n_a=n_a, d_h=d_h, n_models=n_models)
        if not SPARSE_TAU_MIN <= tau <= SPARSE_TAU_MAX:
            raise ValueError(
                f"tau must lie in [{SPARSE_TAU_MIN}, {SPARSE_TAU_MAX}], got {tau}"
            )
        # Zero-init W → uniform policy at start, matching LinearHead / params.initial_theta.
        self.weight = torch.nn.Parameter(torch.zeros(n_a, d_h))
        # Gate scores start at 1.0 (all edges equally live, gate is the identity on h).
        self.alpha = torch.nn.Parameter(torch.ones(d_h))
        self.rho = torch.nn.Parameter(torch.tensor(float(rho)))
        self.tau = torch.nn.Parameter(torch.tensor(float(tau)))
        self._generator = generator

    def keep_k(self) -> int:
        """``k = max(1, ⌊d_h·(1−σ(ρ))⌋)`` — how many input dims survive the gate."""
        keep_frac = 1.0 - float(torch.sigmoid(self.rho.detach()).item())
        return max(1, int(math.floor(self.d_h * keep_frac)))

    def gate(self, *, hard: bool) -> Tensor:
        """Build the multiplicative gate over input dimensions.

        Parameters
        ----------
        hard:
            ``True`` → deterministic hard top-k of ``α`` (inference). ``False`` → Gumbel
            top-k relaxation (training).
        """
        k = self.keep_k()
        scores: Tensor = self.alpha
        if not hard:
            scores = scores + _gumbel_like(scores, generator=self._generator)

        if k >= self.d_h:
            mask = torch.ones_like(scores)
        else:
            kth = torch.topk(scores, k).values[-1]
            if hard:
                mask = (scores >= kth).to(scores.dtype)
            else:
                tau = self.tau.clamp(SPARSE_TAU_MIN, SPARSE_TAU_MAX)
                mask = torch.sigmoid((scores - kth) / tau)
        return self.alpha * mask

    def logits(self, h: Tensor) -> Tensor:
        # Hard top-k whenever grads are off (eval / select); relaxed while training.
        gate = self.gate(hard=not torch.is_grad_enabled())
        return torch.matmul(h * gate, self.weight.t())


class BlockDiagonalHead(VariantHead):
    """Block-diagonal head: ``B`` proportional blocks along both axes (SPEC §3.5).

    Input dims are partitioned into ``B`` contiguous chunks of ``d_h/B`` and output
    logits into ``B`` chunks of ``n_a/B``; chunk ``b`` of ``z`` depends only on chunk
    ``b`` of ``h``. That is an exact ``B×`` parameter reduction: ``d_h·n_a/B``.

    ``B = n_a`` is SPEC's *"one block per logit"* variant (its "block-diagonal-10" at the
    paper's ``n_a=10``), and it is the one SPEC pairs with **argmax** output.

    **Why the input blocks are only _near_-equal.** ``B`` must divide ``n_a`` (the output
    logits have to split cleanly into blocks), but it need **not** divide ``d_h`` — and in
    fact it does not for the shapes SPEC itself quotes: ``1024/10`` is 102.4, so the
    paper's own block-diag-10 cannot use equal input chunks. The inputs are therefore
    partitioned into ``B`` contiguous **near-equal** chunks whose sizes sum to exactly
    ``d_h`` (e.g. ``102,102,…,103``). Because the chunk sizes sum to ``d_h``, the total
    parameter count is ``(n_a/B)·Σ_b in_b = (n_a/B)·d_h`` — *exact*, with no rounding, at
    any ``d_h``. That reproduces every Table 3 count: 5,120 at ``B=2`` and SPEC's "exact
    10× reduction" to 1,024 at one-block-per-logit. Requiring ``B | d_h`` instead would
    make the paper's headline variant unconstructible.
    """

    def __init__(
        self,
        n_a: int = 6,
        d_h: int = 1024,
        n_models: int = 3,
        *,
        n_blocks: int = 2,
    ) -> None:
        super().__init__(n_a=n_a, d_h=d_h, n_models=n_models)
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        if n_a % n_blocks:
            raise ValueError(
                f"n_blocks={n_blocks} must divide n_a={n_a} so the output logits "
                "split evenly into blocks"
            )
        self.n_blocks = n_blocks
        self.out_per_block = n_a // n_blocks
        self.in_sizes = _near_equal_split(d_h, n_blocks)
        self.prefers_argmax = n_blocks == n_a
        # Zero-init → uniform policy at start, like LinearHead. One parameter per block
        # (rather than one padded tensor) so the count is exactly (n_a/B)·d_h.
        self.blocks = torch.nn.ParameterList(
            [torch.zeros(self.out_per_block, in_b) for in_b in self.in_sizes]
        )

    def logits(self, h: Tensor) -> Tensor:
        chunks = torch.split(h, self.in_sizes, dim=-1)
        # Block b of z depends only on block b of h.
        parts = [torch.matmul(c, W.t()) for c, W in zip(chunks, self.blocks)]
        return torch.cat(parts, dim=-1)


def _near_equal_split(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` contiguous near-equal sizes summing to ``total``.

    The first ``total % parts`` chunks get one extra element, so e.g. ``(1024, 10)`` gives
    four 103s and six 102s — 1024 exactly, never 1020 or 1030.
    """
    base, extra = divmod(total, parts)
    if base == 0:
        raise ValueError(
            f"cannot split {total} inputs into {parts} non-empty blocks"
        )
    return [base + 1] * extra + [base] * (parts - extra)


def _uniform(
    shape: tuple[int, ...], bound: float, *, generator: "torch.Generator | None"
) -> Tensor:
    """Draw ``U[-bound, +bound]`` of the given shape (Xavier-uniform per SPEC §3.5)."""
    t = torch.empty(*shape)
    return t.uniform_(-bound, bound, generator=generator)


def _gumbel_like(t: Tensor, *, generator: "torch.Generator | None") -> Tensor:
    """Standard Gumbel(0,1) noise shaped like ``t``."""
    u = torch.rand(t.shape, generator=generator, dtype=t.dtype, device=t.device)
    # Clamp away from 0 so the double log stays finite.
    u = u.clamp_min(torch.finfo(t.dtype).tiny)
    return -torch.log(-torch.log(u))


def _make_linear(n_a: int, d_h: int, n_models: int, **kwargs: Any) -> torch.nn.Module:
    """Build the default linear head, reusing ``coordinator/head.py`` (no re-implementation)."""
    from trinity.coordinator.head import LinearHead

    return LinearHead(n_a=n_a, d_h=d_h, n_models=n_models, **kwargs)


#: Name → builder. ``head.type`` in ``configs/trinity.yaml`` selects from these keys;
#: :func:`make_head` is that knob's first reader.
HEAD_VARIANTS: dict[str, Callable[..., torch.nn.Module]] = {
    "linear": _make_linear,
    "low_rank": LowRankHead,
    "sparse": SparseHead,
    "block_diag_2": lambda n_a, d_h, n_models, **kw: BlockDiagonalHead(
        n_a=n_a, d_h=d_h, n_models=n_models, n_blocks=2, **kw
    ),
    # SPEC's "block-diagonal-10" = one block per logit (see the module docstring).
    "block_diag_10": lambda n_a, d_h, n_models, **kw: BlockDiagonalHead(
        n_a=n_a, d_h=d_h, n_models=n_models, n_blocks=n_a, **kw
    ),
}


def make_head(
    variant: str = "linear",
    *,
    n_a: int = 6,
    d_h: int = 1024,
    n_models: int = 3,
    **kwargs: Any,
) -> torch.nn.Module:
    """Build a head by SPEC §3.5 variant name.

    This is the first reader of the ``coordinator.head.type`` config knob, which has had
    no consumer since the head was written. The default is ``"linear"``, so calling this
    with a config that was never edited reproduces exactly what
    :mod:`trinity.coordinator.policy` builds today.

    Parameters
    ----------
    variant:
        One of :data:`HEAD_VARIANTS`.
    n_a, d_h, n_models:
        Head shape, as for :class:`~trinity.coordinator.head.LinearHead`.
    **kwargs:
        Forwarded to the variant (e.g. ``rank=20`` for ``low_rank``).

    Raises
    ------
    ValueError
        If ``variant`` is not a known head name.
    """
    try:
        builder = HEAD_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"unknown head variant {variant!r}; expected one of "
            f"{sorted(HEAD_VARIANTS)}"
        ) from None
    return builder(n_a=n_a, d_h=d_h, n_models=n_models, **kwargs)


def from_config(cfg: Mapping[str, Any]) -> torch.nn.Module:
    """Build the head described by a loaded ``configs/trinity.yaml``.

    This is the first actual **reader** of the ``coordinator.head`` block. Four of its
    keys — ``type``, ``hidden_dim``, ``include_stop_action`` and ``factorize`` — had no
    consumer anywhere in ``src/``, so editing them silently did nothing. Here they are
    honoured or rejected:

    * ``type`` selects the variant (:data:`HEAD_VARIANTS`).
    * ``n_models`` / ``n_a`` set the shape; ``n_a`` defaults to ``n_models + 3``.
    * ``hidden_dim`` must be ``0`` — every SPEC §3.5 head maps ``h → z`` directly; there
      is no hidden layer in any of them. A non-zero value describes a head this
      replication does not have, so it is an error rather than a silent no-op.
    * ``include_stop_action`` must be false — termination is implicit (SPEC §3.3: "no
      stop logit"), and a stop action would change ``n_a`` and hence the frozen
      submission dimension.
    * ``factorize`` must be ``"two_softmax"`` — the two-group split every head here
      implements.

    ``d_h`` is not in the config (it is implied by ``encoder_model``), so it comes from
    :data:`trinity.coordinator.params.DEFAULT_D_H`, keeping one source of truth.

    Raises
    ------
    ValueError
        If the config selects an unknown variant or requests a behaviour no head here
        implements.
    """
    from trinity.coordinator.params import DEFAULT_D_H

    head_cfg = dict((cfg.get("coordinator") or {}).get("head") or {})

    hidden_dim = int(head_cfg.get("hidden_dim", 0) or 0)
    if hidden_dim != 0:
        raise ValueError(
            f"coordinator.head.hidden_dim={hidden_dim}: no SPEC §3.5 head has a hidden "
            "layer (all map h -> z directly); expected 0"
        )
    if head_cfg.get("include_stop_action", False):
        raise ValueError(
            "coordinator.head.include_stop_action=true is not implemented: termination "
            "is implicit (SPEC §3.3, no stop logit) and a stop logit would change n_a"
        )
    factorize = str(head_cfg.get("factorize", "two_softmax"))
    if factorize != "two_softmax":
        raise ValueError(
            f"coordinator.head.factorize={factorize!r}: every head here splits z into "
            "agent/role groups with one softmax each; expected 'two_softmax'"
        )

    n_models = int(head_cfg.get("n_models", 3))
    n_a = int(head_cfg.get("n_a", n_models + len(ROLE_ORDER)))
    return make_head(
        str(head_cfg.get("type", "linear")),
        n_a=n_a,
        d_h=DEFAULT_D_H,
        n_models=n_models,
    )


def spec_param_count(
    variant: str,
    *,
    n_a: int = 6,
    d_h: int = 1024,
    rank: int = LOW_RANK_DEFAULT_RANK,
) -> int:
    """Closed-form trainable-parameter count for a variant, without building it.

    Lets the R10 ablation report parameter counts (the efficiency context Table 3
    highlights) without allocating a ``d_h``-sized module per variant.
    """
    if variant == "linear":
        return n_a * d_h
    if variant == "low_rank":
        return rank * (d_h + n_a)
    if variant == "sparse":
        return d_h * n_a + d_h + 2
    # Block-diagonal: (n_a/B) output rows per block times d_h total inputs (the
    # near-equal input chunks sum to exactly d_h), so the count is exact at any d_h.
    if variant == "block_diag_2":
        return (n_a // 2) * d_h
    if variant == "block_diag_10":
        return d_h  # one block per logit → B = n_a → exactly d_h
    raise ValueError(
        f"unknown head variant {variant!r}; expected one of {sorted(HEAD_VARIANTS)}"
    )


def param_counts(
    *, n_a: int = 6, d_h: int = 1024, rank: int = LOW_RANK_DEFAULT_RANK
) -> dict[str, int]:
    """All variants' parameter counts — feeds ``analysis.head_variants.analyze_heads``.

    :func:`trinity.analysis.head_variants.analyze_heads` takes an optional ``params``
    mapping to add Table 3's efficiency column. This produces that mapping for *this*
    build's head shape instead of the paper's.
    """
    return {
        name: spec_param_count(name, n_a=n_a, d_h=d_h, rank=rank)
        for name in HEAD_VARIANTS
    }


def describe(counts: Mapping[str, int] | None = None) -> str:
    """One-line-per-variant summary of the head zoo (for logs / smoke output)."""
    counts = counts or param_counts()
    width = max(len(n) for n in counts)
    lines = [f"{'head'.ljust(width)}  params"]
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name.ljust(width)}  {n:,}")
    return "\n".join(lines)
