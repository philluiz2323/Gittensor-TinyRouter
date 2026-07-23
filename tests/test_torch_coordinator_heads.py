"""Offline CPU tests for the SPEC §3.5 alternative coordinator heads.

No network, no GPU — every head here is at most a few hundred KB of weights, so the
update rules and structural invariants are all exercisable on CPU with small tensors.

``heads.py`` imports ``torch`` at module scope (like its sibling ``head.py``), so both
it and ``torch`` are imported **lazily inside the tests**. That is load-bearing, not
style: pytest imports every selected test module during *collection*, before running
anything, so a module-scope ``import torch`` here would put torch in ``sys.modules``
before ``test_shaped_fitness.py::test_no_torch_imported`` ever runs and fail it — no
matter what this file is named. For the same reason the ``@parametrize`` decorators
below use the literal :data:`VARIANTS` rather than ``HEAD_VARIANTS``, since decorator
arguments are evaluated at collection time too; ``test_variants_list_matches_registry``
pins the literal against the real registry so it cannot drift.

The ``test_torch_`` filename prefix then keeps this module *running* after that
invariant in pytest's alphabetical order (once any test here runs, torch stays in
``sys.modules`` for the rest of the process). Do not rename it earlier.
"""
from __future__ import annotations

import math
import pathlib

import pytest

from trinity.types import ROLE_ORDER  # torch-free

# The paper's shape (SPEC §3.5 Table 3 is quoted at these values).
PAPER = {"n_a": 10, "d_h": 1024, "n_models": 7}
# This build's shape (SPEC §3.4 replication delta: L=3 → n_a=6).
OURS = {"n_a": 6, "d_h": 1024, "n_models": 3}

#: Literal copy of the registry keys — see the module docstring on why this cannot be
#: ``sorted(HEAD_VARIANTS)``. Pinned by ``test_variants_list_matches_registry``.
VARIANTS = ["block_diag_10", "block_diag_2", "linear", "low_rank", "sparse"]
#: Variants that start from a zero-initialized weight (so they begin uniform).
ZERO_INIT = ["block_diag_10", "block_diag_2", "linear", "sparse"]
#: Table 3 counts this module reproduces exactly at the paper's shape (low-rank is the
#: documented exception — see ``test_lowrank_rank14_contradicts_spec_table3``).
EXACT_TABLE3 = ["linear", "sparse", "block_diag_2", "block_diag_10"]


def _torch():
    return pytest.importorskip("torch", reason="torch required for the coordinator heads")


def _heads():
    import trinity.coordinator.heads as heads

    return heads


def _LinearHead():
    from trinity.coordinator.head import LinearHead

    return LinearHead


def _rand_h(d_h: int, *, batch: tuple[int, ...] = (), seed: int = 0):
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*batch, d_h, generator=g, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_variants_list_matches_registry():
    """The collection-time literal must stay in sync with the real registry."""
    assert sorted(_heads().HEAD_VARIANTS) == sorted(VARIANTS)


# --------------------------------------------------------------------------------------
# SPEC Table 3 parameter counts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", EXACT_TABLE3)
def test_closed_form_reproduces_spec_table3_at_paper_shape(variant):
    """Every Table 3 count except low-rank is reproduced exactly at d_h=1024, n_a=10."""
    h = _heads()
    got = h.spec_param_count(variant, n_a=PAPER["n_a"], d_h=PAPER["d_h"])
    assert got == h.SPEC_TABLE3_PARAMS[variant]


def test_lowrank_rank14_contradicts_spec_table3():
    """SPEC states r=14 AND 20,680 params for low-rank; those cannot both hold.

    ``r·(d_h + n_a)`` is 14,476 at r=14 and 20,680 at r=20. This pins the discrepancy so
    it stays documented rather than silently "fixed" — see the module docstring.
    """
    h = _heads()
    n_a, d_h = PAPER["n_a"], PAPER["d_h"]

    at_spec_rank = h.spec_param_count("low_rank", n_a=n_a, d_h=d_h, rank=14)
    assert at_spec_rank == 14_476
    assert at_spec_rank != h.SPEC_TABLE3_PARAMS["low_rank"]

    # Only r=20 reproduces the tabled number.
    assert h.spec_param_count("low_rank", n_a=n_a, d_h=d_h, rank=20) == 20_680

    # The stated hyperparameter is what we default to.
    assert h.LOW_RANK_DEFAULT_RANK == 14


def test_block_diag_10_keeps_exactly_d_h_params_at_both_shapes():
    """"One block per logit" lands on exactly d_h params at n_a=10 AND at our n_a=6."""
    h = _heads()
    assert h.spec_param_count("block_diag_10", n_a=PAPER["n_a"], d_h=PAPER["d_h"]) == 1024
    assert h.spec_param_count("block_diag_10", n_a=OURS["n_a"], d_h=OURS["d_h"]) == 1024


@pytest.mark.parametrize("variant", VARIANTS)
def test_built_module_matches_the_closed_form_count(variant):
    """The closed form must agree with the modules actually allocated."""
    h = _heads()
    head = h.make_head(variant, **OURS)
    expected = h.spec_param_count(variant, n_a=OURS["n_a"], d_h=OURS["d_h"])
    n = sum(int(p.numel()) for p in head.parameters() if p.requires_grad)
    assert n == expected


def test_param_counts_covers_every_registered_variant():
    h = _heads()
    counts = h.param_counts(n_a=OURS["n_a"], d_h=OURS["d_h"])
    assert set(counts) == set(VARIANTS)
    assert counts["linear"] == 6 * 1024
    # Table 3's headline: block-diag-10 is a 6x reduction at n_a=6 (10x at n_a=10).
    assert counts["linear"] // counts["block_diag_10"] == 6


def test_describe_lists_every_variant_largest_first():
    h = _heads()
    text = h.describe(h.param_counts(n_a=OURS["n_a"], d_h=OURS["d_h"]))
    for name in VARIANTS:
        assert name in text
    assert text.index("sparse") < text.index("block_diag_10")


# --------------------------------------------------------------------------------------
# The registry must not drift from the shipped linear head
# --------------------------------------------------------------------------------------


def test_registry_linear_matches_head_py():
    """``make_head("linear")`` must BE the shipped LinearHead, not a re-implementation."""
    torch = _torch()
    head = _heads().make_head("linear", **OURS)
    assert isinstance(head, _LinearHead())

    reference = _LinearHead()(**OURS)
    W = torch.randn(OURS["n_a"], OURS["d_h"], generator=torch.Generator().manual_seed(7))
    head.load_weight(W)
    reference.load_weight(W)

    h = _rand_h(OURS["d_h"], seed=1)
    for a, b in zip(head(h), reference(h)):
        assert torch.equal(a, b)


def test_default_variant_is_linear():
    assert isinstance(_heads().make_head(**OURS), _LinearHead())


def test_unknown_variant_raises_with_the_valid_names():
    h = _heads()
    with pytest.raises(ValueError, match="unknown head variant"):
        h.make_head("mlp", **OURS)
    with pytest.raises(ValueError, match="unknown head variant"):
        h.spec_param_count("mlp")


# --------------------------------------------------------------------------------------
# from_config — the first reader of the coordinator.head block
# --------------------------------------------------------------------------------------


def _head_cfg(**overrides):
    base = {
        "type": "linear",
        "hidden_dim": 0,
        "n_models": 3,
        "n_roles": 3,
        "n_a": 6,
        "include_stop_action": False,
        "factorize": "two_softmax",
    }
    base.update(overrides)
    return {"coordinator": {"head": base}}


def test_from_config_builds_the_shipped_config_unchanged():
    """The real configs/trinity.yaml must round-trip to today's linear head."""
    yaml = pytest.importorskip("yaml")

    cfg = yaml.safe_load(pathlib.Path("configs/trinity.yaml").read_text())
    head = _heads().from_config(cfg)
    assert isinstance(head, _LinearHead())
    assert (head.n_a, head.d_h, head.n_models) == (6, 1024, 3)


def test_from_config_honours_the_type_knob():
    """head.type had zero readers; it now actually selects the variant."""
    h = _heads()
    assert isinstance(h.from_config(_head_cfg(type="sparse")), h.SparseHead)
    assert isinstance(h.from_config(_head_cfg(type="low_rank")), h.LowRankHead)
    assert isinstance(h.from_config(_head_cfg(type="block_diag_2")), h.BlockDiagonalHead)
    with pytest.raises(ValueError, match="unknown head variant"):
        h.from_config(_head_cfg(type="mlp"))


def test_from_config_rejects_a_hidden_layer():
    """hidden_dim>0 describes a head no SPEC §3.5 variant implements — not a silent no-op."""
    with pytest.raises(ValueError, match="no SPEC .3.5 head has a hidden layer"):
        _heads().from_config(_head_cfg(hidden_dim=64))


def test_from_config_rejects_a_stop_action():
    with pytest.raises(ValueError, match="include_stop_action"):
        _heads().from_config(_head_cfg(include_stop_action=True))


def test_from_config_rejects_an_unimplemented_factorization():
    with pytest.raises(ValueError, match="two_softmax"):
        _heads().from_config(_head_cfg(factorize="one_softmax"))


def test_from_config_defaults_n_a_from_n_models():
    head = _heads().from_config({"coordinator": {"head": {"n_models": 3}}})
    assert head.n_a == 6


def test_from_config_tolerates_a_missing_coordinator_block():
    assert isinstance(_heads().from_config({}), _LinearHead())


# --------------------------------------------------------------------------------------
# The submission path is untouched
# --------------------------------------------------------------------------------------


def test_submission_dimension_is_still_frozen_at_13312():
    """This PR must not move n_total: make_spec is linear-only and feeds submission."""
    from trinity.coordinator.params import make_spec

    assert make_spec().n_total == 13_312


def test_policy_still_builds_the_linear_head():
    """No variant is wired into the decision path — the default is unchanged."""
    source = pathlib.Path("src/trinity/coordinator/policy.py").read_text()
    assert "LinearHead" in source


# --------------------------------------------------------------------------------------
# Shared head contract: shapes, grouping, selection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_splits_into_agent_and_role_groups(variant):
    torch = _torch()
    head = _heads().make_head(variant, **OURS)
    with torch.no_grad():
        agent, role = head(_rand_h(OURS["d_h"], seed=2))
    assert agent.shape == (OURS["n_models"],)
    assert role.shape == (len(ROLE_ORDER),)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_preserves_leading_batch_dims(variant):
    torch = _torch()
    head = _heads().make_head(variant, **OURS)
    with torch.no_grad():
        agent, role = head(_rand_h(OURS["d_h"], batch=(4, 3), seed=3))
    assert agent.shape == (4, 3, OURS["n_models"])
    assert role.shape == (4, 3, len(ROLE_ORDER))


@pytest.mark.parametrize("variant", VARIANTS)
def test_select_returns_an_in_range_agent_and_a_real_role(variant):
    head = _heads().make_head(variant, **OURS)
    h = _rand_h(OURS["d_h"], seed=4)
    idx, role, dbg = head.select(h, sample=False)
    assert 0 <= idx < OURS["n_models"]
    assert role in ROLE_ORDER
    assert dbg["sampled"] is False
    assert dbg["agent_probs"].shape == (OURS["n_models"],)
    assert dbg["agent_probs"].sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_select_argmax_is_deterministic(variant):
    """Eval-time selection must be reproducible — including for the stochastic gate."""
    head = _heads().make_head(variant, **OURS)
    h = _rand_h(OURS["d_h"], seed=5)
    first = head.select(h, sample=False)[:2]
    for _ in range(5):
        assert head.select(h, sample=False)[:2] == first


@pytest.mark.parametrize("variant", VARIANTS)
def test_select_rejects_a_real_batch(variant):
    head = _heads().make_head(variant, **OURS)
    with pytest.raises(ValueError, match="single hidden state"):
        head.select(_rand_h(OURS["d_h"], batch=(4,), seed=6), sample=False)


@pytest.mark.parametrize("variant", ["low_rank", "sparse", "block_diag_2", "block_diag_10"])
def test_role_group_width_is_validated(variant):
    """n_a - n_models must equal len(ROLE_ORDER); a bad split is a construction error."""
    with pytest.raises(ValueError, match="ROLE_ORDER"):
        _heads().make_head(variant, n_a=6, d_h=64, n_models=2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gradients_reach_every_trainable_parameter(variant):
    torch = _torch()
    head = _heads().make_head(variant, **OURS)
    agent, role = head(_rand_h(OURS["d_h"], seed=7))
    (agent.sum() + role.sum()).backward()
    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        # rho is the one documented exception — see test_sparse_rho_has_no_gradient_path.
        if variant == "sparse" and name == "rho":
            continue
        assert p.grad is not None, f"{variant}.{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{variant}.{name} got a non-finite gradient"


@pytest.mark.parametrize("variant", ZERO_INIT)
def test_zero_init_gives_the_uniform_policy(variant):
    """Zero-initialized heads must start uniform, matching params.initial_theta."""
    torch = _torch()
    head = _heads().make_head(variant, **OURS)
    with torch.no_grad():
        _, _, dbg = head.select(_rand_h(OURS["d_h"], seed=8), sample=False)
    assert dbg["agent_probs"] == pytest.approx(
        [1.0 / OURS["n_models"]] * OURS["n_models"], abs=1e-6
    )


# --------------------------------------------------------------------------------------
# Block-diagonal
# --------------------------------------------------------------------------------------


def test_block_diagonal_output_chunk_depends_only_on_its_own_input_chunk():
    """The defining structural property: block b of z sees only block b of h."""
    torch = _torch()
    head = _heads().BlockDiagonalHead(n_a=6, d_h=8, n_models=3, n_blocks=2)
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for block in head.blocks:
            block.normal_(generator=g)

    h = _rand_h(8, seed=10)
    with torch.no_grad():
        base = torch.cat(head(h))
        bumped = h.clone()
        bumped[: head.in_sizes[0]] += 5.0  # perturb ONLY the first input chunk
        after = torch.cat(head(bumped))

    out = head.out_per_block
    assert not torch.allclose(base[:out], after[:out])       # own block moved
    assert torch.allclose(base[out:], after[out:])           # every other block is inert


def test_block_diagonal_is_an_exact_parameter_reduction():
    BlockDiagonalHead = _heads().BlockDiagonalHead
    linear_n = 6 * 1024
    for b in (2, 3, 6):
        head = BlockDiagonalHead(n_a=6, d_h=1024, n_models=3, n_blocks=b)
        assert head.n_params() == linear_n // b


def test_block_diagonal_requires_the_block_count_to_divide_n_a_only():
    BlockDiagonalHead = _heads().BlockDiagonalHead
    with pytest.raises(ValueError, match="must divide n_a"):
        BlockDiagonalHead(n_a=6, d_h=1024, n_models=3, n_blocks=4)  # 4 ∤ 6
    with pytest.raises(ValueError, match="n_blocks must be >= 1"):
        BlockDiagonalHead(n_a=6, d_h=1024, n_models=3, n_blocks=0)


def test_block_diagonal_allows_an_indivisible_d_h():
    """1024 is not divisible by 6 (or 10) — the paper's own shape. It must still build."""
    head = _heads().BlockDiagonalHead(n_a=6, d_h=1024, n_models=3, n_blocks=6)
    assert sum(head.in_sizes) == 1024          # partitions every input, none dropped
    assert max(head.in_sizes) - min(head.in_sizes) <= 1  # near-equal
    assert head.n_params() == 1024             # still the exact SPEC count


def test_only_one_block_per_logit_prefers_argmax():
    """SPEC pairs argmax output with block-diag-10 (= one block per logit), not B=2."""
    h = _heads()
    assert h.make_head("block_diag_10", **OURS).prefers_argmax is True
    assert h.make_head("block_diag_2", **OURS).prefers_argmax is False


# --------------------------------------------------------------------------------------
# Sparse
# --------------------------------------------------------------------------------------


def test_sparse_keep_k_follows_the_spec_formula():
    """k = max(1, floor(d_h * (1 - sigmoid(rho))))."""
    SparseHead = _heads().SparseHead
    for rho in (-2.0, 0.0, 1.5):
        head = SparseHead(n_a=6, d_h=1024, n_models=3, rho=rho)
        expected = max(1, int(math.floor(1024 * (1.0 - 1.0 / (1.0 + math.exp(-rho))))))
        assert head.keep_k() == expected

    assert SparseHead(n_a=6, d_h=1024, n_models=3, rho=0.0).keep_k() == 512
    # Even a saturating rho must keep at least one edge.
    assert SparseHead(n_a=6, d_h=32, n_models=3, rho=50.0).keep_k() == 1


def test_sparse_hard_gate_keeps_exactly_k_dimensions():
    """"Hard top-k at inference" — the eval gate is exactly k non-zeros, no noise."""
    torch = _torch()
    head = _heads().SparseHead(n_a=6, d_h=64, n_models=3, rho=0.0)
    with torch.no_grad():
        head.alpha.normal_(generator=torch.Generator().manual_seed(11))
        gate = head.gate(hard=True)
    assert int((gate != 0).sum()) == head.keep_k() == 32


def test_sparse_hard_gate_selects_the_largest_alphas():
    torch = _torch()
    head = _heads().SparseHead(n_a=6, d_h=8, n_models=3, rho=0.0)
    with torch.no_grad():
        head.alpha.copy_(torch.tensor([5.0, -1.0, 4.0, 0.0, 3.0, -2.0, 6.0, 1.0]))
        live = head.gate(hard=True) != 0
    # k=4 → the four largest alphas are at indices 0, 2, 4, 6.
    assert sorted(torch.nonzero(live).flatten().tolist()) == [0, 2, 4, 6]


def test_sparse_relaxed_gate_is_stochastic_but_hard_gate_is_not():
    torch = _torch()
    head = _heads().SparseHead(n_a=6, d_h=64, n_models=3, rho=0.0)
    with torch.no_grad():
        head.alpha.normal_(generator=torch.Generator().manual_seed(12))
        soft = [head.gate(hard=False) for _ in range(2)]
        hard = [head.gate(hard=True) for _ in range(2)]
    assert not torch.allclose(soft[0], soft[1])  # Gumbel noise resampled
    assert torch.equal(hard[0], hard[1])         # deterministic at inference


def test_sparse_rho_has_no_gradient_path():
    """rho sets an INTEGER k through a floor, so its gradient is structurally zero.

    Documented rather than hidden: rho is still counted (SPEC's ``+2``) and still
    settable, and the gradient-free optimizers this repo uses for the head can search it
    — but no relaxation of the mask can make ``how many`` edges survive differentiable.
    """
    head = _heads().SparseHead(**OURS)
    agent, role = head(_rand_h(OURS["d_h"], seed=18))
    (agent.sum() + role.sum()).backward()

    assert head.rho.requires_grad          # counted as a parameter, per SPEC's +2
    assert head.rho.grad is None           # ...but never receives one
    assert head.tau.grad is not None       # tau, in contrast, IS gradient-trainable


def test_sparse_tau_must_lie_in_the_spec_range():
    SparseHead = _heads().SparseHead
    for bad in (0.5, 25.0):
        with pytest.raises(ValueError, match=r"tau must lie in"):
            SparseHead(n_a=6, d_h=64, n_models=3, tau=bad)
    SparseHead(n_a=6, d_h=64, n_models=3, tau=1.0)
    SparseHead(n_a=6, d_h=64, n_models=3, tau=20.0)


def test_sparse_gate_is_the_identity_when_nothing_is_dropped():
    """rho → -inf keeps every dimension, so the head degenerates to plain linear."""
    torch = _torch()
    head = _heads().SparseHead(n_a=6, d_h=16, n_models=3, rho=-50.0)
    assert head.keep_k() == 16
    with torch.no_grad():
        head.weight.normal_(generator=torch.Generator().manual_seed(13))
        h = _rand_h(16, seed=14)
        got = torch.cat(head(h))
        want = head.weight @ h  # alpha is ones, mask is all-ones
    assert torch.allclose(got, want, atol=1e-6)


# --------------------------------------------------------------------------------------
# Low-rank
# --------------------------------------------------------------------------------------


def test_low_rank_factors_have_the_spec_xavier_bounds():
    """U ~ U[+-sqrt(6/(d_h+r))], V ~ U[+-sqrt(18/(r+n_a))] — SPEC §3.5, verbatim."""
    torch = _torch()
    n_a, d_h, r = 6, 1024, 14
    head = _heads().LowRankHead(
        n_a=n_a, d_h=d_h, n_models=3, rank=r, generator=torch.Generator().manual_seed(15)
    )
    bound_u = math.sqrt(6.0 / (d_h + r))
    bound_v = math.sqrt(18.0 / (r + n_a))

    assert head.U.abs().max().item() <= bound_u
    assert head.V.abs().max().item() <= bound_v
    # Not degenerate: a uniform draw should populate most of its range.
    assert head.U.abs().max().item() > 0.5 * bound_u
    assert head.V.abs().max().item() > 0.5 * bound_v


def test_low_rank_applies_elu_with_alpha_point_one_then_the_fixed_sigma():
    """z = (V · ELU(U·h, alpha=0.1)) * sigma, computed by hand."""
    torch = _torch()
    head = _heads().LowRankHead(n_a=6, d_h=8, n_models=3, rank=3, sigma=2.5)
    h = _rand_h(8, seed=16)
    with torch.no_grad():
        u = torch.nn.functional.elu(head.U @ h, alpha=0.1)
        want = (head.V @ u) * 2.5
        got = torch.cat(head(h))
    assert torch.allclose(got, want, atol=1e-6)
    assert head.elu_alpha == 0.1


def test_low_rank_is_rank_limited():
    """The map factors through R^r, so the effective logit map has rank <= r."""
    torch = _torch()
    r = 3
    head = _heads().LowRankHead(
        n_a=6, d_h=32, n_models=3, rank=r, generator=torch.Generator().manual_seed(17)
    )
    # ELU is linear on the positive orthant, so on strictly positive pre-activations
    # the effective map is exactly V @ U.
    with torch.no_grad():
        effective = head.V @ head.U
    assert torch.linalg.matrix_rank(effective).item() <= r


def test_low_rank_rejects_a_degenerate_rank():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        _heads().LowRankHead(n_a=6, d_h=32, n_models=3, rank=0)


def test_low_rank_rank20_reproduces_the_table3_module_size():
    h = _heads()
    head = h.LowRankHead(n_a=10, d_h=1024, n_models=7, rank=20)
    assert head.n_params() == h.SPEC_TABLE3_PARAMS["low_rank"]
