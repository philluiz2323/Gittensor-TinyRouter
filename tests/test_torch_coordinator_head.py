"""Offline coverage for the linear coordinator head (SPEC §3.3, Eq. 5).

`LinearHead` maps the SLM hidden state to two independent logit groups
(agent + role), each with its own softmax, and selects a `(agent_idx, role)` by
argmax (eval) or categorical sample (train). It is pure torch — no HF model, no
network — but had **zero** test coverage. These tests pin every documented
contract: the bias-free `z = W·h`, the `[:n_models]` / `[n_models:]` split, batch
preservation, shape validation, argmax-vs-sample selection, seeded-sample
reproducibility, and the zero-weight uniform-policy start.

`head.py` imports `torch` at module scope, so `LinearHead` and `torch` are
imported *lazily inside the tests*. Even so, once any test here runs, torch is in
`sys.modules` for the rest of the process, and `test_shaped_fitness.py::
test_no_torch_imported` asserts torch stays out of `sys.modules`. That invariant
inspects the *global* module table, so it only holds if every torch-importing
test file sorts **after** `test_shaped_fitness.py` (as `test_warmstart.py` and
`test_slm_head_input_suffix.py` already do). Hence the `test_torch_` prefix on
this filename — it keeps the file after the invariant in pytest's alphabetical
collection order. Do not rename it earlier.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.types import ROLE_ORDER  # torch-free


def _torch():
    return pytest.importorskip("torch", reason="torch required for LinearHead")


def _LinearHead():
    from trinity.coordinator.head import LinearHead

    return LinearHead


def _head(n_a: int = 6, d_h: int = 4, n_models: int = 3):
    return _LinearHead()(n_a=n_a, d_h=d_h, n_models=n_models)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_default_shape_and_zero_init():
    torch = _torch()
    head = _LinearHead()()  # defaults: n_a=6, d_h=1024, n_models=3
    assert tuple(head.weight.shape) == (6, 1024)
    assert int(torch.count_nonzero(head.weight)) == 0  # uniform-policy start
    assert (head.n_a, head.d_h, head.n_models, head.n_roles) == (6, 1024, 3, 3)


def test_role_count_mismatch_raises():
    # n_a - n_models = 4 role logits, but ROLE_ORDER has 3.
    with pytest.raises(ValueError, match="role logits"):
        _head(n_a=6, d_h=4, n_models=2)


def test_custom_dimensions():
    head = _head(n_a=7, d_h=8, n_models=4)
    assert (head.n_models, head.n_roles) == (4, 3)
    assert tuple(head.weight.shape) == (7, 8)


# --------------------------------------------------------------------------- #
# load_weight
# --------------------------------------------------------------------------- #
def test_load_weight_from_numpy_round_trips_through_forward():
    torch = _torch()
    head = _head()
    W = np.arange(24, dtype=np.float32).reshape(6, 4)
    head.load_weight(W)
    assert torch.allclose(head.weight, torch.from_numpy(W))


def test_load_weight_from_tensor():
    torch = _torch()
    head = _head()
    W = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    head.load_weight(W)
    assert torch.allclose(head.weight, W)


def test_load_weight_wrong_shape_raises():
    torch = _torch()
    head = _head()
    with pytest.raises(ValueError, match="weight shape"):
        head.load_weight(torch.zeros(6, 5))


def test_load_weight_casts_to_param_dtype():
    torch = _torch()
    head = _head()
    head.load_weight(np.ones((6, 4), dtype=np.float64))
    assert head.weight.dtype == torch.float32  # param dtype preserved


# --------------------------------------------------------------------------- #
# forward: z = W·h, split into the two groups
# --------------------------------------------------------------------------- #
def test_forward_splits_agent_and_role_logits():
    torch = _torch()
    head = _head()
    W = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    head.load_weight(W)
    h = torch.tensor([1.0, 2.0, 3.0, 4.0])

    agent, role = head.forward(h)
    z = W @ h
    assert torch.allclose(agent, z[:3])
    assert torch.allclose(role, z[3:])


def test_forward_has_no_bias():
    torch = _torch()
    head = _head()  # zero weight
    agent, role = head.forward(torch.randn(4))
    assert torch.allclose(agent, torch.zeros(3))
    assert torch.allclose(role, torch.zeros(3))


def test_forward_preserves_batch_dimension():
    torch = _torch()
    head = _head()
    W = torch.randn(6, 4)
    head.load_weight(W)
    H = torch.randn(5, 4)

    agent, role = head.forward(H)
    assert tuple(agent.shape) == (5, 3)
    assert tuple(role.shape) == (5, 3)
    assert torch.allclose(agent, (H @ W.t())[:, :3])


# --------------------------------------------------------------------------- #
# select: argmax (eval)
# --------------------------------------------------------------------------- #
def _dominant_head(agent_idx: int, role_idx: int):
    """A head whose forward on h=e0 makes `agent_idx` / `role_idx` the argmax."""
    torch = _torch()
    head = _head()
    W = torch.zeros(6, 4)
    W[agent_idx, 0] = 5.0
    W[3 + role_idx, 0] = 5.0
    head.load_weight(W)
    return head, torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_select_argmax_picks_dominant_agent_and_role():
    head, h = _dominant_head(agent_idx=2, role_idx=1)
    idx, role, dbg = head.select(h, sample=False)
    assert idx == 2
    assert role is ROLE_ORDER[1]  # worker
    assert dbg["sampled"] is False


def test_select_maps_role_through_role_order():
    for role_idx in range(len(ROLE_ORDER)):
        head, h = _dominant_head(agent_idx=0, role_idx=role_idx)
        _, role, _ = head.select(h, sample=False)
        assert role is ROLE_ORDER[role_idx]


def test_select_squeezes_leading_batch_of_one():
    head, h = _dominant_head(agent_idx=1, role_idx=0)
    idx, _, _ = head.select(h.unsqueeze(0), sample=False)  # shape (1, 4)
    assert idx == 1


def test_select_rejects_multi_row_batch():
    torch = _torch()
    head = _head()
    with pytest.raises(ValueError, match="single hidden state"):
        head.select(torch.zeros(2, 4), sample=False)


def test_select_debug_dict_is_well_formed():
    torch = _torch()
    head, h = _dominant_head(agent_idx=2, role_idx=1)
    _, _, dbg = head.select(h, sample=False)

    assert set(dbg) >= {
        "agent_logits", "role_logits", "agent_probs", "role_probs",
        "agent_idx", "role_pos", "role", "sampled",
    }
    assert dbg["agent_probs"].shape == (3,)
    assert dbg["role_probs"].shape == (3,)
    assert float(dbg["agent_probs"].sum()) == pytest.approx(1.0)
    assert float(dbg["role_probs"].sum()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# select: zero-weight uniform policy (documented start state)
# --------------------------------------------------------------------------- #
def test_zero_weight_is_uniform_policy():
    torch = _torch()
    head = _head()  # zero weight
    idx, role, dbg = head.select(torch.randn(4), sample=False)
    # all logits equal -> argmax is index 0 in each group
    assert idx == 0
    assert role is ROLE_ORDER[0]
    assert np.allclose(dbg["agent_probs"], 1.0 / 3.0)
    assert np.allclose(dbg["role_probs"], 1.0 / 3.0)


# --------------------------------------------------------------------------- #
# select: categorical sampling (train)
# --------------------------------------------------------------------------- #
def test_sampling_is_reproducible_under_a_seeded_generator():
    torch = _torch()
    head, h = _dominant_head(agent_idx=1, role_idx=2)

    g1 = torch.Generator().manual_seed(0)
    a1, r1, dbg1 = head.select(h, sample=True, rng=g1)
    g2 = torch.Generator().manual_seed(0)
    a2, r2, dbg2 = head.select(h, sample=True, rng=g2)

    assert (a1, r1) == (a2, r2)
    assert dbg1["sampled"] is True


def test_sampling_from_a_near_onehot_picks_the_dominant_index():
    torch = _torch()
    head = _head()
    W = torch.zeros(6, 4)
    W[1, 0] = 100.0        # agent index 1 dominates (softmax ~ 1.0)
    W[3 + 2, 0] = 100.0    # role index 2 dominates
    head.load_weight(W)
    h = torch.tensor([1.0, 0.0, 0.0, 0.0])

    idx, role, _ = head.select(h, sample=True, rng=torch.Generator().manual_seed(7))
    assert idx == 1
    assert role is ROLE_ORDER[2]  # verifier


def test_sampled_indices_are_always_in_range():
    torch = _torch()
    head = _head()
    head.load_weight(torch.randn(6, 4))
    h = torch.randn(4)
    for seed in range(20):
        idx, role, _ = head.select(h, sample=True, rng=torch.Generator().manual_seed(seed))
        assert 0 <= idx < head.n_models
        assert role in ROLE_ORDER


def test_sampling_draws_on_the_generator_device_not_the_heads(monkeypatch):
    """Regression: GPU training crashed every sampled trajectory (issue: device mix).

    ``optim.fitness.evaluate_candidate`` passes a **CPU** ``torch.Generator``
    (from ``optim.sampling.trajectory_sampling_rng``) while the head — and so
    the softmax probs — lives on the training GPU (``configs/trinity.yaml``
    sets ``device: cuda:0``). ``torch.multinomial`` requires its input and its
    generator to share a device, so ``select(..., sample=True)`` raised
    ``RuntimeError: Expected a 'cuda' device type for generator but found
    'cpu'`` on the first turn of every trajectory; the gather's
    ``return_exceptions=True`` swallowed it and every candidate scored 0.

    CI has no GPU, so the mismatch is simulated with a ``meta``-device
    generator stand-in and a ``multinomial`` spy: ``select`` must move the
    probs onto the *generator's* device before drawing.
    """
    torch = _torch()
    head = _head()
    head.load_weight(torch.randn(6, 4))
    h = torch.randn(4)

    class _MetaGenerator:
        """Duck-typed generator pinned to a device the head is not on."""

        device = torch.device("meta")

    seen: list[tuple] = []

    def _spy(probs, num_samples, generator=None):
        seen.append((probs.device, generator.device))
        return torch.zeros(num_samples, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", _spy)
    idx, role, dbg = head.select(h, sample=True, rng=_MetaGenerator())

    assert len(seen) == 2  # one draw per logit group (agent, role)
    for probs_device, gen_device in seen:
        assert probs_device == gen_device == torch.device("meta")
    # The spy always picks index 0; the wiring around it must be intact.
    assert idx == 0
    assert role is ROLE_ORDER[0]
    assert dbg["sampled"] is True


def test_sampling_without_a_generator_stays_on_the_probs_device(monkeypatch):
    """``rng=None`` (eval-time sampling paths) must not move the probs at all."""
    torch = _torch()
    head = _head()
    head.load_weight(torch.randn(6, 4))
    h = torch.randn(4)

    seen: list = []

    def _spy(probs, num_samples, generator=None):
        seen.append(probs.device)
        return torch.zeros(num_samples, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", _spy)
    head.select(h, sample=True, rng=None)

    assert seen == [h.device, h.device]
