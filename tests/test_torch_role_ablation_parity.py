"""Parity between the ablation wrapper and the real ``LinearHead.select``.

The wrapper reimplements the argmax/sample rule in numpy so the module stays
torch-free at import. That is a drift risk: if ``head.select`` ever changes its
selection rule, the ablated numbers would stop being comparable with the full
model's. These tests pin the two together on the un-ablated path.

torch is imported **inside** each test body (never at module scope, which pytest
would execute at collection time and break
``test_shaped_fitness.py::test_no_torch_imported``); the filename's ``test_torch_``
prefix keeps this file after that guard in alphabetical run order.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.coordinator.ablations import AblatedPolicy, categorical_pick, role_mask
from trinity.types import ROLE_ORDER, Role

N_MODELS = 3
N_ROLES = 3
D_H = 32


def _torch():
    return pytest.importorskip("torch")


def _head(n_models=N_MODELS, d_h=D_H):
    from trinity.coordinator.head import LinearHead

    return LinearHead(n_a=n_models + N_ROLES, d_h=d_h, n_models=n_models)


def _random_head(seed):
    torch = _torch()
    head = _head()
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N_MODELS + N_ROLES, D_H)).astype(np.float64)
    head.load_weight(W)
    torch.manual_seed(seed)
    return head, rng


def test_unmasked_argmax_matches_head_select():
    """The wrapper's pick rule must agree with the shipped head's, exactly."""
    torch = _torch()
    head, rng = _random_head(0)
    for _ in range(100):
        h = rng.normal(size=D_H).astype(np.float32)
        h_t = torch.as_tensor(h)
        agent_idx, role, _dbg = head.select(h_t, sample=False)
        a_logits, r_logits = head.forward(h_t)
        a_np = a_logits.detach().numpy()
        r_np = r_logits.detach().numpy()
        assert categorical_pick(a_np, sample=False) == agent_idx
        assert ROLE_ORDER[categorical_pick(r_np, sample=False)] is role


def test_masked_probs_agree_with_a_torch_softmax_over_survivors():
    torch = _torch()
    from trinity.coordinator.ablations import masked_role_probs

    rng = np.random.default_rng(5)
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    for _ in range(50):
        z = rng.normal(size=N_ROLES)
        got = masked_role_probs(z, mask)
        want = torch.softmax(torch.as_tensor(z[1:]), dim=-1).numpy()
        assert got[1:] == pytest.approx(want, abs=1e-9)
        assert got[0] == 0.0


class _StubEncoder:
    """Stands in for the SLM: returns a fixed hidden state, no model load."""

    def __init__(self, h):
        self._h = np.asarray(h, dtype=np.float32)

    def encode(self, _text: str) -> np.ndarray:
        return self._h


class _StubPolicy:
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head
        self.spec = None
        self.configured = None

    def configure(self, theta, spec=None):
        self.configured = (np.asarray(theta), spec)


def test_wrapper_drives_a_real_head_through_the_encoder_head_path():
    """Exercise ``_logits``' real branch -- encoder + torch head, no logits_fn."""
    head, rng = _random_head(1)
    for _ in range(50):
        h = rng.normal(size=D_H)
        inner = _StubPolicy(_StubEncoder(h), head)
        pol = AblatedPolicy(inner, "no_thinker")
        idx, role = pol.decide("transcript")
        assert role is not Role.THINKER
        assert 0 <= idx < N_MODELS


def test_wrapper_agent_choice_equals_the_unablated_head_choice():
    torch = _torch()
    head, rng = _random_head(2)
    for _ in range(50):
        h = rng.normal(size=D_H)
        full_idx, _role, _dbg = head.select(torch.as_tensor(h.astype(np.float32)), sample=False)
        inner = _StubPolicy(_StubEncoder(h), head)
        for variant in ("no_thinker", "no_trirole"):
            assert AblatedPolicy(inner, variant).decide("t")[0] == full_idx


def test_wrapper_role_equals_head_role_whenever_the_head_already_agrees():
    """When the full model's role already survives the ablation, they must match."""
    torch = _torch()
    head, rng = _random_head(3)
    agreed = 0
    for _ in range(200):
        h = rng.normal(size=D_H)
        _idx, full_role, _dbg = head.select(torch.as_tensor(h.astype(np.float32)), sample=False)
        if full_role is Role.THINKER:
            continue
        inner = _StubPolicy(_StubEncoder(h), head)
        assert AblatedPolicy(inner, "no_thinker").decide("t")[1] is full_role
        agreed += 1
    assert agreed > 0, "no non-Thinker argmax sampled; test proved nothing"


def test_ablation_changes_the_role_when_the_head_wanted_thinker():
    """The complement of the previous test: the ablation must actually bite."""
    torch = _torch()
    head, rng = _random_head(4)
    diverted = 0
    for _ in range(200):
        h = rng.normal(size=D_H)
        _idx, full_role, _dbg = head.select(torch.as_tensor(h.astype(np.float32)), sample=False)
        if full_role is not Role.THINKER:
            continue
        inner = _StubPolicy(_StubEncoder(h), head)
        got = AblatedPolicy(inner, "no_thinker").decide("t")[1]
        assert got in (Role.WORKER, Role.VERIFIER)
        diverted += 1
    assert diverted > 0, "head never chose Thinker; ablation was never exercised"


def test_configure_reaches_the_wrapped_policy_on_the_real_path():
    head, _rng = _random_head(6)
    inner = _StubPolicy(_StubEncoder(np.zeros(D_H)), head)
    pol = AblatedPolicy(inner, "no_trirole")
    pol.configure(np.arange(3, dtype=np.float64))
    assert inner.configured is not None
    assert inner.configured[0].tolist() == [0.0, 1.0, 2.0]


def test_wrapper_satisfies_the_session_policy_protocol():
    """The session calls ``decide(text, sample=..., rng=...)`` -- nothing else."""
    head, _rng = _random_head(7)
    inner = _StubPolicy(_StubEncoder(np.ones(D_H)), head)
    pol = AblatedPolicy(inner, "no_thinker")
    idx, role = pol.decide("t", sample=False, rng=None)
    assert isinstance(idx, int)
    assert isinstance(role, Role)
