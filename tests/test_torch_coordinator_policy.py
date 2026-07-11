"""Offline coverage for CoordinatorPolicy (coordinator/policy.py).

`CoordinatorPolicy` ties the SLM encoder + SVF + linear head into one routing
decision. `build()` needs a real GPU checkpoint, but the class takes its three
components by injection, so `configure()` and `decide()` are testable off-GPU with
a fake encoder + SVF and a real `LinearHead`. The module sat at 25%.

`decide()` imports `torch`, and the real `LinearHead` imports it at module scope,
so — like `test_torch_coordinator_head.py` — `torch` and the classes are imported
lazily and this file is named `test_torch_*` to sort after
`test_shaped_fitness.py::test_no_torch_imported` (which inspects global
`sys.modules`). See that file's docstring for the rationale.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.coordinator import params as P
from trinity.types import ROLE_ORDER, Role


def _torch():
    return pytest.importorskip("torch", reason="torch required for LinearHead")


def _classes():
    from trinity.coordinator.head import LinearHead
    from trinity.coordinator.policy import CoordinatorPolicy

    return CoordinatorPolicy, LinearHead


class _FakeEncoder:
    """Records the transcripts it is asked to encode; returns a fixed vector."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.seen: list[str] = []

    def encode(self, transcript_text):
        self.seen.append(transcript_text)
        return self.vector


class _FakeSVF:
    """Records the scale vector `configure` applies."""

    def __init__(self):
        self.scales = None

    def set_scales(self, scales):
        self.scales = np.asarray(scales)


_N_A, _D_H, _N_MODELS, _N_SVF = 6, 4, 3, 8


def _spec():
    return P.make_spec(n_a=_N_A, d_h=_D_H, n_svf=_N_SVF)


def _policy(encoder_vec=(1.0, 0.0, 0.0, 0.0)):
    CoordinatorPolicy, LinearHead = _classes()
    encoder = _FakeEncoder(encoder_vec)
    svf = _FakeSVF()
    head = LinearHead(n_a=_N_A, d_h=_D_H, n_models=_N_MODELS)
    return CoordinatorPolicy(encoder, svf, head, n_models=_N_MODELS), encoder, svf, head


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_stores_components_and_defaults_spec_to_none():
    policy, encoder, svf, head = _policy()
    assert policy.encoder is encoder
    assert policy.svf is svf
    assert policy.head is head
    assert policy.n_models == _N_MODELS
    assert policy.spec is None


# --------------------------------------------------------------------------- #
# configure
# --------------------------------------------------------------------------- #
def test_configure_writes_head_weight_and_svf_scales():
    torch = _torch()
    policy, _, svf, head = _policy()
    head_W = np.arange(_N_A * _D_H, dtype=np.float64).reshape(_N_A, _D_H)
    svf_scales = np.linspace(0.5, 1.5, _N_SVF)
    theta = P.pack(head_W, svf_scales)

    policy.configure(theta, _spec())

    assert torch.allclose(head.weight, torch.from_numpy(head_W).float(), atol=1e-5)
    assert np.allclose(svf.scales, svf_scales)


def test_configure_uses_self_spec_when_not_passed():
    policy, _, svf, _ = _policy()
    policy.spec = _spec()
    theta = P.initial_theta(policy.spec)

    policy.configure(theta)  # no explicit spec

    assert svf.scales is not None
    assert svf.scales.shape == (_N_SVF,)


def test_configure_without_any_spec_raises():
    policy, _, _, _ = _policy()  # spec is None
    theta = np.zeros(_N_A * _D_H + _N_SVF)
    with pytest.raises(RuntimeError, match="spec is unset"):
        policy.configure(theta)


def test_configure_initial_theta_is_identity_svf_and_zero_head():
    torch = _torch()
    policy, _, svf, head = _policy()
    spec = _spec()
    policy.configure(P.initial_theta(spec), spec)
    # initial_theta = zero head + all-ones SVF scales (unmodified SLM).
    assert torch.count_nonzero(head.weight) == 0
    assert np.allclose(svf.scales, 1.0)


# --------------------------------------------------------------------------- #
# decide
# --------------------------------------------------------------------------- #
def _load_dominant(head, agent_idx: int, role_idx: int):
    """Set head weight so h = e0 makes `agent_idx` / `role_idx` win the argmax."""
    torch = _torch()
    W = torch.zeros(_N_A, _D_H)
    W[agent_idx, 0] = 5.0
    W[_N_MODELS + role_idx, 0] = 5.0
    head.load_weight(W)


def test_decide_returns_selected_agent_and_role():
    policy, encoder, _, head = _policy(encoder_vec=(1.0, 0.0, 0.0, 0.0))
    _load_dominant(head, agent_idx=2, role_idx=1)

    agent_idx, role = policy.decide("QUERY: solve it", sample=False)

    assert agent_idx == 2
    assert role is ROLE_ORDER[1]
    assert isinstance(agent_idx, int)
    assert isinstance(role, Role)


def test_decide_encodes_the_transcript_text():
    policy, encoder, _, head = _policy()
    _load_dominant(head, agent_idx=0, role_idx=0)
    policy.decide("the transcript", sample=False)
    assert encoder.seen == ["the transcript"]


def test_decide_returns_a_plain_two_tuple():
    policy, _, _, head = _policy()
    _load_dominant(head, agent_idx=1, role_idx=2)
    result = policy.decide("x", sample=False)
    assert isinstance(result, tuple) and len(result) == 2


def test_decide_sampling_is_reproducible_with_a_seeded_generator():
    torch = _torch()
    policy, _, _, head = _policy()
    # A near-one-hot head so the sample is stable, but still exercise the rng path.
    _load_dominant(head, agent_idx=1, role_idx=2)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert policy.decide("x", sample=True, rng=g1) == policy.decide("x", sample=True, rng=g2)


def test_decide_default_head_is_uniform_first_choice():
    # Zero head weight -> uniform -> argmax index 0 for both groups.
    policy, _, _, _ = _policy()
    agent_idx, role = policy.decide("anything", sample=False)
    assert agent_idx == 0
    assert role is ROLE_ORDER[0]
