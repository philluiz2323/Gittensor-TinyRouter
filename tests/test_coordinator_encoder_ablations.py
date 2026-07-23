"""Encoder-side R9 ablations (``no_svf`` / ``last_token``).

Pure-numpy tests: no torch is imported here, matching the module under test.
Tensor-level position parity lives in ``test_torch_encoder_ablation_positions.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity.coordinator import params as _params
from trinity.coordinator.encoder_ablations import (
    ENCODER_ABLATIONS,
    LAST_TOKEN,
    PENULTIMATE_TOKEN,
    TOKEN_POSITIONS,
    ablate_svf,
    is_svf_ablated,
    make_ablated_encoder,
    select_token_position,
    token_index_for,
)

#: Child interpreters must import *this* checkout, not an installed copy.
_SRC = str(Path(__file__).resolve().parents[1] / "src")

N_A, D_H, N_SVF = 6, 8, 5


def _probe(code: str) -> str:
    env = {
        **os.environ,
        "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    return out.stdout.strip()


def _spec():
    return _params.make_spec(n_a=N_A, d_h=D_H, n_svf=N_SVF)


def _theta(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N_A, D_H))
    scales = rng.normal(size=N_SVF) + 2.0  # deliberately not 1.0
    return _params.pack(W, scales), _spec()


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_variant_names_are_the_r9_encoder_side_names():
    assert set(ENCODER_ABLATIONS) == {"no_svf", "last_token"}


def test_token_positions_pin_the_spec_choice():
    assert PENULTIMATE_TOKEN == -2
    assert LAST_TOKEN == -1
    assert TOKEN_POSITIONS["full"] == PENULTIMATE_TOKEN
    assert TOKEN_POSITIONS["last_token"] == LAST_TOKEN


def test_last_token_is_the_only_variant_that_moves_the_read_position():
    assert token_index_for("full") == PENULTIMATE_TOKEN
    assert token_index_for("no_svf") == PENULTIMATE_TOKEN
    assert token_index_for("last_token") == LAST_TOKEN


def test_token_index_for_rejects_unknown_variant():
    with pytest.raises(KeyError, match="unknown variant"):
        token_index_for("no_thinker")


# --------------------------------------------------------------------------
# ablate_svf
# --------------------------------------------------------------------------


def test_ablate_svf_sets_every_scale_to_one():
    theta, spec = _theta()
    _W, scales = _params.unpack(ablate_svf(theta, spec), spec)
    assert scales.tolist() == [1.0] * N_SVF


def test_ablate_svf_leaves_the_head_block_untouched():
    theta, spec = _theta(1)
    before, _ = _params.unpack(theta, spec)
    after, _ = _params.unpack(ablate_svf(theta, spec), spec)
    assert after.tolist() == before.tolist()


def test_ablate_svf_preserves_theta_width():
    """Ablated runs must stay comparable with the full model's θ."""
    theta, spec = _theta(2)
    assert ablate_svf(theta, spec).shape == (spec.n_total,)


def test_ablate_svf_does_not_mutate_its_input():
    theta, spec = _theta(3)
    original = theta.copy()
    ablate_svf(theta, spec)
    assert theta.tolist() == original.tolist()


def test_ablate_svf_is_idempotent():
    theta, spec = _theta(4)
    once = ablate_svf(theta, spec)
    assert ablate_svf(once, spec).tolist() == once.tolist()


def test_ablate_svf_matches_the_adapters_identity_vector():
    """The ablation must be exactly what SVFAdapter calls 'no adaptation'."""
    theta, spec = _theta(5)
    _W, scales = _params.unpack(ablate_svf(theta, spec), spec)
    # SVFAdapter.identity_scales() is np.ones(num_scales, dtype=float64).
    assert scales.tolist() == np.ones(N_SVF, dtype=np.float64).tolist()


def test_ablate_svf_rejects_a_mismatched_theta():
    _theta_ok, spec = _theta(6)
    with pytest.raises(ValueError):
        ablate_svf(np.zeros(spec.n_total + 3), spec)


# --------------------------------------------------------------------------
# is_svf_ablated
# --------------------------------------------------------------------------


def test_is_svf_ablated_true_after_ablation():
    theta, spec = _theta(7)
    assert is_svf_ablated(ablate_svf(theta, spec), spec) is True


def test_is_svf_ablated_false_for_a_trained_theta():
    theta, spec = _theta(8)
    assert is_svf_ablated(theta, spec) is False


def test_is_svf_ablated_is_exact_not_approximate():
    """A run whose scales merely sit near 1 is NOT a no_svf run."""
    spec = _spec()
    W = np.zeros((N_A, D_H))
    nearly = np.ones(N_SVF)
    nearly[2] = 1.0 + 1e-9
    assert is_svf_ablated(_params.pack(W, nearly), spec) is False


# --------------------------------------------------------------------------
# select_token_position
# --------------------------------------------------------------------------


def test_select_reads_the_penultimate_row():
    hs = np.arange(24, dtype=np.float64).reshape(1, 4, 6)
    assert select_token_position(hs, PENULTIMATE_TOKEN).tolist() == hs[0, -2, :].tolist()


def test_select_reads_the_last_row_for_the_ablation():
    hs = np.arange(24, dtype=np.float64).reshape(1, 4, 6)
    assert select_token_position(hs, LAST_TOKEN).tolist() == hs[0, -1, :].tolist()


def test_the_two_positions_actually_differ():
    """If these ever coincided the ablation would measure nothing."""
    hs = np.arange(24, dtype=np.float64).reshape(1, 4, 6)
    a = select_token_position(hs, PENULTIMATE_TOKEN)
    b = select_token_position(hs, LAST_TOKEN)
    assert a.tolist() != b.tolist()


def test_select_reads_only_the_first_batch_row():
    hs = np.arange(48, dtype=np.float64).reshape(2, 4, 6)
    assert select_token_position(hs, LAST_TOKEN).tolist() == hs[0, -1, :].tolist()


def test_select_rejects_non_3d_hidden_states():
    with pytest.raises(ValueError, match="batch, seq_len, hidden"):
        select_token_position(np.zeros((4, 6)), -2)


def test_select_rejects_a_too_short_sequence():
    hs = np.zeros((1, 1, 6))
    with pytest.raises(ValueError, match="too short"):
        select_token_position(hs, PENULTIMATE_TOKEN)


def test_select_allows_last_index_on_a_length_one_sequence():
    hs = np.arange(6, dtype=np.float64).reshape(1, 1, 6)
    assert select_token_position(hs, LAST_TOKEN).tolist() == hs[0, -1, :].tolist()


# --------------------------------------------------------------------------
# encoder wiring (no model is loaded)
# --------------------------------------------------------------------------


def test_encoder_default_read_position_is_the_spec_penultimate_token():
    """The shipped default must not move; this is the un-ablated behaviour."""
    import inspect

    from trinity.coordinator.slm import CoordinatorEncoder

    param = inspect.signature(CoordinatorEncoder.__init__).parameters["token_index"]
    assert param.default == -2


def test_from_config_on_the_shipped_config_keeps_the_penultimate_token(monkeypatch):
    """Reading the repo's own config must reproduce today's encoder exactly."""
    from trinity.coordinator import slm as _slm

    seen = {}

    def _fake_init(self, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(_slm.CoordinatorEncoder, "__init__", _fake_init)
    _slm.CoordinatorEncoder.from_config()
    assert seen["token_index"] == -2


def test_from_config_honours_an_explicit_token_index(tmp_path, monkeypatch):
    from trinity.coordinator import slm as _slm

    cfg = tmp_path / "trinity.yaml"
    cfg.write_text(
        "coordinator:\n"
        "  encoder_model: Qwen/Qwen3-0.6B\n"
        "  device: cuda:0\n"
        "  dtype: bfloat16\n"
        "  hidden_state:\n"
        "    l2_normalize: true\n"
        "    token_index: -1\n"
    )
    seen = {}

    def _fake_init(self, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(_slm.CoordinatorEncoder, "__init__", _fake_init)
    _slm.CoordinatorEncoder.from_config(cfg)
    assert seen["token_index"] == -1


def test_make_ablated_encoder_rejects_a_conflicting_token_index():
    """Both a variant and an explicit index means two intentions for one knob."""
    with pytest.raises(TypeError, match="determined by the variant"):
        make_ablated_encoder("last_token", token_index=-1)


def test_make_ablated_encoder_rejects_an_unknown_variant():
    with pytest.raises(KeyError, match="unknown variant"):
        make_ablated_encoder("no_thinker")


# --------------------------------------------------------------------------
# import cost
# --------------------------------------------------------------------------


def test_importing_the_module_does_not_import_torch():
    got = _probe(
        "import sys; import trinity.coordinator.encoder_ablations as m; "
        "print('torch' in sys.modules)"
    )
    assert got == "False", got


def test_rejection_paths_do_not_import_torch():
    """Validation must fail before the lazy encoder import is reached."""
    got = _probe(
        "import sys\n"
        "from trinity.coordinator.encoder_ablations import make_ablated_encoder\n"
        "for bad in (dict(variant='no_thinker'), None):\n"
        "    try:\n"
        "        if bad is None:\n"
        "            make_ablated_encoder('last_token', token_index=-1)\n"
        "        else:\n"
        "            make_ablated_encoder(bad['variant'])\n"
        "    except (KeyError, TypeError):\n"
        "        pass\n"
        "print('torch' in sys.modules)\n"
    )
    assert got == "False", got
