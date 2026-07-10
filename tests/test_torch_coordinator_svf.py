"""Offline coverage for the SVF singular-value adapter (coordinator/svf.py).

`SVFAdapter` is the second half of the CMA-ES search vector: it SVD-decomposes a
fixed set of linear matrices in one transformer block, freezes ``U``/``Vh``, and
exposes the singular-value *scales* as the learnable parameters, reconstructing
``W' = U @ diag(s * scale_block) @ Vh`` in place. It was at **0%** coverage.

It needs `torch` but no HuggingFace checkpoint: a tiny fake model with the Qwen3
module layout (``model.model.layers[i].self_attn.{q,k,v,o}_proj`` +
``.mlp.{gate,up,down}_proj``) exercises every path with real SVDs on small
matrices. Using non-1024 dims also proves ``num_scales`` is computed from the real
SVD shapes rather than hardcoded to 7168.

Like `test_torch_coordinator_head.py`, `torch` and `SVFAdapter` are imported
lazily and this file is named `test_torch_*` so it sorts AFTER
`test_shaped_fitness.py::test_no_torch_imported` (which asserts torch stays out of
the global `sys.modules`). See that file's docstring for the full rationale.
"""
from __future__ import annotations

import numpy as np
import pytest


def _torch():
    return pytest.importorskip("torch", reason="torch required for SVFAdapter")


def _SVFAdapter():
    from trinity.coordinator.svf import SVFAdapter

    return SVFAdapter


# Small dims so SVDs are cheap; each proj yields min(out,in) = D singular values,
# so the default 7-matrix set has num_scales = 7 * D (56 for D=8) -- NOT 7168.
_D = 8


def _fake_model(d: int = _D, n_layers: int = 3):
    """A minimal module tree with the Qwen3 layout SVFAdapter walks."""
    torch = _torch()
    nn = torch.nn

    def attn():
        m = nn.Module()
        m.q_proj = nn.Linear(d, 2 * d, bias=False)
        m.k_proj = nn.Linear(d, d, bias=False)
        m.v_proj = nn.Linear(d, d, bias=False)
        m.o_proj = nn.Linear(2 * d, d, bias=False)
        return m

    def mlp():
        m = nn.Module()
        m.gate_proj = nn.Linear(d, 3 * d, bias=False)
        m.up_proj = nn.Linear(d, 3 * d, bias=False)
        m.down_proj = nn.Linear(3 * d, d, bias=False)
        return m

    layers = []
    for _ in range(n_layers):
        layer = nn.Module()
        layer.self_attn = attn()
        layer.mlp = mlp()
        layers.append(layer)

    inner = nn.Module()
    inner.layers = nn.ModuleList(layers)
    model = nn.Module()
    model.model = inner
    # Deterministic, non-trivial weights.
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.normal_(p)
    return model


def _adapter(model=None, **kw):
    model = model if model is not None else _fake_model()
    return _SVFAdapter()(model, target_layer=kw.pop("target_layer", 1), **kw), model


def _weight(model, layer, parent, name):
    return getattr(getattr(model.model.layers[layer], parent), name).weight


# --------------------------------------------------------------------------- #
# Construction / layout
# --------------------------------------------------------------------------- #
def test_num_scales_is_summed_from_real_shapes():
    adapter, _ = _adapter()
    # 7 matrices x min-dim D singular values each; explicitly NOT 7168.
    assert adapter.num_scales == 7 * _D
    assert adapter.num_scales != 7168


def test_scale_slices_are_contiguous_and_cover_the_vector():
    adapter, _ = _adapter()
    ordered = [adapter.scale_slices[n] for n in adapter.matrix_names]
    assert ordered[0][0] == 0
    for (_, end), (start, _) in zip(ordered, ordered[1:]):
        assert end == start
    assert ordered[-1][1] == adapter.num_scales
    assert sum(e - s for s, e in ordered) == adapter.num_scales


def test_default_targets_are_all_seven_matrices_in_order():
    adapter, _ = _adapter()
    assert adapter.matrix_names == (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )


def test_matrix_subset_preserves_caller_order():
    model = _fake_model()
    adapter = _SVFAdapter()(model, target_layer=1, matrices=["v_proj", "q_proj"])
    assert adapter.matrix_names == ("v_proj", "q_proj")
    assert set(adapter.scale_slices) == {"v_proj", "q_proj"}
    assert adapter.num_scales == 2 * _D


def test_unknown_matrix_name_raises():
    model = _fake_model()
    with pytest.raises(KeyError, match="Unknown SVF matrix"):
        _SVFAdapter()(model, target_layer=1, matrices=["not_a_proj"])


# --------------------------------------------------------------------------- #
# set_scales reconstruction
# --------------------------------------------------------------------------- #
def test_identity_scales_round_trip_to_original_weight():
    torch = _torch()
    adapter, model = _adapter()
    w0 = _weight(model, 1, "self_attn", "q_proj").detach().clone()

    adapter.set_scales(adapter.identity_scales())

    assert torch.allclose(_weight(model, 1, "self_attn", "q_proj"), w0, atol=1e-5)


def test_doubling_a_block_doubles_only_that_matrix():
    torch = _torch()
    adapter, model = _adapter()
    q0 = _weight(model, 1, "self_attn", "q_proj").detach().clone()
    k0 = _weight(model, 1, "self_attn", "k_proj").detach().clone()

    scales = adapter.identity_scales()
    s, e = adapter.scale_slices["q_proj"]
    scales[s:e] = 2.0
    adapter.set_scales(scales)

    assert torch.allclose(_weight(model, 1, "self_attn", "q_proj"), 2.0 * q0, atol=1e-5)
    # k_proj's block stayed at identity, so its weight is unchanged.
    assert torch.allclose(_weight(model, 1, "self_attn", "k_proj"), k0, atol=1e-5)


def test_set_scales_does_not_compound():
    """Reconstruction uses the frozen s, so 3x then 2x gives 2x, not 6x."""
    torch = _torch()
    adapter, model = _adapter()
    q0 = _weight(model, 1, "self_attn", "q_proj").detach().clone()
    s, e = adapter.scale_slices["q_proj"]

    three = adapter.identity_scales()
    three[s:e] = 3.0
    adapter.set_scales(three)
    two = adapter.identity_scales()
    two[s:e] = 2.0
    adapter.set_scales(two)

    assert torch.allclose(_weight(model, 1, "self_attn", "q_proj"), 2.0 * q0, atol=1e-5)


def test_set_scales_wrong_length_raises():
    adapter, _ = _adapter()
    with pytest.raises(ValueError, match="Expected"):
        adapter.set_scales(np.ones(adapter.num_scales + 1))


def test_set_scales_accepts_a_python_list_length_match():
    torch = _torch()
    adapter, model = _adapter()
    w0 = _weight(model, 1, "self_attn", "q_proj").detach().clone()
    adapter.set_scales([1.0] * adapter.num_scales)  # reshaped/asarray'd internally
    assert torch.allclose(_weight(model, 1, "self_attn", "q_proj"), w0, atol=1e-5)


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #
def test_reset_restores_bit_identical_weight():
    torch = _torch()
    adapter, model = _adapter()
    q0 = _weight(model, 1, "self_attn", "q_proj").detach().clone()

    scales = adapter.identity_scales()
    scales[:] = np.linspace(0.5, 1.5, adapter.num_scales)
    adapter.set_scales(scales)
    assert not torch.allclose(_weight(model, 1, "self_attn", "q_proj"), q0)

    adapter.reset()
    # reset() copies the cached pristine w0 -> exactly equal, no SVD drift.
    assert torch.equal(_weight(model, 1, "self_attn", "q_proj"), q0)


def test_reset_restores_every_targeted_matrix():
    torch = _torch()
    adapter, model = _adapter()
    originals = {
        (parent, name): _weight(model, 1, parent, name).detach().clone()
        for parent, name in (
            ("self_attn", "q_proj"), ("self_attn", "o_proj"), ("mlp", "down_proj"),
        )
    }
    adapter.set_scales(adapter.identity_scales() * 1.7)
    adapter.reset()
    for (parent, name), w0 in originals.items():
        assert torch.equal(_weight(model, 1, parent, name), w0)


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
def test_identity_scales_is_all_ones_of_the_right_length():
    adapter, _ = _adapter()
    ids = adapter.identity_scales()
    assert ids.shape == (adapter.num_scales,)
    assert np.all(ids == 1.0)


def test_describe_returns_a_defensive_copy():
    adapter, _ = _adapter()
    d = adapter.describe()
    assert d == adapter.scale_slices
    d["q_proj"] = (-1, -1)
    assert adapter.scale_slices["q_proj"] != (-1, -1)


def test_target_layer_is_honoured():
    model = _fake_model(n_layers=3)
    a0 = _SVFAdapter()(model, target_layer=0)
    a2 = _SVFAdapter()(model, target_layer=2)
    assert a0.target_layer == 0 and a2.target_layer == 2
