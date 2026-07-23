"""Tensor-level parity for the ``last_token`` read position.

``select_token_position`` is written against numpy semantics but is called on
torch tensors in the real path, so these tests drive it with actual tensors and
compare against the literal indexing expression ``slm.CoordinatorEncoder.encode``
uses. If the two ever diverge, the ablated feature would no longer be the same
quantity as the full model's.

torch is imported **inside** each test body -- never at module scope, which
pytest executes at collection time and which would break the in-process
``test_no_torch_imported`` guards that sort after this file. The ``test_torch_``
filename prefix keeps it after them in alphabetical run order.
"""
from __future__ import annotations

import pytest

from trinity.coordinator.encoder_ablations import (
    LAST_TOKEN,
    PENULTIMATE_TOKEN,
    select_token_position,
    token_index_for,
)

SEQ, HIDDEN = 7, 16


def _torch():
    return pytest.importorskip("torch")


def _hidden(seed=0):
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, SEQ, HIDDEN, generator=g)


def test_penultimate_matches_the_shipped_indexing_expression():
    """Pin against ``out.hidden_states[-1][0, -2, :]`` verbatim."""
    hs = _hidden(0)
    got = select_token_position(hs, PENULTIMATE_TOKEN)
    assert bool((got == hs[0, -2, :]).all())


def test_last_token_matches_the_ablated_indexing_expression():
    hs = _hidden(1)
    got = select_token_position(hs, LAST_TOKEN)
    assert bool((got == hs[0, -1, :]).all())


def test_the_two_read_positions_differ_on_real_tensors():
    hs = _hidden(2)
    a = select_token_position(hs, PENULTIMATE_TOKEN)
    b = select_token_position(hs, LAST_TOKEN)
    assert not bool((a == b).all())


def test_selection_preserves_shape_and_dtype():
    hs = _hidden(3)
    got = select_token_position(hs, LAST_TOKEN)
    assert tuple(got.shape) == (HIDDEN,)
    assert got.dtype == hs.dtype


def test_selection_is_a_view_not_a_copy():
    """Indexing must not silently detach the ablation from the forward pass."""
    hs = _hidden(4)
    got = select_token_position(hs, LAST_TOKEN)
    hs[0, -1, 0] = 123.0
    assert float(got[0]) == 123.0


def test_variant_lookup_drives_the_right_tensor_row():
    """End-to-end: variant name -> index -> the row the encoder would read."""
    hs = _hidden(5)
    for variant, want_row in (
        ("full", hs[0, -2, :]),
        ("no_svf", hs[0, -2, :]),
        ("last_token", hs[0, -1, :]),
    ):
        got = select_token_position(hs, token_index_for(variant))
        assert bool((got == want_row).all()), variant


def test_rejects_a_too_short_torch_sequence():
    torch = _torch()
    hs = torch.zeros(1, 1, HIDDEN)
    with pytest.raises(ValueError, match="too short"):
        select_token_position(hs, PENULTIMATE_TOKEN)


def test_rejects_a_non_3d_torch_tensor():
    torch = _torch()
    with pytest.raises(ValueError, match="batch, seq_len, hidden"):
        select_token_position(torch.zeros(SEQ, HIDDEN), LAST_TOKEN)
