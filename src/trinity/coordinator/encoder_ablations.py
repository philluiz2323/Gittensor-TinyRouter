"""Encoder-side ablations for SPEC R9 (``no_svf`` / ``last_token``).

``docs/SPEC.md`` §1.3 invariant **R9** claims that removing each design component
hurts accuracy. Its *verifier* is merged -- ``trinity.analysis.ablations.analyze``
consumes ``{full, no_svf, no_thinker, no_trirole, last_token}`` -- but nothing in
``src/`` could ever *produce* those numbers; the variant names appear only in
docstrings and the example JSON in ``scripts/ablations_report.py``.

This module produces the two ablations that concern the **feature pipeline**
rather than the routing decision:

``no_svf``
    Remove SVF adaptation. The SVF block of θ is set to the identity (all
    scales 1.0), so the encoder runs with its stock singular values. This is a
    pure θ transformation -- no code path changes and no module is monkeyed
    with; ``SVFAdapter.set_scales(ones)`` is already exactly "no adaptation",
    which is why :meth:`~trinity.coordinator.svf.SVFAdapter.identity_scales`
    exists.

``last_token``
    Read the **last** output token (the appended EOS) instead of SPEC §3.2's
    penultimate one. ``slm.py`` documents why the penultimate position is the
    canonical choice; this ablation is what measures that claim.

The role ablations (``no_thinker`` / ``no_trirole``) are a separate concern and
live in :mod:`trinity.coordinator.ablations`.

Scope
-----
This ships the *producers*. Turning them into R9's accuracy numbers still means
running the benchmark, which needs the GPU box -- as does any call into
``CoordinatorEncoder`` itself. What is verifiable offline, and is tested, is the
θ transformation, the position registry, and the wiring that carries a token
index from config into the encoder.

Import cost
-----------
No torch at module scope: ``no_svf`` is pure numpy over the canonical
``params.pack``/``unpack``, and the token-position helpers are plain integer
bookkeeping. Only :func:`make_ablated_encoder` reaches for the encoder, and it
imports it lazily inside the call.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import params as _params

__all__ = [
    "ENCODER_ABLATIONS",
    "PENULTIMATE_TOKEN",
    "TOKEN_POSITIONS",
    "ablate_svf",
    "is_svf_ablated",
    "make_ablated_encoder",
    "select_token_position",
    "token_index_for",
]

#: SPEC §3.2's canonical read position: the final content token, NOT the EOS
#: appended after it. ``slm.CoordinatorEncoder`` defaults to this.
PENULTIMATE_TOKEN: int = -2

#: The last output position -- the appended EOS. This is the ``last_token``
#: ablation's read position.
LAST_TOKEN: int = -1

#: Variant name -> sequence index read from the final hidden layer. ``"full"``
#: is included so a caller can drive the un-ablated run through the same code.
TOKEN_POSITIONS: Mapping[str, int] = {
    "full": PENULTIMATE_TOKEN,
    "last_token": LAST_TOKEN,
}

#: The encoder-side R9 variant names, matching what the merged verifier expects.
ENCODER_ABLATIONS: tuple[str, ...] = ("no_svf", "last_token")


def token_index_for(variant: str) -> int:
    """Sequence index to read for ``variant``.

    ``no_svf`` does not move the read position, so it maps to the canonical
    penultimate index like ``full`` does -- the two differ in θ, not in where
    the hidden state is sampled.

    Raises
    ------
    KeyError
        If ``variant`` is neither a known token position nor ``"no_svf"``.
    """
    if variant == "no_svf":
        return PENULTIMATE_TOKEN
    try:
        return TOKEN_POSITIONS[variant]
    except KeyError:
        raise KeyError(
            f"unknown variant {variant!r}; known: "
            f"{sorted(set(TOKEN_POSITIONS) | {'no_svf'})}"
        ) from None


def select_token_position(hidden_last_layer: Any, token_index: int) -> Any:
    """Read one position out of a final-layer hidden-state tensor.

    Mirrors ``slm.CoordinatorEncoder.encode``'s extraction so the ablation and
    the shipped path index identically.

    Parameters
    ----------
    hidden_last_layer:
        Tensor shaped ``(batch, seq_len, hidden_size)``. Only batch 0 is read --
        encoding is a per-turn operation.
    token_index:
        Sequence index, normally :data:`PENULTIMATE_TOKEN` or
        :data:`LAST_TOKEN`.

    Raises
    ------
    ValueError
        If the tensor is not 3-D, or is too short for ``token_index``.
    """
    shape = tuple(hidden_last_layer.shape)
    if len(shape) != 3:
        raise ValueError(
            f"expected (batch, seq_len, hidden) hidden states, got shape {shape}"
        )
    seq_len = shape[1]
    if seq_len < abs(token_index):
        raise ValueError(
            f"sequence length {seq_len} is too short to read index {token_index}"
        )
    return hidden_last_layer[0, token_index, :]


def ablate_svf(theta: np.ndarray, spec: _params.ParamSpec) -> np.ndarray:
    """Return a copy of ``theta`` with the SVF block set to the identity.

    Removing SVF adaptation *is* running the encoder at its stock singular
    values, i.e. all scales 1.0 -- the same vector
    :meth:`~trinity.coordinator.svf.SVFAdapter.identity_scales` hands out. The
    head block is passed through untouched, so an ablated run differs from the
    full model in exactly one component.

    θ keeps its full width (``spec.n_total``), so the ablated policy stays
    directly comparable with the full one and needs no separate ``ParamSpec``.

    Parameters
    ----------
    theta:
        Flat parameter vector of length ``spec.n_total``.
    spec:
        The spec ``theta`` was packed against.

    Returns
    -------
    np.ndarray
        A new ``float64`` vector; ``theta`` is not modified in place.

    Raises
    ------
    ValueError
        If ``theta`` does not match ``spec`` (raised by ``params.unpack``).
    """
    head_W, _svf_scales = _params.unpack(theta, spec)
    identity = np.ones(spec.n_svf, dtype=np.float64)
    return _params.pack(head_W, identity)


def is_svf_ablated(theta: np.ndarray, spec: _params.ParamSpec) -> bool:
    """Whether ``theta``'s SVF block is exactly the identity.

    Useful as a run-manifest assertion: it distinguishes a genuine ``no_svf``
    run from a full run whose scales merely happen to be near 1.
    """
    _head_W, svf_scales = _params.unpack(theta, spec)
    return bool(np.array_equal(svf_scales, np.ones(spec.n_svf, dtype=np.float64)))


def make_ablated_encoder(variant: str, **encoder_kwargs: Any) -> Any:
    """Build a ``CoordinatorEncoder`` configured for ``variant``.

    ``last_token`` sets the encoder's read position to the EOS; ``full`` and
    ``no_svf`` leave it at SPEC §3.2's penultimate token (``no_svf`` is applied
    to θ, not to the encoder).

    An explicit ``token_index=`` in ``encoder_kwargs`` is rejected rather than
    silently overridden -- passing one alongside a variant means two different
    intentions for the same setting.

    Notes
    -----
    Loading the encoder needs the GPU box; only the argument wiring is
    exercised offline.
    """
    if "token_index" in encoder_kwargs:
        raise TypeError(
            "token_index is determined by the variant; drop it or call "
            "CoordinatorEncoder directly"
        )
    index = token_index_for(variant)

    from .slm import CoordinatorEncoder  # lazy: pulls torch/transformers

    return CoordinatorEncoder(token_index=index, **encoder_kwargs)
