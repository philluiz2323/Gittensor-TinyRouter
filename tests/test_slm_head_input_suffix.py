"""The coordinator reads a fixed `<Head Input>` position, not the transcript's tail.

SPEC §3.2 pins the canonical extraction:

    tokenize `transcript + "\\n<Head Input>"`, do one forward pass with a single
    appended EOS, read index `-2`

and records the cost of getting it wrong: reading a content token instead of the
intended one collapses LiveCodeBench 61.46 -> 50.85.

`CoordinatorEncoder.encode` appended the EOS and read `-2`, but never appended the
suffix. Index `-2` therefore landed on the transcript's **last content token**, so
the head's sole input varied with whatever the transcript happened to end on.

Appending EOS does not rescue that. Attention is causal, so the state at `-2`
cannot attend to the EOS at `-1`; `h[-2]` with an EOS appended is bit-identical to
the last content token's state with no EOS at all.
`test_appending_eos_is_a_noop_under_causal_attention` proves that directly.

Offline: `transformers` is never imported -- `encode` touches only injectable
attributes, so the encoder is built with `object.__new__` and fakes. `torch` is
imported lazily *inside* the tests, never at module scope, so collecting this file
does not violate `test_shaped_fitness.py::test_no_torch_imported`.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.coordinator.slm import HEAD_INPUT_SUFFIX, CoordinatorEncoder

_EOS_ID = 999
_HIDDEN = 8


def _torch():
    return pytest.importorskip("torch", reason="torch is required to exercise encode()")


class _FakeTokenizer:
    """Char-code tokenizer. Records the exact text it was handed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        import torch

        self.seen.append(text)
        ids = [ord(c) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _EchoModel:
    """Hidden states that encode each position's token id.

    Row ``i`` of the final layer is ``[token_id_i, i, 0, ...]``, so a test can read
    back exactly which token the encoder sampled.
    """

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=None,
                 use_cache=None):
        import torch

        seq = input_ids.shape[1]
        h = torch.zeros(1, seq, _HIDDEN, dtype=torch.float32)
        for i in range(seq):
            h[0, i, 0] = float(input_ids[0, i])
            h[0, i, 1] = float(i)

        class _Out:
            hidden_states = (h,)

        return _Out()


def _encoder(*, l2: bool = False) -> tuple[CoordinatorEncoder, _FakeTokenizer]:
    """Build an encoder without running `__init__` (which would import transformers)."""
    torch = _torch()
    enc = object.__new__(CoordinatorEncoder)
    tok = _FakeTokenizer()
    enc._torch = torch
    enc.tokenizer = tok
    enc.model = _EchoModel()
    enc.device = "cpu"
    enc._eos_id = _EOS_ID
    enc.l2_normalize = l2
    return enc, tok


def _token_read(enc: CoordinatorEncoder, transcript: str) -> int:
    """The token id at the position `encode` actually sampled."""
    return int(round(float(enc.encode(transcript)[0])))


# --------------------------------------------------------------------------- #
# The suffix is appended
# --------------------------------------------------------------------------- #
def test_the_suffix_is_exactly_the_spec_string():
    assert HEAD_INPUT_SUFFIX == "\n<Head Input>"


def test_the_head_input_suffix_is_appended_to_the_transcript():
    enc, tok = _encoder()
    enc.encode("QUERY: 2+2")
    assert tok.seen == ["QUERY: 2+2" + HEAD_INPUT_SUFFIX]


# --------------------------------------------------------------------------- #
# The regression: index -2 must be the suffix's last token, not content
# --------------------------------------------------------------------------- #
def test_read_position_is_the_final_suffix_token():
    enc, _ = _encoder()
    assert _token_read(enc, "QUERY: 2+2") == ord(HEAD_INPUT_SUFFIX[-1])  # '>'


def test_read_position_is_not_the_transcripts_last_token():
    """The regression: previously this read '2', the transcript's last char."""
    enc, _ = _encoder()
    assert _token_read(enc, "QUERY: 2+2") != ord("2")


@pytest.mark.parametrize("transcript", ["a", "ends with x", "ends with y", "!@#"])
def test_read_token_is_invariant_to_the_transcripts_tail(transcript):
    """A fixed decision position: the sampled TOKEN never depends on the tail."""
    enc, _ = _encoder()
    assert _token_read(enc, transcript) == ord(HEAD_INPUT_SUFFIX[-1])


def test_read_index_is_penultimate_not_eos():
    """-2, never -1: the EOS itself must never be the head's input."""
    enc, _ = _encoder()
    assert _token_read(enc, "hello") != _EOS_ID


def test_eos_is_still_appended_as_the_final_position():
    enc, _ = _encoder()
    vec = enc.encode("hi")
    seq_index = int(round(float(vec[1])))
    expected_len = len("hi" + HEAD_INPUT_SUFFIX) + 1  # + EOS
    assert seq_index == expected_len - 2


# --------------------------------------------------------------------------- #
# Why appending EOS alone could never have worked
# --------------------------------------------------------------------------- #
def test_appending_eos_is_a_noop_under_causal_attention():
    """`h[-2]` with EOS appended == `h[-1]` with no EOS. Hence the suffix.

    A minimal causal self-attention block: position i attends only to j <= i, so a
    token appended after i cannot change i's output. This is why the old
    "append EOS, read -2" could not create a decision position -- it only renamed
    the last content token.
    """
    torch = _torch()
    torch.manual_seed(0)
    d = 6
    wq, wk, wv = (torch.randn(d, d) for _ in range(3))

    def causal_block(x):
        q, k, v = x @ wq, x @ wk, x @ wv
        scores = q @ k.T / (d ** 0.5)
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    content = torch.randn(5, d)
    eos = torch.randn(1, d)

    without_eos = causal_block(content)[-1]
    with_eos = causal_block(torch.cat([content, eos], dim=0))[-2]

    assert torch.allclose(without_eos, with_eos, atol=1e-6)


# --------------------------------------------------------------------------- #
# Surrounding behaviour is unchanged
# --------------------------------------------------------------------------- #
def test_output_is_float32_and_one_dimensional():
    enc, _ = _encoder()
    vec = enc.encode("hi")
    assert vec.dtype == np.float32
    assert vec.ndim == 1


def test_encode_is_deterministic():
    enc, _ = _encoder()
    assert np.array_equal(enc.encode("same text"), enc.encode("same text"))


def test_l2_normalisation_still_applies():
    enc, _ = _encoder(l2=True)
    vec = enc.encode("hello world")
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)


def test_empty_transcript_still_has_a_penultimate_position():
    """The suffix guarantees >= 2 tokens, so an empty transcript no longer raises."""
    enc, _ = _encoder()
    assert _token_read(enc, "") == ord(HEAD_INPUT_SUFFIX[-1])
