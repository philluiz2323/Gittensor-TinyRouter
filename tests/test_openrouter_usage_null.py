"""Token accounting must survive a null `usage` block from the provider.

OpenAI-compatible backends send `"usage": null` for some providers and for
certain 200-OK responses with an empty completion. `data.get("usage", {})` only
substitutes the default `{}` when the key is *absent*; a present-but-null value
leaves `usage = None`, and `None.get("prompt_tokens", 0)` raises
``AttributeError`` -- crashing an otherwise-successful inference call.

That is the same present-but-null trap merged PR #72 fixed for `content` in this
same file (`_message_text` guards `if content is None`). The token-accounting path
had the identical bug and was missed. It is worse than the `content` case: an
``AttributeError`` is not in the retryable set, so a successful 200 response
propagates the exception -- and with `eval`/`fitness` now gathering
`return_exceptions=True`, a correct trajectory is silently scored 0.0.

Offline: no network, no config, no API key. The pool is built via
``object.__new__`` with only the attributes ``chat``/``_do`` touch, and a fake
client returns the response.
"""
from __future__ import annotations

import asyncio

import pytest

from trinity.llm.openrouter_client import ChatResult, OpenRouterPool


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Records the request and returns a canned response; no network."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    async def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self._payload)


def _pool() -> OpenRouterPool:
    """A pool with only what `chat`/`_do` read -- no config file, no API key."""
    pool = object.__new__(OpenRouterPool)
    pool.base_url = "https://example.test"
    pool._headers = {}
    pool.timeout_s = 30.0
    pool.max_retries = 1
    pool.provider = {}
    pool._sem = asyncio.Semaphore(1)
    pool.models = {"m": "provider/m"}
    return pool


def _chat(payload: dict) -> ChatResult:
    pool = _pool()
    client = _FakeClient(payload)
    return asyncio.run(
        pool.chat("m", [{"role": "user", "content": "hi"}], max_tokens=8, client=client)
    )


def _response(usage) -> dict:
    return {
        "choices": [{"message": {"content": "the answer"}, "finish_reason": "stop"}],
        "usage": usage,
    }


# --------------------------------------------------------------------------- #
# The bug: usage: null must not crash a successful call
# --------------------------------------------------------------------------- #
def test_null_usage_does_not_raise():
    """The regression: `usage: null` used to raise AttributeError."""
    res = _chat(_response(usage=None))
    assert res.text == "the answer"


def test_null_usage_counts_zero_tokens():
    res = _chat(_response(usage=None))
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0


def test_null_usage_preserves_the_completion():
    """A correct answer must not be lost to a null usage block."""
    res = _chat(_response(usage=None))
    assert res.text == "the answer"
    assert res.finish_reason == "stop"


# --------------------------------------------------------------------------- #
# Existing behaviour must not regress
# --------------------------------------------------------------------------- #
def test_absent_usage_key_counts_zero():
    payload = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    res = _chat(payload)
    assert (res.prompt_tokens, res.completion_tokens) == (0, 0)


def test_populated_usage_is_recorded():
    res = _chat(_response(usage={"prompt_tokens": 11, "completion_tokens": 7}))
    assert res.prompt_tokens == 11
    assert res.completion_tokens == 7


def test_empty_usage_dict_counts_zero():
    res = _chat(_response(usage={}))
    assert (res.prompt_tokens, res.completion_tokens) == (0, 0)


def test_partial_usage_defaults_the_missing_field():
    res = _chat(_response(usage={"prompt_tokens": 5}))
    assert res.prompt_tokens == 5
    assert res.completion_tokens == 0
