"""Offline tests for the opt-in on-disk completion cache. No network, no GPU."""
from __future__ import annotations

import asyncio
import json

import pytest

from trinity.llm.cache import (
    CachedPool,
    ResponseCache,
    cache_from_env,
    cache_key,
    wrap_pool,
)

MSGS = [{"role": "user", "content": "2+2?"}]


class _StubPool:
    """Counts calls and returns a distinct answer each time."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.models = {"m-a": "vendor/m-a"}
        self.decoding = {"temperature": 0.0}

    def model_id(self, name: str) -> str:
        return self.models[name]

    async def chat(self, model, messages, *, temperature=0.7, top_p=0.95,
                   max_tokens=4096, reasoning=None, **kwargs):
        from trinity.llm.openrouter_client import ChatResult

        self.calls.append((model, temperature))
        n = len(self.calls)
        return ChatResult(
            model=model, text=f"answer-{n}", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", raw={"n": n},
        )


def _cached(tmp_path, **kw) -> tuple[CachedPool, _StubPool]:
    pool = _StubPool()
    return CachedPool(pool, ResponseCache(tmp_path), **kw), pool


def _chat(cp, **kw):
    return asyncio.run(cp.chat("m-a", MSGS, temperature=0.0, **kw))


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------
def test_key_is_stable_across_calls_and_dict_order():
    a = cache_key("m", [{"role": "user", "content": "x"}], temperature=0.0)
    b = cache_key("m", [{"content": "x", "role": "user"}], temperature=0.0)
    assert a == b and len(a) == 64


@pytest.mark.parametrize("kw", [
    {"model": "other"},
    {"temperature": 0.3},
    {"top_p": 0.5},
    {"max_tokens": 8},
    {"reasoning": "minimal"},
])
def test_key_changes_when_any_completion_affecting_field_changes(kw):
    base = dict(model="m", temperature=0.0, top_p=0.95, max_tokens=4096, reasoning=None)
    a = cache_key(base.pop("model"), MSGS, **base)
    args = dict(model="m", temperature=0.0, top_p=0.95, max_tokens=4096, reasoning=None)
    args.update(kw)
    b = cache_key(args.pop("model"), MSGS, **args)
    assert a != b


def test_key_changes_with_the_messages():
    a = cache_key("m", [{"role": "user", "content": "x"}], temperature=0.0)
    b = cache_key("m", [{"role": "user", "content": "y"}], temperature=0.0)
    assert a != b


# ---------------------------------------------------------------------------
# hit / miss
# ---------------------------------------------------------------------------
def test_second_identical_deterministic_call_is_served_from_disk(tmp_path):
    cp, pool = _cached(tmp_path)
    first = _chat(cp)
    second = _chat(cp)

    assert len(pool.calls) == 1, "the pool must not be called twice"
    assert second.text == first.text == "answer-1"
    assert second.prompt_tokens == 10 and second.completion_tokens == 5
    assert second.finish_reason == "stop"
    assert cp.stats.hits == 1 and cp.stats.misses == 1 and cp.stats.writes == 1
    assert cp.stats.tokens_saved == 15
    assert cp.stats.hit_rate == 0.5


def test_a_different_request_is_a_miss(tmp_path):
    cp, pool = _cached(tmp_path)
    _chat(cp)
    asyncio.run(cp.chat("m-a", [{"role": "user", "content": "3+3?"}], temperature=0.0))
    assert len(pool.calls) == 2 and cp.stats.hits == 0


def test_cache_survives_a_new_instance_over_the_same_dir(tmp_path):
    cp1, pool1 = _cached(tmp_path)
    _chat(cp1)
    cp2, pool2 = _cached(tmp_path)
    result = _chat(cp2)
    assert pool2.calls == [], "second pool should never be called"
    assert result.text == "answer-1" and cp2.stats.hits == 1


# ---------------------------------------------------------------------------
# The determinism rule: sampled reps must stay independent
# ---------------------------------------------------------------------------
def test_sampled_calls_are_never_cached(tmp_path):
    # fugu.eval.evaluate takes a strict-majority vote over `reps` sampled draws.
    # Serving them from one cache entry would make every vote identical.
    cp, pool = _cached(tmp_path)
    a = asyncio.run(cp.chat("m-a", MSGS, temperature=1.0))
    b = asyncio.run(cp.chat("m-a", MSGS, temperature=1.0))

    assert len(pool.calls) == 2, "sampled draws must stay independent"
    assert a.text != b.text
    assert cp.stats.bypassed == 2
    assert cp.stats.hits == 0 and cp.stats.writes == 0
    assert not list(tmp_path.rglob("*.json")), "nothing may be written for sampled calls"


def test_sampled_calls_can_be_cached_when_explicitly_opted_in(tmp_path):
    cp, pool = _cached(tmp_path, cache_sampled=True)
    a = asyncio.run(cp.chat("m-a", MSGS, temperature=1.0))
    b = asyncio.run(cp.chat("m-a", MSGS, temperature=1.0))
    assert len(pool.calls) == 1 and a.text == b.text


# ---------------------------------------------------------------------------
# Disabled cache and passthrough
# ---------------------------------------------------------------------------
def test_a_none_cache_delegates_every_call(tmp_path):
    pool = _StubPool()
    cp = CachedPool(pool, None)
    a = asyncio.run(cp.chat("m-a", MSGS, temperature=0.0))
    b = asyncio.run(cp.chat("m-a", MSGS, temperature=0.0))
    assert len(pool.calls) == 2 and a.text != b.text
    assert cp.stats.hits == 0 and cp.stats.misses == 0


def test_unknown_attributes_forward_to_the_wrapped_pool(tmp_path):
    cp, pool = _cached(tmp_path)
    assert cp.models == pool.models
    assert cp.model_id("m-a") == "vendor/m-a"
    assert cp.decoding == {"temperature": 0.0}


def test_transport_kwargs_do_not_change_the_key_and_are_forwarded(tmp_path):
    cp, pool = _cached(tmp_path)
    _chat(cp, client=object())
    _chat(cp, client=object())
    assert len(pool.calls) == 1, "an httpx client cannot change the completion"


# ---------------------------------------------------------------------------
# Robustness: a broken cache must never break a run
# ---------------------------------------------------------------------------
def test_a_corrupt_entry_is_treated_as_a_miss(tmp_path):
    cp, pool = _cached(tmp_path)
    _chat(cp)
    entry = next(tmp_path.rglob("*.json"))
    entry.write_text("{not json")

    result = _chat(cp)
    assert result.text == "answer-2", "must fall back to a real call"
    assert cp.stats.errors == 1 and len(pool.calls) == 2


def test_an_entry_from_an_older_schema_is_ignored(tmp_path):
    cp, pool = _cached(tmp_path)
    _chat(cp)
    entry = next(tmp_path.rglob("*.json"))
    record = json.loads(entry.read_text())
    record["v"] = 0  # pretend it was written by an older version
    entry.write_text(json.dumps(record))

    _chat(cp)
    assert cp.stats.errors == 1 and len(pool.calls) == 2


def test_put_leaves_no_temp_files_behind(tmp_path):
    cp, _ = _cached(tmp_path)
    _chat(cp)
    assert not list(tmp_path.rglob("*.tmp"))
    assert len(list(tmp_path.rglob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Env-driven activation
# ---------------------------------------------------------------------------
def test_cache_is_disabled_unless_the_env_var_is_set(monkeypatch, tmp_path):
    monkeypatch.delenv("TRINITY_LLM_CACHE", raising=False)
    assert cache_from_env() is None

    pool = _StubPool()
    assert wrap_pool(pool) is pool, "no env var -> the pool is returned untouched"

    monkeypatch.setenv("TRINITY_LLM_CACHE", str(tmp_path))
    assert isinstance(cache_from_env(), ResponseCache)
    assert isinstance(wrap_pool(pool), CachedPool)


def test_a_blank_env_var_does_not_enable_the_cache(monkeypatch):
    monkeypatch.setenv("TRINITY_LLM_CACHE", "   ")
    assert cache_from_env() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class _ErrorThenGoodPool:
    """Returns a transient error on call 1, then a real answer on call 2."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.models = {"m-a": "vendor/m-a"}
        self.decoding = {"temperature": 0.0}

    def model_id(self, name: str) -> str:
        return self.models[name]

    async def chat(self, model, messages, *, temperature=0.7, top_p=0.95,
                   max_tokens=4096, reasoning=None, **kwargs):
        from trinity.llm.openrouter_client import ChatResult

        self.calls.append((model, temperature))
        if len(self.calls) == 1:  # transient moderation/empty/CDN failure as HTTP 200
            return ChatResult(model=model, text="", prompt_tokens=12,
                              completion_tokens=0, finish_reason="error", raw={})
        return ChatResult(model=model, text="the answer is 4", prompt_tokens=10,
                          completion_tokens=5, finish_reason="stop", raw={})


def test_transient_error_completion_is_not_cached(tmp_path):
    pool = _ErrorThenGoodPool()
    cache = ResponseCache(tmp_path)
    cp = CachedPool(pool, cache)
    key = cache_key("m-a", MSGS, temperature=0.0, top_p=0.95, max_tokens=4096, reasoning=None)

    first = asyncio.run(cp.chat("m-a", MSGS, temperature=0.0))
    assert first.finish_reason == "error" and first.text == ""
    assert len(pool.calls) == 1
    assert cache.get(key) is None, "a transient error must never be cached"
    assert cache.stats.skipped == 1 and cache.stats.writes == 0

    # A retry issues a FRESH call and gets the real answer, which IS cached.
    second = asyncio.run(cp.chat("m-a", MSGS, temperature=0.0))
    assert second.text == "the answer is 4"
    assert len(pool.calls) == 2
    assert cache.get(key) is not None

    # The now-cached real answer is served from disk (no new pool call).
    third = asyncio.run(cp.chat("m-a", MSGS, temperature=0.0))
    assert third.text == "the answer is 4"
    assert len(pool.calls) == 2


def test_blank_text_completion_is_skipped(tmp_path):
    # An empty completion with finish_reason="stop" is still not a real answer.
    class _BlankPool(_StubPool):
        async def chat(self, model, messages, **kw):
            from trinity.llm.openrouter_client import ChatResult

            self.calls.append((model, kw.get("temperature")))
            return ChatResult(model=model, text="  ", prompt_tokens=1,
                              completion_tokens=0, finish_reason="stop", raw={})

    pool = _BlankPool()
    cache = ResponseCache(tmp_path)
    cp = CachedPool(pool, cache)
    asyncio.run(cp.chat("m-a", MSGS, temperature=0.0))
    key = cache_key("m-a", MSGS, temperature=0.0, top_p=0.95, max_tokens=4096, reasoning=None)
    assert cache.get(key) is None
    assert cache.stats.skipped == 1


def test_stats_to_dict_reports_skipped():
    from trinity.llm.cache import CacheStats

    assert CacheStats(skipped=3).to_dict()["skipped"] == 3
