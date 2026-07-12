"""Opt-in on-disk cache for deterministic pool completions.

Why this exists
---------------
The evaluation paths re-issue the *same* deterministic request many times.
``trinity.eval._score_single_model`` calls ``pool.chat(..., temperature=0.0)``
once per (task, model), and ``--single-reps`` repeats the whole sweep; the
oracle-matrix collector and the hidden-benchmark builder do the same. Every
repeat is a fresh paid API call for a completion the pool already produced.

This module wraps any pool with a content-addressed cache, so a repeated
deterministic request is served from disk for $0. It is **opt-in and
zero-touch**: nothing imports it unless you ask for it, no existing module is
modified, and with the cache disabled every call goes straight through.

The determinism rule (why sampled calls are never cached)
---------------------------------------------------------
Only ``temperature == 0`` requests are cacheable. A sampled request
(``temperature > 0``) is an independent draw, and the code relies on that:
``trinity.fugu.eval.evaluate`` runs ``reps`` sampled rollouts per task and
reduces them with a strict-majority vote. Serving those reps from one cache
entry would make every vote identical and silently turn a majority into a
single sample. So sampled calls always bypass the cache, and
:class:`CachedPool` will not even write them.

Cost accounting stays honest: a cache hit never reaches
``OpenRouterPool.chat``, so no cost-ledger row is appended for a call that was
never billed. Only real API calls are metered.

Pure stdlib. No network, no torch, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CacheStats",
    "ResponseCache",
    "CachedPool",
    "cache_key",
    "cache_from_env",
    "wrap_pool",
]

#: Set this to a directory path to enable the cache (see :func:`cache_from_env`).
CACHE_ENV_VAR = "TRINITY_LLM_CACHE"

#: Bump when the on-disk record shape changes, so stale entries are ignored.
_SCHEMA_VERSION = 1


def _canonical(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    reasoning: str | None = None,
) -> str:
    """Content-address a completion request.

    Every field that can change the completion is hashed; transport-only
    arguments (an ``httpx`` client, timeouts) are not, since they cannot affect
    the text the model returns.

    Returns:
        A 64-char sha256 hex digest.
    """
    payload = {
        "v": _SCHEMA_VERSION,
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "reasoning": reasoning,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass
class CacheStats:
    """Running tally for one :class:`ResponseCache` instance."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    bypassed: int = 0          # sampled requests, never cacheable
    errors: int = 0            # unreadable/corrupt entries, treated as misses
    skipped: int = 0           # transient empty/error completions, never persisted
    prompt_tokens_saved: int = 0
    completion_tokens_saved: int = 0

    @property
    def lookups(self) -> int:
        """Cacheable lookups performed (hits + misses); excludes bypasses."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of cacheable lookups served from disk (``0.0`` when none)."""
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def tokens_saved(self) -> int:
        """Total tokens that did not have to be re-billed."""
        return self.prompt_tokens_saved + self.completion_tokens_saved

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view for run summaries."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "bypassed": self.bypassed,
            "errors": self.errors,
            "skipped": self.skipped,
            "hit_rate": round(self.hit_rate, 4),
            "prompt_tokens_saved": self.prompt_tokens_saved,
            "completion_tokens_saved": self.completion_tokens_saved,
            "tokens_saved": self.tokens_saved,
        }


@dataclass
class ResponseCache:
    """Content-addressed completion store under ``root``.

    One JSON file per key, sharded by the first two hex characters so a large
    run does not put a million entries in one directory. Writes are atomic
    (write to a temp file in the same directory, then ``os.replace``), so a
    crashed or concurrent writer can never leave a half-written record that a
    later reader would parse as a real completion.

    The cache is *best effort*: any I/O or decode failure is counted and treated
    as a miss. A broken cache must never break a run.
    """

    root: Path
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def path_for(self, key: str) -> Path:
        """On-disk location of ``key`` (sharded by its first two hex chars)."""
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored record for ``key``, or ``None`` on miss.

        A record written by an older schema, or one that fails to parse, is
        reported as a miss (and counted in ``stats.errors`` when it existed but
        could not be used).
        """
        path = self.path_for(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            self.stats.errors += 1
            self.stats.misses += 1
            return None
        if not isinstance(record, dict) or record.get("v") != _SCHEMA_VERSION:
            self.stats.errors += 1
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        self.stats.prompt_tokens_saved += int(record.get("prompt_tokens", 0) or 0)
        self.stats.completion_tokens_saved += int(record.get("completion_tokens", 0) or 0)
        return record

    def put(self, key: str, record: dict[str, Any]) -> bool:
        """Atomically store ``record`` under ``key``. Returns ``True`` on success."""
        path = self.path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stamped = dict(record)
            stamped["v"] = _SCHEMA_VERSION
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(_canonical(stamped))
                os.replace(tmp, path)
            except BaseException:
                # Never leave the temp file behind on a failed write.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except (OSError, ValueError):
            self.stats.errors += 1
            return False
        self.stats.writes += 1
        return True


def cache_from_env(env_var: str = CACHE_ENV_VAR) -> ResponseCache | None:
    """Build a cache from ``$TRINITY_LLM_CACHE``, or ``None`` when unset/empty.

    Keeping activation in the environment means no call site has to change: a
    run opts in with ``TRINITY_LLM_CACHE=.cache/llm python -m trinity.eval ...``.
    """
    root = os.environ.get(env_var, "").strip()
    return ResponseCache(Path(root)) if root else None


def _is_cacheable(temperature: float) -> bool:
    """Only greedy (temperature 0) requests are reproducible enough to cache."""
    return float(temperature) == 0.0


def _is_cacheable_result(result: Any) -> bool:
    """True iff a completion is a real answer worth persisting.

    A transient empty/error response — ``finish_reason == "error"`` or blank
    ``text``, which is what :class:`~trinity.llm.openrouter_client.OpenRouterPool`
    returns for a moderation block, an upstream-error-surfaced-as-200, or a
    non-JSON CDN body — must never be cached. Persisting it would re-serve the
    empty text for every future identical request, turning a one-off transient
    failure into a permanent 0 score for that item (and reporting bogus
    ``tokens_saved``). Skipping the write only ever costs a recomputation; it can
    never return a wrong cached answer.
    """
    if getattr(result, "finish_reason", None) == "error":
        return False
    return bool(str(getattr(result, "text", "") or "").strip())


class CachedPool:
    """Wrap a pool so deterministic completions are served from disk.

    Every attribute other than :meth:`chat` (``models``, ``model_id``,
    ``decoding``, ...) is forwarded to the wrapped pool, so a ``CachedPool`` is a
    drop-in substitute anywhere a pool is accepted.

    Args:
        pool: The pool to wrap; must expose an async ``chat``.
        cache: The store to use. ``None`` disables caching entirely and every
            call is delegated untouched.
        cache_sampled: Cache ``temperature > 0`` requests too. **Off by default,
            and unsafe for repeated sampling**: `fugu.eval.evaluate` treats reps
            as independent draws and takes a strict-majority vote over them.
    """

    def __init__(self, pool: Any, cache: ResponseCache | None, *, cache_sampled: bool = False):
        self._pool = pool
        self._cache = cache
        self._cache_sampled = bool(cache_sampled)

    def __getattr__(self, name: str) -> Any:
        """Forward everything we do not override to the wrapped pool."""
        return getattr(self._pool, name)

    @property
    def cache(self) -> ResponseCache | None:
        """The backing store, or ``None`` when caching is disabled."""
        return self._cache

    @property
    def stats(self) -> CacheStats:
        """Cache counters (all zero when caching is disabled)."""
        return self._cache.stats if self._cache else CacheStats()

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        reasoning: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a completion, from disk when this exact request was seen before.

        On a hit the wrapped pool is never called, so no API spend and no
        cost-ledger row is produced. On a miss the real call is made and its
        result is stored before being returned.
        """
        cacheable = self._cache is not None and (
            self._cache_sampled or _is_cacheable(temperature)
        )
        if not cacheable:
            if self._cache is not None:
                self._cache.stats.bypassed += 1
            return await self._pool.chat(
                model, messages, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, reasoning=reasoning, **kwargs,
            )

        assert self._cache is not None  # narrowed by `cacheable`
        key = cache_key(
            model, messages, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, reasoning=reasoning,
        )
        record = self._cache.get(key)
        if record is not None:
            return self._result_from(record)

        result = await self._pool.chat(
            model, messages, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, reasoning=reasoning, **kwargs,
        )
        # Never persist a transient empty/error completion: it would be re-served
        # for every future identical request, freezing a one-off failure into a
        # permanent wrong (0) score. Skipping only forces a fresh call next time.
        if _is_cacheable_result(result):
            self._cache.put(key, self._record_from(result))
        else:
            self._cache.stats.skipped += 1
        return result

    @staticmethod
    def _record_from(result: Any) -> dict[str, Any]:
        """Project a ChatResult onto the fields worth persisting.

        ``raw`` (the whole provider response) is deliberately dropped: it is
        large, provider-specific, and nothing downstream reads it from a cached
        completion.
        """
        return {
            "model": getattr(result, "model", ""),
            "text": getattr(result, "text", ""),
            "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(result, "completion_tokens", 0) or 0),
            "finish_reason": getattr(result, "finish_reason", None),
        }

    @staticmethod
    def _result_from(record: dict[str, Any]) -> Any:
        """Rebuild a ``ChatResult`` from a stored record.

        Imported lazily so this module stays importable (and unit-testable)
        without the client's optional dependencies.
        """
        from trinity.llm.openrouter_client import ChatResult

        return ChatResult(
            model=str(record.get("model", "")),
            text=str(record.get("text", "")),
            prompt_tokens=int(record.get("prompt_tokens", 0) or 0),
            completion_tokens=int(record.get("completion_tokens", 0) or 0),
            finish_reason=record.get("finish_reason"),
            raw={"cached": True},
        )


def wrap_pool(pool: Any, *, cache_sampled: bool = False) -> Any:
    """Wrap ``pool`` iff ``$TRINITY_LLM_CACHE`` is set, else return it unchanged.

    This is the one-liner a script adds to opt in without any other change::

        pool = wrap_pool(OpenRouterPool())
    """
    cache = cache_from_env()
    if cache is None:
        return pool
    return CachedPool(pool, cache, cache_sampled=cache_sampled)
