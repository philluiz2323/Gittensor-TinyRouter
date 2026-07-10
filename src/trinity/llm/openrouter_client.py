"""OpenRouter client for the coordinated LLM pool.

OpenAI-compatible chat-completions client with retry/backoff and bounded
concurrency. The API key is read from the ``OPENROUTER_API_KEY`` environment
variable and is NEVER read from a file inside the repo (see AGENTS.md §4).

Run a self-test (pings all configured pool models):

    source ~/.config/trinity/secrets.env
    python -m trinity.llm.openrouter_client --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ModuleNotFoundError:  # pragma: no cover - smoke env can omit runtime deps
    def retry(*args, **kwargs):
        del args, kwargs

        def _wrap(fn):
            return fn

        return _wrap

    def retry_if_exception_type(*args, **kwargs):
        del args, kwargs
        return None

    def stop_after_attempt(*args, **kwargs):
        del args, kwargs
        return None

    def wait_exponential(*args, **kwargs):
        del args, kwargs
        return None

if TYPE_CHECKING:
    import httpx

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "models.yaml"


@dataclass
class ChatResult:
    """One completion plus the accounting we need for fitness/cost terms."""

    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


class _Retryable(Exception):
    """Wraps transient HTTP failures so tenacity retries them."""


def _ledger_append(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Append one token-usage record to the cost ledger, if TRINITY_COST_LEDGER is set.

    Each entry includes a hash-chain link ``h`` = sha256(prev_hash + entry_data).
    This makes the ledger append-only and tamper-evident: any modification or
    deletion of a line breaks the chain, and ``scripts/cost_report.py`` verifies
    it before reporting totals.

    Used by scripts/cost_report.py to compute exact spend. Append-only JSONL,
    one short line per call. Disk appends take an exclusive sidecar lock around
    read-tip + write so concurrent training processes share one chain tip.
    Best-effort: never let cost bookkeeping break an inference call.
    """
    path = os.environ.get("TRINITY_COST_LEDGER")
    if not path:
        return
    try:
        from trinity.llm.cost_ledger import append_ledger_entry

        append_ledger_entry(path, model, prompt_tokens, completion_tokens)
    except Exception:
        pass


def _message_text(choice_message: dict) -> str:
    """Normalize OpenRouter/OpenAI message content to plain text.

    ``message.content`` is nullable in the chat-completions schema: providers send
    ``null`` for an empty completion, or when the assistant message carries only
    reasoning / tool-call metadata. "No text" normalizes to ``""`` — the same as a
    missing key or an empty content list — never to the string ``"None"``.
    """

    content = choice_message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


class OpenRouterPool:
    """Async-first client over the OpenRouter chat-completions endpoint."""

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG):
        cfg = yaml.safe_load(Path(config_path).read_text())
        orc = cfg["openrouter"]
        self.base_url: str = orc["base_url"].rstrip("/")
        self.timeout_s: float = float(orc.get("timeout_s", 120))
        self.max_retries: int = int(orc.get("max_retries", 4))
        self._sem = asyncio.Semaphore(int(orc.get("max_concurrency", 8)))
        self.provider: dict = dict(orc.get("provider") or {})

        api_key = os.environ.get(orc.get("api_key_env", "OPENROUTER_API_KEY"), "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Run: source ~/.config/trinity/secrets.env"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if orc.get("http_referer"):
            headers["HTTP-Referer"] = str(orc["http_referer"])
        if orc.get("app_title"):
            headers["X-OpenRouter-Title"] = str(orc["app_title"])
        self._headers = headers

        self.models: dict[str, str] = {m["name"]: m["id"] for m in cfg["pool"]}
        self.decoding: dict = cfg.get("decoding", {})

    def model_id(self, name: str) -> str:
        if name in self.models:
            return self.models[name]
        if name in self.models.values():
            return name
        raise KeyError(f"Unknown model '{name}'. Known: {list(self.models)}")

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        reasoning: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> ChatResult:
        import httpx

        payload = {
            "model": self.model_id(model),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if reasoning is not None:
            payload["reasoning_effort"] = reasoning
        if self.provider:
            payload["provider"] = self.provider

        @retry(
            retry=retry_if_exception_type(_Retryable),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            reraise=True,
        )
        async def _do(cli: httpx.AsyncClient) -> ChatResult:
            async with self._sem:
                try:
                    resp = await cli.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                        timeout=self.timeout_s,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    raise _Retryable(f"network: {type(exc).__name__}: {exc}") from exc
            if resp.status_code in (429, 500, 502, 503, 504):
                raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            # `data.get("usage", {})` only defaults an ABSENT key. OpenAI-compatible
            # providers also send `"usage": null` (some providers, and 200s with an
            # empty completion), which would make `usage.get(...)` raise on `None` --
            # crashing an otherwise-successful call. `or {}` covers absent and null,
            # matching how `_message_text` already guards `content: null`.
            usage = data.get("usage") or {}
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            _ledger_append(payload["model"], pt, ct)
            return ChatResult(
                model=payload["model"],
                text=_message_text(choice["message"]),
                prompt_tokens=pt,
                completion_tokens=ct,
                finish_reason=choice.get("finish_reason"),
                raw=data,
            )

        if client is not None:
            return await _do(client)
        async with httpx.AsyncClient() as cli:
            return await _do(cli)


async def _selftest() -> int:
    import httpx

    pool = OpenRouterPool()
    print(f"Pool: {list(pool.models)}")
    async with httpx.AsyncClient() as cli:
        results = await asyncio.gather(
            *[
                pool.chat(
                    name,
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=8,
                    temperature=0.0,
                    client=cli,
                )
                for name in pool.models
            ],
            return_exceptions=True,
        )
    ok = True
    for name, res in zip(pool.models, results):
        if isinstance(res, Exception):
            ok = False
            print(f"  [FAIL] {name}: {res!r}")
        else:
            print(
                f"  [ OK ] {name:18s} -> {res.text.strip()[:40]!r} "
                f"({res.completion_tokens} toks)"
            )
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenRouter pool client")
    ap.add_argument("--selftest", action="store_true", help="ping all pool models")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(asyncio.run(_selftest()))
    ap.print_help()


if __name__ == "__main__":
    main()
