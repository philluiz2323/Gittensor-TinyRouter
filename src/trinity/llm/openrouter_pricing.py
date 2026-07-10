"""OpenRouter per-model token pricing for ledger cost totals.

The default model pool's OpenRouter rates ($/1M prompt and completion tokens)
must be applied consistently anywhere the repo turns a verified cost ledger
into dollars — ``scripts/cost_report.py --ledger`` and the training receipt
written by ``scripts/pack_submission.py``.

A single blended rate multiplied by ``prompt + completion`` token counts
mis-prices every model in the pool because input and output are billed at
different rates. This module is the single source of truth for those rates and
for summing :class:`~trinity.llm.cost_ledger.LedgerEntry` rows.
"""
from __future__ import annotations

from pathlib import Path

from trinity.llm.cost_ledger import LedgerEntry, read_ledger_entries, verify_ledger_chain

__all__ = [
    "OPENROUTER_POOL_PRICES",
    "default_blended_rates",
    "normalize_model_slug",
    "resolve_rates",
    "token_cost",
    "sum_entry_costs",
    "sum_ledger_cost",
    "verified_ledger_total_usd",
]

# OpenRouter prices, $/1M tokens (prompt, completion). Keep in sync with
# scripts/oracle_ceiling.py::_DEFAULT_PRICES and configs/models.yaml pool.
OPENROUTER_POOL_PRICES: dict[str, tuple[float, float]] = {
    "qwen3.5-35b-a3b": (0.14, 1.00),
    "minimax-m3": (0.30, 1.20),
    "deepseek-v4-flash": (0.09, 0.18),
}


def normalize_model_slug(model: str) -> str:
    """Strip an OpenRouter provider prefix from a model slug."""
    return model.rsplit("/", 1)[-1]


def default_blended_rates() -> tuple[float, float]:
    """Return the arithmetic mean of pool in/out rates (for unknown models)."""
    if not OPENROUTER_POOL_PRICES:
        return (0.0, 0.0)
    ins = [p[0] for p in OPENROUTER_POOL_PRICES.values()]
    outs = [p[1] for p in OPENROUTER_POOL_PRICES.values()]
    return (sum(ins) / len(ins), sum(outs) / len(outs))


def resolve_rates(model: str) -> tuple[float, float]:
    """Return ``(price_in, price_out)`` in $/1M tokens for ``model``.

    Known pool slugs use their table entry. Unknown slugs fall back to the
    blended default so a ledger with a new model name still produces a finite
    total instead of silently dropping rows.
    """
    slug = normalize_model_slug(model)
    return OPENROUTER_POOL_PRICES.get(slug, default_blended_rates())


def token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Price one API call from per-million in/out rates."""
    in_rate, out_rate = resolve_rates(model)
    pt = int(prompt_tokens)
    ct = int(completion_tokens)
    return pt / 1e6 * in_rate + ct / 1e6 * out_rate


def sum_entry_costs(entries: list[LedgerEntry]) -> float:
    """Sum dollar cost for parsed ledger rows (no hash verification)."""
    return sum(
        token_cost(entry.model, entry.prompt_tokens, entry.completion_tokens)
        for entry in entries
    )


def sum_ledger_cost(path: str | Path) -> float:
    """Sum dollar cost for a ledger file without verifying the hash chain."""
    return sum_entry_costs(read_ledger_entries(path))


def verified_ledger_total_usd(path: str | Path) -> float | None:
    """Return total ledger spend in USD when the hash chain is intact.

    Args:
        path: ``cost_ledger.jsonl`` written by :func:`openrouter_client._ledger_append`.

    Returns:
        Rounded total cost, or ``None`` when verification fails or the path is
        unreadable.
    """
    try:
        valid, _, _ = verify_ledger_chain(path)
        if not valid:
            return None
        return sum_ledger_cost(path)
    except OSError:
        return None
