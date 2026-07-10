"""Offline tests for OpenRouter ledger pricing (per-model in/out split).

Regression cover for the bug where ``pack_submission._estimate_cost`` priced
every ledger row as ``0.90 * (prompt + completion) / 1M`` instead of applying
the same per-million input/output rates as ``scripts/cost_report.py``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from trinity.llm import cost_ledger as CL
from trinity.llm import openrouter_pricing as OP

_REPO = Path(__file__).resolve().parents[1]
_PACK = _REPO / "scripts" / "pack_submission.py"
_spec = importlib.util.spec_from_file_location("pack_submission", _PACK)
pack_submission = importlib.util.module_from_spec(_spec)
sys.modules["pack_submission"] = pack_submission
_spec.loader.exec_module(pack_submission)


# --------------------------------------------------------------------------- #
# Rate resolution
# --------------------------------------------------------------------------- #
def test_resolve_rates_returns_pool_table_for_known_slug():
    assert OP.resolve_rates("qwen3.5-35b-a3b") == (0.14, 1.00)
    assert OP.resolve_rates("openrouter/qwen3.5-35b-a3b") == (0.14, 1.00)


def test_resolve_rates_falls_back_to_blended_default_for_unknown():
    default_in, default_out = OP.default_blended_rates()
    assert OP.resolve_rates("unknown-model") == (default_in, default_out)


def test_token_cost_splits_prompt_and_completion_rates():
    # 1M prompt @ 0.14 + 1M completion @ 1.00 = 1.14 for qwen
    assert OP.token_cost("qwen3.5-35b-a3b", 1_000_000, 1_000_000) == pytest.approx(1.14)


def test_flat_blended_formula_differs_from_split_pricing():
    """The old bug: one rate on (p+c) is not the same as split in/out."""
    prompt_only = 1_000_000
    completion_only = 0
    split = OP.token_cost("qwen3.5-35b-a3b", prompt_only, completion_only)
    flat_bug = 0.90 * (prompt_only + completion_only) / 1_000_000
    assert split == pytest.approx(0.14)
    assert flat_bug == pytest.approx(0.90)
    assert split != pytest.approx(flat_bug)


@pytest.mark.parametrize(
    "model, prompt, completion, expected",
    [
        ("qwen3.5-35b-a3b", 1_000_000, 0, 0.14),
        ("qwen3.5-35b-a3b", 0, 1_000_000, 1.00),
        ("minimax-m3", 1_000_000, 1_000_000, 1.50),
        ("deepseek-v4-flash", 2_000_000, 500_000, 0.27),
    ],
)
def test_token_cost_matches_manual_arithmetic(model, prompt, completion, expected):
    in_rate, out_rate = OP.resolve_rates(model)
    manual = prompt / 1e6 * in_rate + completion / 1e6 * out_rate
    assert OP.token_cost(model, prompt, completion) == pytest.approx(manual)
    assert OP.token_cost(model, prompt, completion) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Ledger totals
# --------------------------------------------------------------------------- #
def test_sum_entry_costs_aggregates_multiple_models(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 1_000_000, 0)
    CL.append_ledger_entry(ledger, "minimax-m3", 0, 1_000_000)
    entries = CL.read_ledger_entries(ledger)
    # 0.14 (qwen prompt) + 1.20 (minimax completion)
    assert OP.sum_entry_costs(entries) == pytest.approx(1.34)


def test_verified_ledger_total_usd_requires_intact_chain(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "deepseek-v4-flash", 1_000_000, 1_000_000)
    total = OP.verified_ledger_total_usd(ledger)
    assert total == pytest.approx(0.27)


def test_verified_ledger_total_usd_returns_none_on_tampered_chain(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 100, 50)
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(text.replace('"p":100', '"p":999'), encoding="utf-8")
    assert OP.verified_ledger_total_usd(ledger) is None


def test_verified_ledger_total_usd_returns_none_for_missing_file(tmp_path: Path):
    assert OP.verified_ledger_total_usd(tmp_path / "missing.jsonl") is None


# --------------------------------------------------------------------------- #
# pack_submission integration
# --------------------------------------------------------------------------- #
def test_pack_submission_estimate_cost_matches_pricing_module(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    CL.append_ledger_entry(ledger, "qwen3.5-35b-a3b", 2_000_000, 500_000)
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))
    assert pack_submission._estimate_cost() == round(OP.verified_ledger_total_usd(ledger), 4)


def test_pack_submission_estimate_cost_zero_without_ledger(monkeypatch):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    assert pack_submission._estimate_cost() == 0.0


def test_pack_submission_estimate_cost_zero_on_broken_chain(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"m":"qwen3.5-35b-a3b","p":1,"c":2}\n', encoding="utf-8")
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))
    assert pack_submission._estimate_cost() == 0.0
