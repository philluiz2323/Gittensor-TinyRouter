"""Offline tests for the canonical cost-ledger hash chain.

Covers payload formatting, chained hashing, verification, tamper detection,
multi-entry chains, blank-line tolerance, and integration with
``scripts/cost_report.verify_ledger_chain``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from trinity.llm import cost_ledger as CL
from trinity.llm.openrouter_client import _ledger_append

_REPO = Path(__file__).resolve().parents[1]
_COST_REPORT = _REPO / "scripts" / "cost_report.py"
_spec = importlib.util.spec_from_file_location("cost_report", _COST_REPORT)
cost_report = importlib.util.module_from_spec(_spec)
sys.modules["cost_report"] = cost_report
_spec.loader.exec_module(cost_report)


# ---------------------------------------------------------------------------
# Payload + hash primitives
# ---------------------------------------------------------------------------


def test_ledger_payload_uses_compact_fixed_key_order():
    assert CL.ledger_payload("qwen/qwen3.5-35b-a3b", 100, 50) == (
        '{"m":"qwen3.5-35b-a3b","p":100,"c":50}'
    )


def test_ledger_payload_strips_provider_prefix_from_model_slug():
    assert CL.ledger_payload("deepseek/deepseek-v4-flash", 1, 2) == (
        '{"m":"deepseek-v4-flash","p":1,"c":2}'
    )


def test_ledger_entry_hash_matches_manual_sha256():
    prev = ""
    payload = CL.ledger_payload("minimax-m3", 10, 20)
    expected = hashlib.sha256((prev + payload).encode()).hexdigest()
    assert CL.ledger_entry_hash(prev, "minimax-m3", 10, 20) == expected


def test_ledger_entry_hash_links_to_previous_digest():
    first = CL.ledger_entry_hash("", "qwen3.5-35b-a3b", 5, 5)
    second = CL.ledger_entry_hash(first, "minimax-m3", 1, 1)
    payload = CL.ledger_payload("minimax-m3", 1, 1)
    assert second == hashlib.sha256((first + payload).encode()).hexdigest()


def test_format_ledger_line_round_trips_through_parse():
    line = CL.format_ledger_line("qwen3.5-35b-a3b", 42, 7)
    entry, digest = CL.parse_ledger_line(line)
    assert entry.model == "qwen3.5-35b-a3b"
    assert entry.prompt_tokens == 42
    assert entry.completion_tokens == 7
    assert digest == json.loads(line)["h"]


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------


def _write_chain(tmp_path: Path, models: list[tuple[str, int, int]]) -> Path:
    path = tmp_path / "ledger.jsonl"
    prev = ""
    lines: list[str] = []
    for model, p, c in models:
        line = CL.format_ledger_line(model, p, c, prev_hash=prev)
        prev = json.loads(line)["h"]
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_verify_ledger_chain_accepts_valid_single_entry(tmp_path):
    path = _write_chain(tmp_path, [("qwen3.5-35b-a3b", 100, 50)])
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is True
    assert count == 1
    assert err == ""


def test_verify_ledger_chain_accepts_valid_multi_entry_chain(tmp_path):
    path = _write_chain(
        tmp_path,
        [
            ("qwen3.5-35b-a3b", 10, 5),
            ("minimax-m3", 20, 15),
            ("deepseek-v4-flash", 1, 1),
        ],
    )
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is True
    assert count == 3
    assert err == ""


def test_verify_ledger_chain_skips_blank_lines(tmp_path):
    path = _write_chain(tmp_path, [("qwen3.5-35b-a3b", 1, 1)])
    text = path.read_text(encoding="utf-8")
    path.write_text("\n\n" + text + "\n\n", encoding="utf-8")
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is True
    assert count == 1
    assert err == ""


def test_verify_ledger_chain_rejects_tampered_token_count(tmp_path):
    path = _write_chain(tmp_path, [("qwen3.5-35b-a3b", 100, 50)])
    text = path.read_text(encoding="utf-8").replace('"p":100', '"p":999')
    path.write_text(text, encoding="utf-8")
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is False
    assert count == 0
    assert "hash mismatch" in err


def test_verify_ledger_chain_rejects_missing_hash_field(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"m":"qwen3.5-35b-a3b","p":1,"c":1}\n', encoding="utf-8")
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is False
    assert "missing hash" in err


def test_verify_ledger_chain_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is False
    assert "invalid JSON" in err


def test_verify_ledger_chain_rejects_broken_link_in_multi_entry_chain(tmp_path):
    path = _write_chain(
        tmp_path,
        [("qwen3.5-35b-a3b", 1, 1), ("minimax-m3", 2, 2)],
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["h"] = "0" * 64
    lines[1] = json.dumps(second, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    valid, _, err = CL.verify_ledger_chain(path)
    assert valid is False
    assert "hash mismatch" in err


# ---------------------------------------------------------------------------
# Append + read helpers
# ---------------------------------------------------------------------------


def test_append_ledger_entry_builds_chain_on_disk(tmp_path):
    path = tmp_path / "ledger.jsonl"
    h1 = CL.append_ledger_entry(path, "qwen3.5-35b-a3b", 10, 5)
    h2 = CL.append_ledger_entry(path, "minimax-m3", 3, 7)
    assert h1 != h2
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is True
    assert count == 2
    assert err == ""


def test_append_continues_from_last_line_after_broken_prefix(tmp_path):
    """Regression: a broken earlier line must not reset later appends to genesis.

    After #87/#88, append only linked when the *entire* chain verified. One race
    (or tamper) then made every subsequent write a new genesis, so cost_report
    and pack_submission could never recover a usable chain tip.
    """
    path = tmp_path / "ledger.jsonl"
    first = CL.format_ledger_line("qwen3.5-35b-a3b", 1, 1, prev_hash="")
    first_h = json.loads(first)["h"]
    # Deliberately broken second line (wrong hash) — chain no longer verifies.
    broken = '{"m":"minimax-m3","p":2,"c":2,"h":"' + ("0" * 64) + '"}'
    path.write_text(first + "\n" + broken + "\n", encoding="utf-8")
    assert CL.verify_ledger_chain(path)[0] is False

    third_h = CL.append_ledger_entry(path, "deepseek-v4-flash", 3, 3)
    tip_before_third = CL.tip_hash_from_text(broken)
    assert tip_before_third == "0" * 64
    expected = CL.ledger_entry_hash(tip_before_third, "deepseek-v4-flash", 3, 3)
    assert third_h == expected
    # Must NOT be a genesis hash (prev_hash="").
    assert third_h != CL.ledger_entry_hash("", "deepseek-v4-flash", 3, 3)
    # Tip still advances from the last written line.
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    assert json.loads(lines[0])["h"] == first_h
    assert json.loads(lines[-1])["h"] == third_h


def test_append_via_file_handle_builds_verifiable_chain():
    buf = StringIO()
    h1 = CL.append_ledger_entry("unused.jsonl", "qwen3.5-35b-a3b", 10, 5, file_handle=buf)
    h2 = CL.append_ledger_entry("unused.jsonl", "minimax-m3", 3, 7, file_handle=buf)
    assert h2 == CL.ledger_entry_hash(h1, "minimax-m3", 3, 7)
    valid, count, err = CL.verify_ledger_chain_text(buf.getvalue())
    assert valid is True
    assert count == 2
    assert err == ""


def test_tip_hash_from_text_returns_empty_for_blank():
    assert CL.tip_hash_from_text("") == ""
    assert CL.tip_hash_from_text("\n\n") == ""


def test_read_ledger_entries_preserves_order(tmp_path):
    path = _write_chain(
        tmp_path,
        [("qwen3.5-35b-a3b", 1, 2), ("minimax-m3", 3, 4)],
    )
    entries = CL.read_ledger_entries(path)
    assert [e.model for e in entries] == ["qwen3.5-35b-a3b", "minimax-m3"]
    assert entries[1].total_tokens == 7


def test_summarize_token_usage_aggregates_per_model():
    entries = [
        CL.LedgerEntry("qwen3.5-35b-a3b", 10, 5),
        CL.LedgerEntry("qwen3.5-35b-a3b", 20, 15),
        CL.LedgerEntry("minimax-m3", 1, 1),
    ]
    summary = CL.summarize_token_usage(entries)
    assert summary["qwen3.5-35b-a3b"] == (30, 20, 2)
    assert summary["minimax-m3"] == (1, 1, 1)


# ---------------------------------------------------------------------------
# OpenRouter writer integration
# ---------------------------------------------------------------------------


def test_openrouter_ledger_append_writes_verifiable_chain(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(path))
    _ledger_append("qwen/qwen3.5-35b-a3b", 100, 25)
    _ledger_append("minimax/minimax-m3", 50, 50)
    valid, count, err = CL.verify_ledger_chain(path)
    assert valid is True
    assert count == 2
    assert err == ""


def test_openrouter_ledger_append_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRINITY_COST_LEDGER", raising=False)
    path = tmp_path / "ledger.jsonl"
    _ledger_append("qwen3.5-35b-a3b", 1, 1)
    assert not path.exists()


# ---------------------------------------------------------------------------
# cost_report.py re-export path
# ---------------------------------------------------------------------------


def test_cost_report_verify_delegates_to_canonical_module(tmp_path):
    path = _write_chain(tmp_path, [("qwen3.5-35b-a3b", 5, 5)])
    valid, count, err = cost_report.verify_ledger_chain(str(path))
    assert valid is True
    assert count == 1
    assert err == ""


def test_cost_report_verify_rejects_sort_keys_payload_mismatch(tmp_path):
    """Regression: old verifier used json.dumps(sort_keys=True)."""
    payload = json.dumps({"m": "qwen3.5-35b-a3b", "p": 10, "c": 5}, sort_keys=True)
    wrong_h = hashlib.sha256(payload.encode()).hexdigest()
    line = json.dumps(
        {"m": "qwen3.5-35b-a3b", "p": 10, "c": 5, "h": wrong_h},
        sort_keys=True,
    )
    path = tmp_path / "legacy.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    valid, _, err = cost_report.verify_ledger_chain(str(path))
    assert valid is False
    assert "hash mismatch" in err


def test_parse_ledger_line_rejects_unexpected_extra_fields():
    line = '{"m":"qwen3.5-35b-a3b","p":1,"c":1,"extra":true,"h":"abc"}'
    with pytest.raises(ValueError, match="unexpected fields"):
        CL.parse_ledger_line(line, line_number=3)


def test_verify_ledger_chain_text_accepts_in_memory_buffer():
    line = CL.format_ledger_line("deepseek-v4-flash", 9, 9)
    buf = StringIO(line + "\n")
    valid, count, err = CL.verify_ledger_chain_text(buf.getvalue())
    assert valid is True
    assert count == 1
    assert err == ""
