"""Offline tests for OpenRouter response parsing. No network, no GPU.

An OpenAI-compatible endpoint can return HTTP 200 with no usable completion:
a content-moderation block (``choices: []``), an error envelope surfaced as 200
(no ``choices`` key), or a choice without a ``message``. ``_parse_completion``
must degrade these to an empty ``ChatResult`` instead of raising
``IndexError``/``KeyError`` out of the retry — while still surfacing the ``usage``
token counts so billed tokens are recorded to the cost ledger.
"""
from __future__ import annotations

import json

from trinity.llm.openrouter_client import _ledger_append, _parse_completion


# ---------------------------------------------------------------------------
# Empty / malformed 200 bodies must not raise.
# ---------------------------------------------------------------------------
def test_empty_choices_degrades_to_empty_text_and_keeps_usage():
    data = {"id": "gen-x", "choices": [], "usage": {"prompt_tokens": 812, "completion_tokens": 0}}
    r = _parse_completion(data, "m")
    assert r.text == ""
    assert r.finish_reason == "error"
    assert r.prompt_tokens == 812  # billed tokens preserved
    assert r.completion_tokens == 0
    assert r.model == "m"


def test_missing_choices_key_error_envelope_does_not_raise():
    data = {"error": {"message": "blocked by content policy", "code": 429}}
    r = _parse_completion(data, "m")
    assert r.text == ""
    assert r.finish_reason == "error"


def test_choice_without_message_yields_empty_text_but_keeps_finish_reason():
    data = {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 0}}
    r = _parse_completion(data, "m")
    assert r.text == ""
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 5


def test_null_choice_entry_does_not_raise():
    r = _parse_completion({"choices": [None]}, "m")
    assert r.text == ""


def test_missing_usage_defaults_to_zero():
    r = _parse_completion({"choices": []}, "m")
    assert r.prompt_tokens == 0 and r.completion_tokens == 0


# ---------------------------------------------------------------------------
# Happy path is preserved.
# ---------------------------------------------------------------------------
def test_normal_completion_parses_text_and_tokens():
    data = {
        "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    r = _parse_completion(data, "deepseek")
    assert r.text == "hi there"
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 10 and r.completion_tokens == 3
    assert r.model == "deepseek"


def test_null_message_content_is_empty_not_the_string_none():
    data = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    assert _parse_completion(data, "m").text == ""


# ---------------------------------------------------------------------------
# Cost accounting: a billed-but-empty response still records usage.
# ---------------------------------------------------------------------------
def test_empty_completion_usage_is_recorded_to_the_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "cost_ledger.jsonl"
    monkeypatch.setenv("TRINITY_COST_LEDGER", str(ledger))
    data = {"choices": [], "usage": {"prompt_tokens": 812, "completion_tokens": 0}}
    r = _parse_completion(data, "m")
    _ledger_append(r.model, r.prompt_tokens, r.completion_tokens)
    assert ledger.exists(), "billed tokens on an empty completion must be recorded"
    lines = [ln for ln in ledger.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    # The ledger stores compact keys: m=model, p=prompt_tokens, c=completion_tokens.
    assert entry.get("p") == 812
