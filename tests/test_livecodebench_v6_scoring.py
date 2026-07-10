"""Frozen LiveCodeBench-v6 items must be scoreable, not raise "Unknown benchmark".

The v6 adapter serialises its release *identity* (``"livecodebench_v6"``) as the
frozen item's ``benchmark`` field (pinned by ``test_livecodebench_v6_adapter``).
The reward dispatch is keyed on family names, so it must resolve that identity to
the ``livecodebench`` code family; otherwise ``score_text``/``has_answer`` raise
``ValueError: Unknown benchmark`` on every frozen v6 item scored by ``pr_eval``.

Pure / offline — pass@1 runs candidate code in the same sandboxed subprocess the
reward checker uses (no network, no GPU).
"""
from __future__ import annotations

import pytest

from trinity.adapters import get_adapter
from trinity.adapters.hidden_item import from_adapter_task, to_protocol_item
from trinity.orchestration.reward import has_answer, resolve_benchmark, score_text

# A minimal stdin/stdout LiveCodeBench-style spec: print n squared.
_SPEC = {"tests": [{"input": "3\n", "output": "9"}, {"input": "5\n", "output": "25"}]}
_GOOD = "n = int(input())\nprint(n * n)\n"
_BAD = "n = int(input())\nprint(n + 1)\n"


def test_resolve_benchmark_maps_v6_identity_to_family():
    assert resolve_benchmark("livecodebench_v6") == "livecodebench"
    # Case-insensitive and whitespace-tolerant, like the dispatch keys.
    assert resolve_benchmark("  LiveCodeBench_V6 ") == "livecodebench"


def test_resolve_benchmark_is_a_noop_for_known_and_unknown_keys():
    for key in ("math500", "mmlu", "gpqa", "livecodebench", "lcb", "bigcodebench"):
        assert resolve_benchmark(key) == key
    # An unrecognized name is returned unchanged (still rejected by score_text).
    assert resolve_benchmark("totally_unknown") == "totally_unknown"
    assert resolve_benchmark("") == ""


def test_has_answer_recognizes_the_v6_identity():
    assert has_answer("livecodebench_v6", "```python\ndef f():\n    return 1\n```") is True
    assert has_answer("livecodebench_v6", "just prose, no code") is False


def test_score_text_scores_a_v6_item_without_raising():
    # The bug: this raised ValueError("Unknown benchmark 'livecodebench_v6'").
    assert score_text("livecodebench_v6", _GOOD, _SPEC) == 1.0
    assert score_text("livecodebench_v6", _BAD, _SPEC) == 0.0


def test_a_truly_unknown_benchmark_still_raises():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        score_text("nonsense_bench", _GOOD, _SPEC)


def test_frozen_v6_item_round_trips_through_scoring():
    # End-to-end: adapter task -> canonical item -> on-disk protocol item -> score.
    adapter = get_adapter("livecodebench_v6")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    # Use a controlled spec/answer so the assertion does not depend on toy-set
    # internals, but keep the adapter-stamped benchmark identity.
    task.answer = _SPEC
    protocol_item = to_protocol_item(from_adapter_task(adapter, task))
    assert protocol_item["benchmark"] == "livecodebench_v6"  # identity preserved
    # pr_eval grades with exactly this key; it must resolve to the code scorer.
    assert score_text(protocol_item["benchmark"], _GOOD, protocol_item["correct_answer"]) == 1.0
    assert score_text(protocol_item["benchmark"], _BAD, protocol_item["correct_answer"]) == 0.0
