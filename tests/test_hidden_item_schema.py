"""Tests for the canonical hidden-benchmark item schema (issue #11)."""
from __future__ import annotations

from trinity.adapters import get_adapter
from trinity.adapters.hidden_item import (
    CANONICAL_ITEM_FIELDS,
    build_hidden_item,
    from_adapter_task,
    to_protocol_item,
)


def test_canonical_fields_cover_issue_schema():
    for field in (
        "task_id",
        "benchmark",
        "prompt",
        "reference",
        "task_type",
        "cached_model_answers",
        "cached_model_scores",
        "meta",
    ):
        assert field in CANONICAL_ITEM_FIELDS


def test_from_adapter_task_round_trips_through_protocol_shape():
    adapter = get_adapter("math500")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    canonical = from_adapter_task(
        adapter,
        task,
        cached_model_answers={"m1": "answer"},
        cached_model_scores={"m1": 1.0},
    )
    protocol_item = to_protocol_item(canonical)
    assert protocol_item["question_id"] == canonical["task_id"]
    assert protocol_item["question_text"] == canonical["prompt"]
    assert protocol_item["correct_answer"] == canonical["reference"]
    assert protocol_item["model_answers"] == {"m1": "answer"}
    assert protocol_item["model_scores"] == {"m1": 1.0}


def test_build_hidden_item_defaults_cached_fields_to_empty_dicts():
    item = build_hidden_item(
        task_id="q1",
        benchmark="mmlu",
        prompt="Q?",
        reference="A",
        task_type="mcq",
    )
    assert item["cached_model_answers"] == {}
    assert item["cached_model_scores"] == {}
