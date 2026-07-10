"""Tests for the canonical hidden-benchmark item schema (issue #11)."""
from __future__ import annotations

from trinity.adapters import get_adapter
from trinity.adapters.hidden_item import (
    CANONICAL_ITEM_FIELDS,
    build_hidden_item,
    from_adapter_task,
    from_protocol_item,
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


def test_from_protocol_item_inverts_to_protocol_item():
    """to_protocol_item -> from_protocol_item recovers the canonical item.

    This is what lets the evaluator read builder output through the same schema
    the builder emits.
    """
    canonical = build_hidden_item(
        task_id="q1",
        benchmark="math500",
        prompt="What is 2+2?",
        reference="4",
        task_type="math",
        cached_model_answers={"m1": "4"},
        cached_model_scores={"m1": 1.0},
        meta={"source": "unit"},
    )
    recovered = from_protocol_item(to_protocol_item(canonical))
    assert recovered == canonical


def test_from_protocol_item_reads_legacy_question_keys():
    """Pre-#59 on-disk items use question_id / question_text / correct_answer."""
    legacy = {
        "question_id": "q7",
        "question_text": "Capital of France?",
        "correct_answer": "Paris",
        "benchmark": "trivia",
        "model_answers": {"m1": "Paris"},
    }
    canonical = from_protocol_item(legacy)
    assert canonical["task_id"] == "q7"
    assert canonical["prompt"] == "Capital of France?"
    assert canonical["reference"] == "Paris"
    assert canonical["cached_model_answers"] == {"m1": "Paris"}


def test_from_protocol_item_preserves_non_string_reference():
    """Code / SWE-bench patch references are dicts and must not be stringified."""
    patch_reference = {"repo": "org/proj", "test_patch": "diff --git ...", "fail_to_pass": ["t1"]}
    on_disk = {
        "question_id": "swe-1",
        "question_text": "Fix the bug",
        "correct_answer": patch_reference,
        "benchmark": "swebench",
        "task_type": "patch",
    }
    canonical = from_protocol_item(on_disk)
    assert canonical["reference"] == patch_reference
    assert isinstance(canonical["reference"], dict)
