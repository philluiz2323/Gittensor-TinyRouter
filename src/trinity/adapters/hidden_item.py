"""Canonical hidden-benchmark item schema shared by builder and evaluator (issue #11)."""
from __future__ import annotations

from typing import Any, Mapping

from trinity.adapters.base import BenchmarkAdapter
from trinity.types import Task

__all__ = [
    "CANONICAL_ITEM_FIELDS",
    "build_hidden_item",
    "from_adapter_task",
    "to_protocol_item",
]

#: Fields every hidden-benchmark item carries across benchmark families.
CANONICAL_ITEM_FIELDS: tuple[str, ...] = (
    "task_id",
    "benchmark",
    "prompt",
    "reference",
    "task_type",
    "cached_model_answers",
    "cached_model_scores",
    "meta",
)


def build_hidden_item(
    *,
    task_id: str,
    benchmark: str,
    prompt: str,
    reference: Any,
    task_type: str,
    meta: Mapping[str, Any] | None = None,
    cached_model_answers: Mapping[str, str] | None = None,
    cached_model_scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return the canonical JSON-safe hidden-benchmark item dict."""
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "prompt": prompt,
        "reference": reference,
        "task_type": task_type,
        "cached_model_answers": dict(cached_model_answers or {}),
        "cached_model_scores": dict(cached_model_scores or {}),
        "meta": dict(meta or {}),
    }


def from_adapter_task(
    adapter: BenchmarkAdapter,
    task: Task,
    *,
    cached_model_answers: Mapping[str, str] | None = None,
    cached_model_scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build a canonical item from an adapter's :meth:`serialize_task` output."""
    serialized = adapter.serialize_task(task)
    return build_hidden_item(
        task_id=str(serialized["task_id"]),
        benchmark=str(serialized["benchmark"]),
        prompt=str(serialized["prompt"]),
        reference=serialized["reference"],
        task_type=str(serialized["task_type"]),
        meta=serialized.get("meta", {}),
        cached_model_answers=cached_model_answers,
        cached_model_scores=cached_model_scores,
    )


def to_protocol_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Map a canonical item to the legacy ``pr_eval`` / builder on-disk shape.

    The frozen protocol still names the prompt ``question_text`` and the gold
    answer ``correct_answer``; this helper is the single conversion point.
    """
    cached_answers = item.get("cached_model_answers") or {}
    protocol_item = {
        "question_id": item["task_id"],
        "question_text": item["prompt"],
        "task_type": item["task_type"],
        "benchmark": item["benchmark"],
        "correct_answer": item["reference"],
        "model_answers": dict(cached_answers),
    }
    if item.get("meta"):
        protocol_item["meta"] = dict(item["meta"])
    scores = item.get("cached_model_scores") or {}
    if scores:
        protocol_item["model_scores"] = dict(scores)
    return protocol_item
