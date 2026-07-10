"""Tests that build_benchmark routes through benchmark adapters (issue #15)."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_build_benchmark():
    spec = importlib.util.spec_from_file_location(
        "build_benchmark",
        _REPO / "scripts" / "build_benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_benchmark():
    return _load_build_benchmark()


def test_task_to_item_uses_registered_adapter(build_benchmark):
    from trinity.adapters import get_adapter

    adapter = get_adapter("math500")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    item = build_benchmark._task_to_item(task, 0)
    assert item["benchmark"] == "math500"
    assert item["question_text"] == task.prompt
    assert item["correct_answer"] == task.answer
    assert item["model_answers"] == {}


def test_cache_answers_uses_adapter_build_prompt(build_benchmark):
    from trinity.adapters import get_adapter
    from trinity.roles.prompts import build_messages
    from trinity.types import Role

    adapter = get_adapter("math500")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    item = {
        "question_id": task.task_id,
        "benchmark": "math500",
        "question_text": "stale prompt",
        "model_answers": {},
    }
    pool = MagicMock()
    pool.chat = AsyncMock(return_value=MagicMock(text="cached"))
    pool.models = {"m1": {}}

    asyncio.run(build_benchmark._cache_answers([(task, item)], pool, ["m1"]))

    pool.chat.assert_awaited_once()
    expected_messages = build_messages(Role.WORKER, adapter.build_prompt(task), [])
    assert pool.chat.await_args.args[1] == expected_messages
    assert item["model_answers"]["m1"] == "cached"
