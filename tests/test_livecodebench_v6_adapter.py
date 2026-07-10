"""Tests for the first-class LiveCodeBench v6 adapter (issue #13).

All offline: task loading falls back to the built-in LiveCodeBench toy set when
``datasets``/network are unavailable, and pass@1 scoring runs candidate code in
the same sandboxed subprocess the reward checker uses — no network required.
"""
from __future__ import annotations

import json

from trinity.adapters import TaskType, get_adapter
from trinity.adapters.livecodebench import LiveCodeBenchV6Adapter


def test_v6_adapter_is_registered_and_selectable():
    adapter = get_adapter("livecodebench_v6")
    assert isinstance(adapter, LiveCodeBenchV6Adapter)
    # Case-insensitive lookup, like the other built-ins.
    assert get_adapter("LiveCodeBench_V6") is adapter


def test_task_type_is_code():
    assert get_adapter("livecodebench_v6").task_type() is TaskType.CODE


def test_metadata_pins_release_v6():
    meta = get_adapter("livecodebench_v6").metadata()
    assert meta["dataset_id"] == "livecodebench/code_generation_lite"
    assert meta["eval_version"] == "release_v6"
    assert meta["train_version"] == "release_v1"


def test_resolve_version_keeps_train_and_eval_explicit():
    adapter = get_adapter("livecodebench_v6")
    assert adapter.resolve_version("train") == "release_v1"
    assert adapter.resolve_version("v1") == "release_v1"
    # Anything that is not an explicit training split freezes to the eval release.
    for split in ("test", "eval", "v6", "release_v6", ""):
        assert adapter.resolve_version(split) == "release_v6"


def test_load_tasks_stamp_release_and_are_deterministic():
    adapter = get_adapter("livecodebench_v6")
    a = adapter.load_tasks("test", max_items=2, seed=7)
    b = adapter.load_tasks("test", max_items=2, seed=7)
    assert a  # offline toy fallback is non-empty
    assert [t.task_id for t in a] == [t.task_id for t in b]
    for task in a:
        assert task.meta["dataset_version"] == "release_v6"
        assert task.meta["dataset_id"] == "livecodebench/code_generation_lite"
        assert task.meta["adapter"] == "livecodebench_v6"


def test_load_tasks_train_split_pins_v1():
    adapter = get_adapter("livecodebench_v6")
    tasks = adapter.load_tasks("train", max_items=1, seed=0)
    assert tasks
    for task in tasks:
        assert task.meta["dataset_version"] == "release_v1"


def test_score_output_runs_pass_at_1():
    adapter = get_adapter("livecodebench_v6")
    spec = {
        "tests": [
            {"input": "3\n", "output": "9"},
            {"input": "5\n", "output": "25"},
        ],
        "fn_name": None,
        "starter_code": None,
    }
    good = "n = int(input())\nprint(n * n)\n"
    bad = "n = int(input())\nprint(n + 1)\n"
    assert adapter.score_output(good, spec) == 1.0
    assert adapter.score_output(bad, spec) == 0.0


def test_serialize_task_is_json_safe_and_versioned():
    adapter = get_adapter("livecodebench_v6")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    item = adapter.serialize_task(task)
    assert item["benchmark"] == "livecodebench_v6"
    assert item["task_type"] == TaskType.CODE.value
    assert item["dataset_version"] == "release_v6"
    assert item["reference"] == task.answer
    # Frozen hidden-benchmark item format must round-trip through JSON.
    assert json.loads(json.dumps(item))["benchmark"] == "livecodebench_v6"
