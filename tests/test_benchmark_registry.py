"""Tests for the benchmark adapter registry (issue #9).

All offline: task loading falls back to the built-in toy sets in
``dataset.py`` when ``datasets``/network are unavailable, so these run with zero
network exactly like the other smoke tests.
"""
from __future__ import annotations

import pytest

from trinity.adapters import (
    BenchmarkAdapter,
    DelegatingBenchmarkAdapter,
    TaskType,
    available_adapters,
    clear_registry,
    get_adapter,
    is_registered,
    register_adapter,
    register_builtin_adapters,
)
from trinity.orchestration.dataset import SUPPORTED_BENCHMARKS

# Known-correct / known-wrong pairs per built-in benchmark, mirroring the toy
# sets the offline loader returns.
_SCORING_CASES = {
    "math500": ("The answer is \\boxed{4}.", "\\boxed{9}.", "4"),
    "mmlu": ("The answer is B.", "The answer is A.", "B"),
    "gpqa": ("Answer: B", "Answer: C", "B"),
}


def test_builtins_registered_for_every_supported_benchmark():
    for name in SUPPORTED_BENCHMARKS:
        assert is_registered(name)
        assert isinstance(get_adapter(name), BenchmarkAdapter)
    # The registry exposes exactly the supported benchmarks (no more, no less).
    assert set(available_adapters()) == set(SUPPORTED_BENCHMARKS)


def test_lookup_is_case_insensitive():
    assert get_adapter("MATH500") is get_adapter("math500")


def test_get_adapter_unknown_lists_available():
    with pytest.raises(KeyError) as excinfo:
        get_adapter("does-not-exist")
    assert "math500" in str(excinfo.value)


def test_task_type_matches_family():
    assert get_adapter("math500").task_type() is TaskType.MATH
    assert get_adapter("mmlu").task_type() is TaskType.MCQ
    assert get_adapter("gpqa").task_type() is TaskType.MCQ
    assert get_adapter("livecodebench").task_type() is TaskType.CODE


def test_load_tasks_deterministic_and_capped():
    adapter = get_adapter("math500")
    a = adapter.load_tasks("test", max_items=2, seed=42)
    b = adapter.load_tasks("test", max_items=2, seed=42)
    assert len(a) <= 2
    assert [t.task_id for t in a] == [t.task_id for t in b]
    for t in a:
        assert t.benchmark == "math500"
        # build_prompt returns exactly what the pool model receives.
        assert adapter.build_prompt(t) == t.prompt


@pytest.mark.parametrize("name", list(_SCORING_CASES))
def test_score_output_binary_correct_and_wrong(name):
    adapter = get_adapter(name)
    right, wrong, ref = _SCORING_CASES[name]
    assert adapter.score_output(right, ref) == 1.0
    assert adapter.score_output(wrong, ref) == 0.0


def test_score_output_code_runs_tests():
    adapter = get_adapter("livecodebench")
    spec = {
        "tests": [{"input": "3\n", "output": "9"}, {"input": "5\n", "output": "25"}],
        "fn_name": None,
        "starter_code": None,
    }
    good = "n = int(input())\nprint(n * n)\n"
    bad = "n = int(input())\nprint(n + 1)\n"
    assert adapter.score_output(good, spec) == 1.0
    assert adapter.score_output(bad, spec) == 0.0


def test_serialize_task_is_json_safe():
    import json

    adapter = get_adapter("mmlu")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    item = adapter.serialize_task(task)
    assert item["task_id"] == task.task_id
    assert item["benchmark"] == "mmlu"
    assert item["task_type"] == TaskType.MCQ.value
    assert item["reference"] == task.answer
    # Must round-trip through JSON (frozen hidden-benchmark item format).
    assert json.loads(json.dumps(item))["task_id"] == task.task_id


def test_cache_baselines_default_is_noop():
    adapter = get_adapter("math500")
    task = adapter.load_tasks("test", max_items=1, seed=0)[0]
    assert adapter.cache_baselines(task, pool=None) is None


def test_delegating_adapter_rejects_unknown_benchmark():
    with pytest.raises(ValueError):
        DelegatingBenchmarkAdapter("totally-unknown-benchmark")


def test_register_duplicate_name_raises():
    with pytest.raises(ValueError):
        register_adapter("math500", DelegatingBenchmarkAdapter("math500"))


def test_decorator_registration_and_registry_isolation():
    """A throwaway registration on a cleared registry, then restored built-ins."""
    try:
        clear_registry()
        assert available_adapters() == ()

        @register_adapter("unit-test-bench")
        class _StubAdapter(BenchmarkAdapter):
            name = "unit-test-bench"

            def load_tasks(self, split, max_items, seed=0):
                return []

            def build_prompt(self, task):
                return task.prompt

            def score_output(self, output, reference):
                return 1.0 if output == reference else 0.0

            def task_type(self):
                return TaskType.MATH

            def serialize_task(self, task):
                return {"task_id": task.task_id}

        assert is_registered("unit-test-bench")
        assert get_adapter("unit-test-bench").score_output("x", "x") == 1.0
    finally:
        # Never leak test state into other tests: restore the real registry.
        clear_registry()
        register_builtin_adapters()

    assert set(available_adapters()) == set(SUPPORTED_BENCHMARKS)
