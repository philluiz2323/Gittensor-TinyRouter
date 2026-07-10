"""Cached vs execution scoring split (issue #16).

One top-level API — :func:`score_item` — grades both a cheap cached benchmark
(MMLU/math) and an expensive execution benchmark (SWE-bench) without the caller
knowing which is which. Offline.
"""
from __future__ import annotations

from trinity.adapters import (
    ScoringMode,
    ScoringOutcome,
    get_adapter,
    requires_execution,
    score_item,
    supports_execution,
)
from trinity.adapters.base import BenchmarkAdapter, TaskType


def test_default_adapter_is_cached_only():
    for name in ("math500", "mmlu", "gpqa", "livecodebench", "mmlu_pro"):
        modes = get_adapter(name).scoring_modes()
        assert modes == frozenset({ScoringMode.CACHED})
        assert not supports_execution(get_adapter(name))
        assert not requires_execution(get_adapter(name))


def test_swebench_declares_both_modes():
    adapter = get_adapter("swebench_verified")
    assert adapter.scoring_modes() == frozenset({ScoringMode.CACHED, ScoringMode.EXECUTION})
    assert supports_execution(adapter)
    assert not requires_execution(adapter)  # it also has a cached path


def test_score_item_cached_path_for_mcq():
    adapter = get_adapter("mmlu")
    out = score_item(adapter, "The answer is B.", "B")
    assert isinstance(out, ScoringOutcome)
    assert out.reward == 1.0
    assert out.mode is ScoringMode.CACHED
    assert score_item(adapter, "The answer is A.", "B").reward == 0.0


def test_score_item_falls_back_to_cached_without_context():
    # SWE-bench supports execution, but with no execution_context the unified API
    # must use the cached exact-match path, not error.
    adapter = get_adapter("swebench_verified")
    ref = {"gold_patch": "diff --git a/x b/x\n+fix\n"}
    out = score_item(adapter, "diff --git a/x b/x\n+fix\n", ref)
    assert out.mode is ScoringMode.CACHED
    assert out.reward == 1.0


def test_score_item_uses_execution_when_context_supplied():
    adapter = get_adapter("swebench_verified")
    ref = {"gold_patch": "diff --git a/x b/x\n+fix\n"}

    # An injected executor (what the #18 runner plugs in as) reports the patch
    # resolved regardless of exact-match — proving the execution path is taken.
    calls = {"n": 0}

    def executor(output, reference):
        calls["n"] += 1
        return 1.0

    out = score_item(adapter, "totally different patch", ref, execution_context=executor)
    assert out.mode is ScoringMode.EXECUTION
    assert out.reward == 1.0
    assert calls["n"] == 1
    # Cached path would have scored this 0.0, so execution genuinely won.
    assert score_item(adapter, "totally different patch", ref).reward == 0.0


def test_execution_miss_falls_back_to_cached():
    # If the executor returns None ("could not execute"), the dispatcher must
    # fall back to the cached path rather than reporting None.
    adapter = get_adapter("swebench_verified")
    ref = {"gold_patch": "diff --git a/x b/x\n+fix\n"}
    out = score_item(adapter, "diff --git a/x b/x\n+fix\n", ref, execution_context=lambda o, r: None)
    assert out.mode is ScoringMode.CACHED
    assert out.reward == 1.0


def test_execution_only_adapter_skips_cached():
    """An adapter with only EXECUTION is flagged so a cheap round can skip it."""

    class _ExecOnly(BenchmarkAdapter):
        name = "exec-only"

        def load_tasks(self, split, max_items, seed=0):
            return []

        def build_prompt(self, task):
            return task.prompt

        def score_output(self, output, reference):
            return 0.0

        def task_type(self):
            return TaskType.PATCH

        def serialize_task(self, task):
            return {"task_id": task.task_id}

        def scoring_modes(self):
            return frozenset({ScoringMode.EXECUTION})

        def score_execution(self, output, reference, *, context=None):
            return 1.0 if output == "good" else 0.0

    adapter = _ExecOnly()
    assert requires_execution(adapter)
    # With a context, execution runs.
    assert score_item(adapter, "good", None, execution_context={}).reward == 1.0
    # Without a context, an execution-only adapter still routes to execution
    # (its only path) rather than mis-reporting a cached score.
    out = score_item(adapter, "good", None)
    assert out.mode is ScoringMode.EXECUTION
    assert out.reward == 1.0
