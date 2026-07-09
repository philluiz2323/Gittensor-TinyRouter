"""Built-in adapters for the benchmarks the repo already ships.

These wrap the existing, already-factored pipeline rather than duplicating it:

* task loading delegates to :func:`trinity.orchestration.dataset.load_tasks`;
* scoring delegates to :func:`trinity.orchestration.reward.score_text`, the one
  de-bugged extractor the evaluator, oracle-ceiling diagnostic, and Fugu reward
  all already share;
* the task-type family is derived from the same benchmark frozensets that
  ``reward.py`` dispatches on, so the taxonomy cannot drift from the scorer.

Because every current benchmark ({math500, mmlu, gpqa, livecodebench}) is served
by that shared pipeline, one parametrised :class:`DelegatingBenchmarkAdapter`
covers them all; each is registered under its own name. When a benchmark grows
logic that no longer fits the shared loader (the SWE-bench patch evaluator in
#17/#18, the MMLU-Pro / LiveCodeBench-v6 adapters in #12/#13), it graduates to a
dedicated :class:`~trinity.adapters.base.BenchmarkAdapter` subclass — the
registry seam means that change touches only this module.
"""
from __future__ import annotations

from typing import Any

from trinity.orchestration import reward as _reward
from trinity.orchestration.dataset import load_tasks as _load_tasks
from trinity.types import Task

from .base import BenchmarkAdapter, TaskType
from .registry import register_adapter

__all__ = ["DelegatingBenchmarkAdapter", "register_builtin_adapters"]


def _task_type_for(benchmark: str) -> TaskType:
    """Classify a benchmark using ``reward.py``'s own dispatch tables.

    Reusing the scorer's frozensets (rather than a second hand-kept map) means a
    benchmark can never be classified one way here and scored another way there.
    """
    key = (benchmark or "").strip().lower()
    if key in _reward.MATH_BENCHMARKS:
        return TaskType.MATH
    if key in _reward.CHOICE_BENCHMARKS:
        return TaskType.MCQ
    if key in _reward.CODE_BENCHMARKS:
        return TaskType.CODE
    raise ValueError(
        f"Benchmark {benchmark!r} has no known task type; add it to the "
        f"reward.py dispatch tables or give it a dedicated adapter."
    )


class DelegatingBenchmarkAdapter(BenchmarkAdapter):
    """Adapter that routes every call to the shared dataset/reward pipeline.

    This is the faithful wrapper for benchmarks already served end-to-end by
    ``dataset.load_tasks`` + ``reward.score_text``. Prompts are produced by the
    loaders (which already format MCQ options, boxed-answer instructions, etc.),
    so :meth:`build_prompt` simply returns the task's prompt.
    """

    def __init__(self, name: str):
        self.name = (name or "").strip().lower()
        if not self.name:
            raise ValueError("DelegatingBenchmarkAdapter requires a benchmark name.")
        # Validate up front so a misconfigured registration fails at import, not
        # deep inside an eval run.
        self._task_type = _task_type_for(self.name)

    def load_tasks(
        self,
        split: str,
        max_items: int | None,
        seed: int = 0,
    ) -> list[Task]:
        return _load_tasks(self.name, split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return _reward.score_text(self.name, output, reference)

    def task_type(self) -> TaskType:
        return self._task_type

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": self._task_type.value,
            "meta": dict(task.meta),
        }


#: Benchmarks currently served by the shared pipeline. Mirrors
#: ``dataset.SUPPORTED_BENCHMARKS``; kept explicit so registration is auditable.
_BUILTIN_BENCHMARKS: tuple[str, ...] = ("math500", "mmlu", "gpqa", "livecodebench")


def register_builtin_adapters() -> None:
    """Register a delegating adapter for each built-in benchmark.

    Idempotent-friendly: skips names already registered so re-import (or a test
    that registered its own adapter for a built-in name) does not raise.
    """
    from .registry import is_registered

    for name in _BUILTIN_BENCHMARKS:
        if not is_registered(name):
            register_adapter(name, DelegatingBenchmarkAdapter(name))
