"""LiveCodeBench v6 — an explicit, version-pinned code-generation adapter.

The shared pipeline serves LiveCodeBench through the delegating adapter, which
maps a logical split onto a release *implicitly* (V1 train / V6 eval; see
:func:`trinity.orchestration.dataset._lcb_version_for_split`). Issue #13 asks for
the eval release to be a first-class, explicitly-named benchmark so that:

* ``livecodebench_v6`` is selectable directly from the evaluator
  (``get_adapter("livecodebench_v6")``);
* the pinned dataset id and release are visible in the adapter's metadata and in
  every serialised task; and
* the hidden-benchmark builder always freezes the same release rather than
  relying on an implicit split string.

Only *version selection* lives here. Prompt rendering and scoring reuse the
shared, already-de-bugged LiveCodeBench path — the sandboxed pass@1 executor in
:func:`trinity.orchestration.reward.score_text` — so this adapter never forks the
scorer or the loader. It pins which release the tasks come from and surfaces that
release as metadata; the underlying :class:`~trinity.types.Task` keeps the base
``livecodebench`` benchmark key so existing reward dispatch keeps working
unchanged. Like ``reward`` and ``dataset``, this module has no torch dependency.
"""
from __future__ import annotations

from typing import Any

from trinity.orchestration import reward as _reward
from trinity.orchestration.dataset import load_tasks as _load_tasks
from trinity.types import Task

from .base import BenchmarkAdapter, TaskType

__all__ = ["LiveCodeBenchV6Adapter"]


class LiveCodeBenchV6Adapter(BenchmarkAdapter):
    """First-class adapter for the LiveCodeBench **v6** code-generation release.

    ``load_tasks`` and ``score_output`` delegate to the shared LiveCodeBench
    pipeline (``dataset.load_tasks`` + ``reward.score_text``) under the base
    ``livecodebench`` key, so test-case parsing and pass@1 scoring stay identical
    to the de-bugged path. This adapter's job is to (a) freeze the release
    explicitly and (b) make that release visible as metadata.
    """

    #: Registry / serialised-item identity (matches the registered name).
    name = "livecodebench_v6"

    #: Pinned HuggingFace dataset id and release configs.
    dataset_id = "livecodebench/code_generation_lite"
    eval_version = "release_v6"
    train_version = "release_v1"

    #: The base benchmark whose shared loader/scorer this adapter reuses. Tasks
    #: carry this key (not ``name``) so the existing reward dispatch — which is
    #: keyed on ``Task.benchmark`` — routes them to the LiveCodeBench code scorer.
    base_benchmark = "livecodebench"

    def resolve_version(self, split: str) -> str:
        """Return the release frozen for ``split``, keeping train/eval explicit.

        An explicit training split resolves to ``release_v1``; anything else
        (test/eval/blank) freezes to the pinned ``release_v6`` eval release, so a
        caller cannot silently score against a different version.

        Args:
            split: Logical split string, e.g. ``"train"`` or ``"test"``.

        Returns:
            The resolved LiveCodeBench release config name.
        """
        s = (split or "").strip().lower()
        if s in {"train", "v1", "release_v1"}:
            return self.train_version
        return self.eval_version

    def load_tasks(
        self,
        split: str,
        max_items: int | None,
        seed: int = 0,
    ) -> list[Task]:
        """Load LiveCodeBench tasks for ``split`` pinned to the resolved release.

        Reuses the shared loader (with its offline toy fallback), then stamps the
        resolved release onto each task's ``meta`` so the frozen item and any
        downstream hidden-benchmark builder record exactly which release was used.
        """
        version = self.resolve_version(split)
        tasks = _load_tasks(self.base_benchmark, version, max_items, seed=seed)
        for task in tasks:
            task.meta.setdefault("dataset_id", self.dataset_id)
            task.meta["dataset_version"] = version
            task.meta["adapter"] = self.name
        return tasks

    def build_prompt(self, task: Task) -> str:
        """Return the task prompt (the loader already renders the problem text)."""
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        """Return binary pass@1 via the shared sandboxed LiveCodeBench scorer."""
        return _reward.score_text(self.base_benchmark, output, reference)

    def task_type(self) -> TaskType:
        """Return :attr:`TaskType.CODE` — v6 is a code-generation benchmark."""
        return TaskType.CODE

    def serialize_task(self, task: Task) -> dict[str, Any]:
        """Return a JSON-safe frozen item that records the pinned release."""
        return {
            "task_id": task.task_id,
            "benchmark": self.name,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": TaskType.CODE.value,
            "dataset_id": self.dataset_id,
            "dataset_version": task.meta.get("dataset_version", self.eval_version),
            "meta": dict(task.meta),
        }

    def metadata(self) -> dict[str, Any]:
        """Return the adapter's pinned dataset metadata (release visibility)."""
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "eval_version": self.eval_version,
            "train_version": self.train_version,
            "task_type": TaskType.CODE.value,
        }
