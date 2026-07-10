"""The :class:`BenchmarkAdapter` interface and task-type taxonomy.

A *benchmark adapter* is the single seam through which the evaluator touches a
benchmark. The core evaluation pipeline (``trinity.eval``) never names a
benchmark or branches on it; it asks the registry for an adapter and drives the
adapter's methods. Adding a new benchmark is therefore a matter of implementing
this interface once and registering it (see :mod:`trinity.adapters.registry`),
with no edits to the shared evaluator, transcript capture, or aggregation.

The interface is intentionally the six-method surface called out in the design
issue (#9):

``load_tasks(split, max_items, seed)``
    Produce the deterministic :class:`~trinity.types.Task` list for a split.
``build_prompt(task)``
    Render the exact text handed to a pool model for one task.
``score_output(output, reference)``
    Binary ``{0.0, 1.0}`` correctness of a model answer against the reference.
``task_type()``
    The coarse family of the benchmark (drives shared formatting/routing).
``serialize_task(task)``
    A JSON-safe dict for the frozen hidden-benchmark item format.
``cache_baselines(task, pool)`` *(optional)*
    Pre-compute per-model baseline answers/scores; default is a no-op.

Concrete adapters live in :mod:`trinity.adapters.builtin`. This module imports
nothing from torch and has no network dependency, so it loads on the local dev
box exactly as it does on the GPU host.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trinity.types import Task, Trajectory


class TaskType(str, Enum):
    """The coarse family a benchmark belongs to.

    The evaluator uses this (not the benchmark name) whenever it needs to make a
    format-level decision — e.g. how to phrase a Worker prompt or which shared
    extractor a reward path expects. New benchmarks reuse an existing type
    rather than introducing another name the core has to know about.
    """

    MATH = "math"          # free-form answer, boxed / last-number extraction
    MCQ = "mcq"            # single multiple-choice letter (A-D...)
    CODE = "code"          # code executed against tests (pass@1)
    PATCH = "patch"        # unified diff applied + test suite (SWE-bench)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ScoringMode(str, Enum):
    """How a benchmark's answers get graded (issue #16).

    Separates two evaluation costs so the pipeline can run each without mixing
    concerns:

    ``CACHED``
        Cheap scoring of a stored answer against the reference — a string/letter
        compare (MMLU, math) or a local sandboxed code run (LiveCodeBench). Runs
        in the ordinary scoring round.
    ``EXECUTION``
        Expensive live execution beyond scoring a stored answer — checking out a
        repository and running its test harness (SWE-bench). Optional, and run
        through a dedicated executor rather than the cached round.

    An adapter declares one or both via :meth:`BenchmarkAdapter.scoring_modes`.
    """

    CACHED = "cached"
    EXECUTION = "execution"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class BenchmarkAdapter(ABC):
    """One benchmark, behind a uniform interface.

    Subclasses implement the abstract methods below. :meth:`cache_baselines` is
    concrete (a no-op) so adapters that do not pre-compute baselines need not
    override it.
    """

    #: The canonical benchmark identifier this adapter serves, e.g. ``"math500"``.
    #: Must match the key it is registered under and the ``Task.benchmark`` field
    #: its tasks carry, so downstream dispatch stays consistent.
    name: str

    @abstractmethod
    def load_tasks(
        self,
        split: str,
        max_items: int | None,
        seed: int = 0,
    ) -> list["Task"]:
        """Return the deterministic task list for ``split``.

        Repeated calls with identical arguments must return identical lists so
        eval splits and training minibatches are reproducible.
        """

    @abstractmethod
    def build_prompt(self, task: "Task") -> str:
        """Return the exact prompt text presented to a pool model for ``task``."""

    @abstractmethod
    def score_output(self, output: str, reference: Any) -> float:
        """Return the binary reward ``{0.0, 1.0}`` for ``output`` vs ``reference``.

        ``reference`` is whatever this benchmark stores in ``Task.answer`` (a gold
        string for math/MCQ, a test spec for code), so the caller can score
        without knowing the benchmark's internal answer representation.
        """

    @abstractmethod
    def task_type(self) -> TaskType:
        """Return the :class:`TaskType` family for this benchmark."""

    @abstractmethod
    def serialize_task(self, task: "Task") -> dict[str, Any]:
        """Return a JSON-safe dict for the frozen hidden-benchmark item format.

        The dict is the portable, on-disk representation of one task. It must be
        round-trippable to the extent the frozen protocol needs (id, prompt,
        reference, type, meta) and must contain only JSON-native values.
        """

    def score_trajectory(self, traj: "Trajectory") -> float:
        """Return the binary reward for a full multi-turn trajectory.

        This is the entry point the routed evaluation path (TRINITY and the
        random-routing baseline) uses, so it must stay consistent with
        :meth:`score_output`: the default picks the *committed answer* across
        turns (the same most-recent-extractable-answer rule the evaluator uses)
        and then delegates to :meth:`score_output`. An adapter that customises
        ``score_output`` therefore gets that behaviour on both the single-turn
        and multi-turn paths for free; one that needs to inspect intermediate
        turns (e.g. a patch adapter) overrides this method.
        """
        from trinity.orchestration.reward import committed_answer

        candidate = committed_answer(self.name, traj)
        return self.score_output(candidate, traj.task.answer)

    # -- Cached vs execution scoring (issue #16) ---------------------------- #
    def scoring_modes(self) -> frozenset[ScoringMode]:
        """Return the scoring paths this benchmark supports.

        Defaults to ``{CACHED}`` — the common case (MMLU, math, code) is scored
        cheaply from a stored answer. A benchmark that also needs a heavyweight
        run (e.g. SWE-bench) overrides this to add :data:`ScoringMode.EXECUTION`.
        The top-level dispatcher (:func:`trinity.adapters.scoring.score_item`)
        reads this to route cheap and expensive benchmarks through one API.
        """
        return frozenset({ScoringMode.CACHED})

    def score_cached(self, output: str, reference: Any) -> float:
        """Cheap cached scoring: grade a stored answer against ``reference``.

        Defaults to :meth:`score_output`, so every existing adapter keeps its
        current cached behaviour unchanged. This is the path the ordinary scoring
        round uses.
        """
        return self.score_output(output, reference)

    def score_execution(
        self, output: str, reference: Any, *, context: Any = None
    ) -> float | None:
        """Expensive execution scoring, or ``None`` if unavailable.

        Default is ``None`` — most benchmarks do not need live execution. An
        adapter declaring :data:`ScoringMode.EXECUTION` implements this to grade
        via a heavyweight run; ``context`` carries whatever that run needs (e.g.
        a prepared work-tree or an executor callable), supplied by the caller.
        Returning ``None`` signals "could not execute" so the dispatcher falls
        back to :meth:`score_cached` instead of guessing.
        """
        return None

    def cache_baselines(self, task: "Task", pool: Any) -> dict[str, Any] | None:
        """Optionally pre-compute per-model baseline answers/scores for ``task``.

        Default implementation is a no-op returning ``None``. Adapters whose
        hidden-benchmark items cache model baselines (so scoring a submission
        avoids re-querying the pool) override this. ``pool`` is the model pool
        the concrete adapter expects; the base class does not constrain it.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={getattr(self, 'name', '?')!r})"
