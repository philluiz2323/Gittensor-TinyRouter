"""Offline conformance auditor for the ``BenchmarkAdapter`` contract.

``adapters/base.py`` documents the contract every adapter must honor — a binary
``score_output`` (``{0.0, 1.0}``), a valid :class:`TaskType`, a non-empty set of
:class:`ScoringMode`, a **deterministic** ``load_tasks`` ("repeated calls with identical
arguments must return identical lists"), and a **JSON-safe** ``serialize_task`` (the
portable on-disk item: id, benchmark, prompt, reference, type, meta) — but nothing enforces
it uniformly. ``tests/test_benchmark_registry.py`` spot-checks only the *scoring* invariant,
and only for 3 of the 9 registered adapters (math500 / mmlu / gpqa). The newer
``drop`` / ``bbh`` / ``mmlu_pro`` / ``livecodebench_v6`` / ``swebench_verified`` adapters —
and every future one (open issues #254 / #212 want more) — were added with bespoke per-file
tests and never folded into a shared contract check.

This module audits **every registered adapter** against the contract, offline (the toy
fallback loaders need no network / datasets / torch). It is a **green-today guard**: it
passes on the current suite and fires the moment a new or edited adapter drifts — a
non-binary score, a ``serialize_task`` field that is not JSON-native, a non-deterministic
loader, an invalid ``task_type``, or a missing portable-format key. It is read-only; it
never touches an adapter's data.

The generic contracts (binary output, JSON round-trip, load determinism, valid
type/modes) need no per-adapter knowledge and cover all adapters. The optional
self-consistency contract (a compliant correct answer scores 1.0, a wrong one 0.0) is
driven by :data:`ADAPTER_CASES`, generalizing the registry test's 3-adapter check to the
math/MCQ adapters; code/patch adapters (which need real execution) are audited on the
generic contracts only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from trinity.adapters.base import BenchmarkAdapter, ScoringMode, TaskType
from trinity.types import Task

__all__ = [
    "ContractResult",
    "ConformanceReport",
    "ADAPTER_CASES",
    "audit_adapter",
    "audit_all",
    "render",
]

# Keys the frozen portable item format requires (base.py serialize_task docstring).
_REQUIRED_SERIALIZE_KEYS = ("task_id", "benchmark", "prompt", "reference", "task_type")
# Probe outputs used to assert score_output stays binary on arbitrary inputs.
_BINARY_PROBES = ("", "garbage nonsense output", "Answer: definitely-not-the-answer")

# Verified (compliant-correct, clearly-wrong, reference) self-consistency cases. Only the
# math/MCQ adapters — code/patch grading needs real execution, so those skip this contract.
ADAPTER_CASES: dict[str, tuple[str, str, Any]] = {
    "math500": ("The answer is \\boxed{4}.", "The answer is \\boxed{9}.", "4"),
    "mmlu": ("The answer is B.", "The answer is A.", "B"),
    "gpqa": ("Answer: B", "Answer: C", "B"),
    "drop": ("Answer: 21", "Answer: 99", {"gold_answers": ["21"]}),
    "bbh": ("Answer: (B)", "Answer: (A)", {"answer": "(B)", "answer_type": "multiple_choice"}),
    "mmlu_pro": ("Answer: B", "Answer: A", "B"),
}


@dataclass(frozen=True)
class ContractResult:
    """Outcome of one contract check for one adapter."""

    adapter: str
    contract: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {"adapter": self.adapter, "contract": self.contract, "ok": self.ok,
                "detail": self.detail}


@dataclass
class ConformanceReport:
    """Every contract result across the audited adapters (all ``ok`` == conformant)."""

    results: list[ContractResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff every contract passed."""
        return all(r.ok for r in self.results)

    @property
    def adapters(self) -> list[str]:
        """The distinct adapters audited, in first-seen order."""
        out: list[str] = []
        for r in self.results:
            if r.adapter not in out:
                out.append(r.adapter)
        return out

    def failures(self) -> list[ContractResult]:
        """Only the failed contract results."""
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "ok": self.ok,
            "n_adapters": len(self.adapters),
            "n_contracts": len(self.results),
            "n_failures": len(self.failures()),
            "results": [r.to_dict() for r in self.results],
        }


def _check(adapter: str, contract: str, fn: Any) -> ContractResult:
    """Run one contract ``fn`` (returns ``(ok, detail)``); a raise is itself a failure."""
    try:
        ok, detail = fn()
    except Exception as exc:  # a contract that crashes the auditor is a contract failure
        return ContractResult(adapter, contract, False, f"raised {type(exc).__name__}: {exc}")
    return ContractResult(adapter, contract, ok, detail)


def audit_adapter(
    name: str,
    adapter: BenchmarkAdapter,
    *,
    case: tuple[str, str, Any] | None = None,
    n_tasks: int = 3,
) -> list[ContractResult]:
    """Audit one adapter against the ``BenchmarkAdapter`` contract; return per-contract results.

    Runs the generic contracts (valid ``task_type`` / ``scoring_modes``, non-empty +
    ``Task``-typed ``load_tasks``, load determinism, JSON-safe ``serialize_task`` with the
    required keys, binary ``score_output``) and — when ``case`` is given — a self-consistency
    check that the compliant correct answer scores ``1.0`` and a wrong one ``0.0``.
    """
    results: list[ContractResult] = []

    def _task_type() -> tuple[bool, str]:
        tt = adapter.task_type()
        return isinstance(tt, TaskType), f"task_type={tt!r}"

    def _scoring_modes() -> tuple[bool, str]:
        modes = adapter.scoring_modes()
        ok = bool(modes) and all(isinstance(m, ScoringMode) for m in modes)
        return ok, f"scoring_modes={sorted(str(m) for m in modes)}"

    tasks = adapter.load_tasks("test", n_tasks, 0)

    def _load() -> tuple[bool, str]:
        if not tasks:
            return False, "load_tasks returned no tasks"
        non_task = [type(t).__name__ for t in tasks if not isinstance(t, Task)]
        return (not non_task), (f"non-Task items: {non_task}" if non_task else f"{len(tasks)} tasks")

    def _determinism() -> tuple[bool, str]:
        again = adapter.load_tasks("test", n_tasks, 0)
        a = [adapter.serialize_task(t) for t in tasks]
        b = [adapter.serialize_task(t) for t in again]
        return a == b, "identical reloads" if a == b else "load_tasks is non-deterministic"

    def _serialize() -> tuple[bool, str]:
        for t in tasks:
            d = adapter.serialize_task(t)
            if not isinstance(d, dict):
                return False, f"serialize_task returned {type(d).__name__}, not dict"
            missing = [k for k in _REQUIRED_SERIALIZE_KEYS if k not in d]
            if missing:
                return False, f"missing required key(s): {missing}"
            try:
                if json.loads(json.dumps(d)) != d:
                    return False, "serialize_task does not round-trip through JSON"
            except (TypeError, ValueError) as exc:
                return False, f"serialize_task is not JSON-safe: {exc}"
        return True, "JSON-safe, keys present"

    def _binary() -> tuple[bool, str]:
        ref = tasks[0].answer
        for probe in _BINARY_PROBES:
            s = adapter.score_output(probe, ref)
            if s not in (0.0, 1.0):
                return False, f"score_output({probe!r}) = {s!r} (not binary {{0.0, 1.0}})"
        return True, "binary on all probes"

    results.append(_check(name, "task_type", _task_type))
    results.append(_check(name, "scoring_modes", _scoring_modes))
    results.append(_check(name, "load_tasks", _load))
    results.append(_check(name, "load_determinism", _determinism))
    results.append(_check(name, "serialize_task", _serialize))
    results.append(_check(name, "score_output_binary", _binary))

    if case is not None:
        correct, wrong, ref = case

        def _self_consistency() -> tuple[bool, str]:
            sc = adapter.score_output(correct, ref)
            sw = adapter.score_output(wrong, ref)
            ok = sc == 1.0 and sw == 0.0
            return ok, f"correct={sc}, wrong={sw}"

        results.append(_check(name, "self_consistency", _self_consistency))
    return results


def _ensure_registered() -> None:
    """Register the full built-in adapter suite (idempotent-friendly, best-effort)."""
    import trinity.adapters as A

    for reg in ("register_builtin_adapters", "register_livecodebench_v6_adapter",
                "register_bbh_adapter", "register_drop_adapter",
                "register_mmlu_pro_adapter", "register_swebench_adapter"):
        fn = getattr(A, reg, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # already registered / optional adapter unavailable
                pass


def audit_all(
    *,
    cases: dict[str, tuple[str, str, Any]] | None = None,
    n_tasks: int = 3,
) -> ConformanceReport:
    """Audit every registered adapter; ``cases`` supplies per-adapter self-consistency inputs."""
    import trinity.adapters as A

    _ensure_registered()
    case_map = ADAPTER_CASES if cases is None else cases
    report = ConformanceReport()
    for name in A.available_adapters():
        adapter = A.get_adapter(name)
        report.results.extend(
            audit_adapter(name, adapter, case=case_map.get(name), n_tasks=n_tasks)
        )
    return report


def render(report: ConformanceReport) -> str:
    """Markdown: per-adapter contract grid + any failures."""
    out = ["# Benchmark adapter conformance\n"]
    if not report.results:
        return "".join(out) + "\n_(no adapters registered)_\n"
    out.append(f"{len(report.adapters)} adapters, {len(report.results)} contract checks — "
               f"{'ALL PASS' if report.ok else str(len(report.failures())) + ' FAILURE(S)'}\n")
    for name in report.adapters:
        rows = [r for r in report.results if r.adapter == name]
        marks = " ".join(f"{r.contract}{'✓' if r.ok else '✗'}" for r in rows)
        out.append(f"- **{name}**: {marks}")
    if not report.ok:
        out.append("\nFailures:")
        out.extend(f"  - {r.adapter}.{r.contract}: {r.detail}" for r in report.failures())
    return "\n".join(out) + "\n"
