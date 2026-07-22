"""SPEC definition-of-done roll-up — does the replication actually meet its own bar?

Why this exists
---------------
``docs/SPEC.md`` states the success criterion for the whole replication in one sentence:

    **Definition of done for the replication:** R1–R4 and R8 hold on at least 2 of our
    chosen in-distribution tasks; the trained coordinator runs end-to-end within the
    atomic-eval budget on one H200; the optimizer drives ``J(θ)`` upward over iterations.

Every ingredient exists, but **nothing computes that sentence**:

* :mod:`trinity.analysis.significance` assesses R1/R2/R4 with CIs, but **per eval file** —
  it has no notion of "how many tasks does this hold on".
* :mod:`trinity.analysis.ensemble` gives the R3 multi-agent baseline, per benchmark.
* :mod:`trinity.analysis.convergence` covers the *optimizer* clause
  (``dod_drives_J_upward``) and the R8 ordering data, but not R1–R4.
* ``scripts/results_table.py`` prints a multi-task summary, but its verdict is an
  **average-based** one ("TRINITY avg > best fixed single avg") — a different rule from
  "holds on at least 2 tasks", and it omits R8 entirely.

So the headline question — *is the replication done?* — has no answer in code. This module
composes the existing verdicts (it never re-derives them: R1/R2/R4 come from
``significance.assess_invariants``, R8 from ``convergence.analyze_runs``' rankings) into the
per-task × per-invariant matrix the SPEC actually asks for, applies the ≥2-task rule, and
emits one combined PASS/FAIL.

Per-task reading of each invariant (stated explicitly so the verdict cannot be misread):

* **R1** — TRINITY's mean > the best single model's mean on that task.
* **R2** — TRINITY beats **every** single model on that task (strictly stronger than R1;
  R1 compares against the max, R2 requires it against all, which is the same test only when
  the comparator really is the maximum — computed independently here).
* **R3** — TRINITY > the realizable multi-agent (plurality) baseline on that task.
* **R4** — TRINITY > random routing on that task.
* **R8** — the SPEC optimizer order holds on that task, i.e. ``sep_cmaes`` tops that
  benchmark's ranking.

An invariant with no evidence for a task is ``None`` ("not measured") and never counts
toward the ≥2 tally — absence is reported, never silently scored as a pass. The SPEC's
third clause (end-to-end within the atomic-eval budget on one H200) is a *hardware runtime*
property and is not offline-checkable; it is surfaced as an explicit caveat rather than
quietly dropped.

Pure / offline — stdlib + numpy over artifacts already on disk. No torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from trinity.analysis.significance import assess_invariants

__all__ = [
    "DOD_INVARIANTS",
    "DOD_MIN_TASKS",
    "TaskInvariants",
    "DoDVerdict",
    "assess_task",
    "assess",
    "render",
]

#: The invariants the SPEC's definition of done requires (R1–R4 and R8).
DOD_INVARIANTS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R8")

#: "...hold on at least 2 of our chosen in-distribution tasks".
DOD_MIN_TASKS: int = 2

#: The SPEC clause this module cannot check offline (hardware/runtime, not an artifact).
RUNTIME_CLAUSE = "trained coordinator runs end-to-end within the atomic-eval budget on one H200"


def _mean(xs: Sequence[float] | None) -> Optional[float]:
    if xs is None or len(xs) == 0:
        return None
    return float(np.mean(np.asarray(xs, dtype=float)))


@dataclass(frozen=True)
class TaskInvariants:
    """Which DoD invariants hold on ONE in-distribution task.

    ``holds`` maps each of :data:`DOD_INVARIANTS` to ``True`` / ``False`` / ``None``
    (no evidence). ``detail`` carries a one-line reason per invariant for the report.
    """

    benchmark: str
    holds: dict[str, Optional[bool]] = field(default_factory=dict)
    detail: dict[str, str] = field(default_factory=dict)

    @property
    def measured(self) -> list[str]:
        return [k for k, v in self.holds.items() if v is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "holds": dict(self.holds),
            "detail": dict(self.detail),
        }


def assess_task(
    benchmark: str,
    per_item: Mapping[str, Any] | None = None,
    *,
    ensemble_accuracy: float | None = None,
    r8_ranking: Sequence[Mapping[str, Any]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> TaskInvariants:
    """Evaluate R1–R4 and R8 for a single task.

    Args:
        benchmark: Task name.
        per_item: The eval ``per_item`` block (``TRINITY`` / ``single::<model>`` /
            ``random_routing`` 0-1 vectors). Drives R1, R2 and R4. R1/R4 reuse
            ``significance.assess_invariants`` so their verdict cannot drift from the
            paired-CI machinery; R2's "beats every single" is computed directly.
        ensemble_accuracy: The realizable multi-agent (plurality) baseline accuracy on this
            task, e.g. ``ensemble.EnsembleSummary.ensemble_accuracy``. Drives R3.
        r8_ranking: This benchmark's entry from ``convergence.analyze_runs()["rankings"]``
            — trainers ordered strongest-first. R8 holds when ``sep_cmaes`` is first.
        n_boot, seed, alpha: Passed through to the paired-significance assessment.

    Returns:
        A :class:`TaskInvariants` with an entry for every invariant (``None`` when the
        evidence for it was not supplied).
    """
    holds: dict[str, Optional[bool]] = {k: None for k in DOD_INVARIANTS}
    detail: dict[str, str] = {}

    tri = per_item.get("TRINITY") if per_item else None
    tri_mean = _mean(tri)

    if per_item and tri_mean is not None:
        singles = {
            k[len("single::"):]: v for k, v in per_item.items() if k.startswith("single::")
        }
        # R1 / R4 — reuse the canonical paired-significance verdicts.
        sig = assess_invariants(
            per_item, benchmark=benchmark, n_boot=n_boot, seed=seed, alpha=alpha
        )
        for cmp_ in sig.comparisons:
            if cmp_.name_b.startswith("best single"):
                holds["R1"] = cmp_.diff > 0.0
                detail["R1"] = (
                    f"TRINITY {cmp_.mean_a:.3f} vs {cmp_.name_b} {cmp_.mean_b:.3f} "
                    f"(diff {cmp_.diff:+.3f}, 95% CI [{cmp_.ci_lo:+.3f}, {cmp_.ci_hi:+.3f}]"
                    f"{', significant' if cmp_.significant else ''})"
                )
            elif cmp_.name_b == "random routing":
                holds["R4"] = cmp_.diff > 0.0
                detail["R4"] = (
                    f"TRINITY {cmp_.mean_a:.3f} vs random {cmp_.mean_b:.3f} "
                    f"(diff {cmp_.diff:+.3f}, 95% CI [{cmp_.ci_lo:+.3f}, {cmp_.ci_hi:+.3f}]"
                    f"{', significant' if cmp_.significant else ''})"
                )

        # R2 — strictly beats EVERY single model, not merely the best-scoring one.
        if singles:
            means = {m: _mean(v) for m, v in singles.items()}
            usable = {m: v for m, v in means.items() if v is not None}
            if usable:
                losers = [m for m, v in usable.items() if tri_mean <= v]
                holds["R2"] = not losers
                detail["R2"] = (
                    f"TRINITY {tri_mean:.3f} beats all {len(usable)} single model(s)"
                    if not losers
                    else f"TRINITY {tri_mean:.3f} does not beat: "
                         + ", ".join(f"{m} ({usable[m]:.3f})" for m in sorted(losers))
                )

    # R3 — vs the realizable multi-agent (plurality) baseline.
    if ensemble_accuracy is not None and tri_mean is not None:
        holds["R3"] = tri_mean > float(ensemble_accuracy)
        detail["R3"] = (
            f"TRINITY {tri_mean:.3f} vs plurality ensemble {float(ensemble_accuracy):.3f} "
            f"(diff {tri_mean - float(ensemble_accuracy):+.3f})"
        )

    # R8 — the SPEC optimizer order on this task: sep_cmaes must rank first.
    if r8_ranking:
        top = str((r8_ranking[0] or {}).get("trainer", ""))
        holds["R8"] = top == "sep_cmaes"
        order = " > ".join(str((r or {}).get("trainer", "?")) for r in r8_ranking)
        detail["R8"] = f"observed {order} (SPEC wants sep_cmaes first)"

    for k in DOD_INVARIANTS:
        detail.setdefault(k, "not measured (no evidence supplied)")
    return TaskInvariants(benchmark=benchmark, holds=holds, detail=detail)


@dataclass(frozen=True)
class DoDVerdict:
    """The SPEC definition-of-done verdict over all in-distribution tasks."""

    tasks: list[TaskInvariants]
    min_tasks: int
    task_counts: dict[str, int]                 # invariant -> #tasks it holds on
    measured_counts: dict[str, int]             # invariant -> #tasks with evidence
    invariants_met: dict[str, bool]             # count >= min_tasks
    drives_J_upward: Optional[bool]             # convergence's optimizer clause
    passed: bool

    @property
    def unmet(self) -> list[str]:
        return [k for k in DOD_INVARIANTS if not self.invariants_met.get(k)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "min_tasks": self.min_tasks,
            "task_counts": dict(self.task_counts),
            "measured_counts": dict(self.measured_counts),
            "invariants_met": dict(self.invariants_met),
            "drives_J_upward": self.drives_J_upward,
            "passed": self.passed,
            "unmet": self.unmet,
            "not_offline_checkable": [RUNTIME_CLAUSE],
        }


def assess(
    tasks: Sequence[TaskInvariants],
    *,
    drives_J_upward: bool | None = None,
    min_tasks: int = DOD_MIN_TASKS,
) -> DoDVerdict:
    """Apply the SPEC rule: every invariant must hold on ``min_tasks`` in-dist tasks.

    Args:
        tasks: One :class:`TaskInvariants` per in-distribution task.
        drives_J_upward: ``convergence.analyze_runs()["dod_drives_J_upward"]`` — the SPEC's
            optimizer clause. ``None`` when no run artifacts were supplied, which cannot
            pass (the clause is required, so unmeasured is not satisfied).
        min_tasks: The SPEC's "at least 2".

    Returns:
        The combined :class:`DoDVerdict`. ``passed`` requires every invariant to clear
        ``min_tasks`` AND the optimizer clause to be ``True``.
    """
    task_counts = {
        k: sum(1 for t in tasks if t.holds.get(k) is True) for k in DOD_INVARIANTS
    }
    measured_counts = {
        k: sum(1 for t in tasks if t.holds.get(k) is not None) for k in DOD_INVARIANTS
    }
    invariants_met = {k: task_counts[k] >= min_tasks for k in DOD_INVARIANTS}
    passed = all(invariants_met.values()) and drives_J_upward is True
    return DoDVerdict(
        tasks=list(tasks),
        min_tasks=min_tasks,
        task_counts=task_counts,
        measured_counts=measured_counts,
        invariants_met=invariants_met,
        drives_J_upward=drives_J_upward,
        passed=passed,
    )


def _cell(v: Optional[bool]) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def render(verdict: DoDVerdict) -> str:
    """Markdown: the per-task invariant matrix, the ≥N-task tally, and the verdict."""
    out = ["# SPEC definition of done\n"]
    out.append(f"_R1–R4 and R8 must hold on at least {verdict.min_tasks} in-distribution "
               "tasks; the optimizer must drive J(θ) upward._\n")

    if not verdict.tasks:
        return "".join(out) + "\n_(no tasks assessed)_\n"

    out.append("| task | " + " | ".join(DOD_INVARIANTS) + " |")
    out.append("|---" * (len(DOD_INVARIANTS) + 1) + "|")
    for t in verdict.tasks:
        out.append(f"| {t.benchmark} | "
                   + " | ".join(_cell(t.holds.get(k)) for k in DOD_INVARIANTS) + " |")
    out.append("| **holds on** | "
               + " | ".join(f"**{verdict.task_counts[k]}/{verdict.measured_counts[k]}**"
                            for k in DOD_INVARIANTS) + " |")

    out.append("")
    for k in DOD_INVARIANTS:
        mark = "✓" if verdict.invariants_met[k] else "✗"
        out.append(f"- {mark} **{k}** holds on {verdict.task_counts[k]} task(s) "
                   f"(need {verdict.min_tasks}; measured on {verdict.measured_counts[k]})")

    j = verdict.drives_J_upward
    j_s = "✓ yes" if j is True else ("✗ no" if j is False else "— not measured")
    out.append(f"- {j_s} — optimizer drives J(θ) upward")

    out.append(f"\n**Verdict: {'PASS' if verdict.passed else 'NOT MET'}**"
               + ("." if verdict.passed else f" — unmet: {', '.join(verdict.unmet) or 'J(θ) clause'}."))
    out.append(f"\n_Not offline-checkable (reported, not scored): {RUNTIME_CLAUSE}._")
    return "\n".join(out) + "\n"
