"""Offline R8 check: does the optimizer ranking sep-CMA-ES > SFT > RS > REINFORCE hold?

``docs/SPEC.md`` §1.3 invariant **R8** — *"sep-CMA-ES > SFT > RS > REINFORCE on all 4
tasks"* (Table 4) — is a replication requirement the SPEC itself flags as *"a
hypothesis to test, not a given"* (the block-ε-separability that justifies sep-CMA-ES
was measured on the paper's 7-agent representation, not our 3-model pool). Yet nothing
in ``src/`` verifies the claim. R8 is the whole reason the outer loop is a
derivative-free ES rather than SFT or a policy gradient: if RS or REINFORCE matched
sep-CMA-ES, the expensive evolutionary search would not be earning its keep.

This reads, per task, the final fitness of each optimizer and reports whether the
observed best→worst order matches the expected chain sep-CMA-ES > SFT > RS > REINFORCE
(strictly, within a tolerance), which adjacent pairs invert it, and whether R8 holds on
every scored task. Optimizers absent from a task are simply skipped — the chain is
checked among those present, in expected order.

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "EXPECTED_ORDER",
    "OptimizerRanking",
    "canonical_optimizer",
    "analyze_task",
    "analyze_tasks",
    "render",
]

#: The R8 optimizer chain, best → worst (docs/SPEC.md §1.3, Table 4).
EXPECTED_ORDER: tuple[str, ...] = ("sep_cmaes", "sft", "rs", "reinforce")

_ALIASES: dict[str, str] = {
    "sep_cmaes": "sep_cmaes", "sepcmaes": "sep_cmaes", "sep-cmaes": "sep_cmaes",
    "sep-cma-es": "sep_cmaes", "sepcma": "sep_cmaes", "cma": "sep_cmaes",
    "cmaes": "sep_cmaes", "cma-es": "sep_cmaes", "cma_es": "sep_cmaes",
    "sft": "sft",
    "rs": "rs", "randomsearch": "rs", "random_search": "rs", "random-search": "rs",
    "reinforce": "reinforce", "policy_gradient": "reinforce", "pg": "reinforce",
}

_TOL = 1e-9
_RANK = {name: i for i, name in enumerate(EXPECTED_ORDER)}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def canonical_optimizer(name: Any) -> str | None:
    """Map an optimizer label to its canonical R8 name, or ``None`` if unknown.

    Case- and separator-insensitive: ``"Sep-CMA-ES"``, ``"random search"``,
    ``"policy_gradient"`` all resolve. Returns ``None`` for a label outside the
    ``sep_cmaes`` / ``sft`` / ``rs`` / ``reinforce`` family so a stray key never
    silently joins the chain.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    return _ALIASES.get(key.replace("_", ""))


@dataclass(frozen=True)
class OptimizerRanking:
    """R8 diagnostics for one task's optimizer→final-fitness scores.

    ``observed`` is the optimizers present, ordered by measured fitness (best first).
    ``expected`` is those same optimizers in the SPEC's expected order. ``inversions``
    lists the adjacent expected-order pairs ``(better, worse)`` where the expected-
    better optimizer did **not** strictly exceed the next one. ``holds`` is the R8
    verdict for this task: at least two optimizers present and no inversion.
    """

    task: str
    scores: dict[str, float]
    observed: list[str]
    expected: list[str]
    inversions: list[tuple[str, str]]
    holds: bool

    @property
    def n_present(self) -> int:
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "task": self.task,
            "scores": dict(self.scores),
            "observed": list(self.observed),
            "expected": list(self.expected),
            "inversions": [list(p) for p in self.inversions],
            "n_present": self.n_present,
            "holds": self.holds,
        }


def analyze_task(scores: Mapping[str, Any], *, task: str = "?", tol: float = _TOL) -> OptimizerRanking:
    """Check the R8 optimizer ordering for one task.

    Args:
        scores: ``{optimizer: final_fitness}``. Keys are canonicalized (see
            :func:`canonical_optimizer`); unknown or non-numeric entries are dropped.
            A later duplicate of the same canonical optimizer overwrites an earlier one.
        task: Name for the report row.
        tol: An expected-better optimizer must exceed the next by more than ``tol`` to
            avoid counting a tie as an ordering pass.

    Returns:
        An :class:`OptimizerRanking`. With fewer than two recognized optimizers the
        chain cannot be judged, so ``holds`` is ``False``.
    """
    clean: dict[str, float] = {}
    for name, value in scores.items():
        canon = canonical_optimizer(name)
        if canon is not None and _is_num(value):
            clean[canon] = float(value)

    present_expected = [o for o in EXPECTED_ORDER if o in clean]
    observed = sorted(clean, key=lambda o: (-clean[o], _RANK[o]))
    inversions: list[tuple[str, str]] = []
    for better, worse in zip(present_expected, present_expected[1:]):
        if not (clean[better] - clean[worse] > tol):
            inversions.append((better, worse))
    holds = len(clean) >= 2 and not inversions
    return OptimizerRanking(
        task=task, scores=clean, observed=observed,
        expected=present_expected, inversions=inversions, holds=holds,
    )


def analyze_tasks(tasks: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """Per-task R8 rankings plus the across-tasks verdict.

    Args:
        tasks: ``{task: {optimizer: final_fitness}}``.
        tol: Ordering tolerance (see :func:`analyze_task`).

    Returns:
        ``{"per_task": [OptimizerRanking.to_dict, ...], "r8_holds": bool,
           "n_tasks_scored": int, "violations": [task, ...]}``. ``r8_holds`` is True
        iff at least one task was scorable (>= 2 optimizers) and **every** scorable
        task holds — matching the SPEC's "on all 4 tasks".
    """
    results = [analyze_task(s, task=str(t), tol=tol) for t, s in sorted(tasks.items())]
    scored = [r for r in results if r.n_present >= 2]
    violations = [r.task for r in scored if not r.holds]
    return {
        "per_task": [r.to_dict() for r in results],
        "r8_holds": bool(scored) and not violations,
        "n_tasks_scored": len(scored),
        "violations": violations,
    }


def render(tasks: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the per-task R8 ordering and the across-tasks verdict."""
    report = analyze_tasks(tasks, tol=tol)
    lines = ["| task | observed order (best->worst) | R8 |", "|---|---|---|"]
    for r in report["per_task"]:
        order = " > ".join(r["observed"]) or "-"
        if r["n_present"] < 2:
            flag = "n/a"
        elif r["holds"]:
            flag = "ok"
        else:
            flag = "inverts " + ", ".join(f"{b}<={w}" for b, w in r["inversions"])
        lines.append(f"| {r['task']} | {order} | {flag} |")
    verdict = "HOLDS" if report["r8_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R8 (sep-CMA-ES > SFT > RS > REINFORCE): {verdict} "
        f"({report['n_tasks_scored']} task(s) scored)"
    )
    if report["violations"]:
        lines.append(f"violations: {', '.join(report['violations'])}")
    return "\n".join(lines)
