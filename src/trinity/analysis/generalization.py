"""Offline eval-vs-audit generalization (overfit-gap) report.

``scripts/pr_eval.py`` enforces a first-class HARD **overfit gate** on every submission —
``gap = hidden_acc - audit_acc``, hard-reject at ``> 0.10`` and a 0.85 score penalty at
``> 0.05`` (GATE 5, ``_OVERFIT_HARD_REJECT`` / ``_OVERFIT_PENALTY``). But on the
maintainer's own TRINITY development there is **no offline report of that same gap**:
``scripts/audit_eval.py`` writes the sealed "final honest number" to ``audit_result.json``
(``results.TRINITY`` on the audit split), yet nothing consumes it — ``results_table.py``
globs only ``experiments/**/eval*.json``, which excludes ``audit_result.json`` by name.
So dev-side eval and the sealed audit are never paired, and the generalization gap the
submission gate lives on is invisible during development. (ROADMAP infra priority:
"audit-set integrity".)

:func:`analyze_pair` pairs one ``eval.json`` with its ``audit_result.json`` and reports
the TRINITY eval→audit gap, flagged against the **same** ``pr_eval`` thresholds (mirrored
here and pinned by a test), so a run that would trip GATE 5 is visible before submission.
Read-only arithmetic over existing JSON — no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeGuard

__all__ = [
    "OVERFIT_HARD_REJECT",
    "OVERFIT_PENALTY",
    "GeneralizationGap",
    "overfit_verdict",
    "analyze_pair",
    "render",
]

#: Mirrors scripts/pr_eval.py GATE 5 (``_OVERFIT_HARD_REJECT`` / ``_OVERFIT_PENALTY``);
#: tests/test_generalization.py pins these equal so this report can never drift from the
#: gate it previews.
OVERFIT_HARD_REJECT: float = 0.10
OVERFIT_PENALTY: float = 0.05


def _is_num(x: Any) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _trinity(d: Mapping[str, Any]) -> float | None:
    v = (d.get("results") or {}).get("TRINITY")
    return float(v) if _is_num(v) else None


def _best_single(d: Mapping[str, Any]) -> float | None:
    singles = [v for k, v in (d.get("results") or {}).items()
               if k.startswith("single::") and _is_num(v)]
    return float(max(singles)) if singles else None


def overfit_verdict(gap: float) -> tuple[str, float]:
    """Map an eval-audit gap to (label, score-penalty-factor), mirroring pr_eval GATE 5.

    ``gap > OVERFIT_HARD_REJECT`` -> ``("reject", 0.0)``; ``gap > OVERFIT_PENALTY`` ->
    ``("penalty", 0.85)``; otherwise ``("ok", 1.0)``. A negative gap (audit >= eval) is
    "ok" — no overfit.
    """
    if gap > OVERFIT_HARD_REJECT:
        return "reject", 0.0
    if gap > OVERFIT_PENALTY:
        return "penalty", 0.85
    return "ok", 1.0


@dataclass(frozen=True)
class GeneralizationGap:
    """The eval→audit generalization gap for one benchmark run."""

    benchmark: str
    eval_trinity: float | None
    audit_trinity: float | None
    gap: float | None
    verdict: str
    penalty_factor: float
    eval_best_single: float | None
    audit_best_single: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "eval_trinity": self.eval_trinity,
            "audit_trinity": self.audit_trinity,
            "gap": self.gap,
            "verdict": self.verdict,
            "penalty_factor": self.penalty_factor,
            "eval_best_single": self.eval_best_single,
            "audit_best_single": self.audit_best_single,
        }


def analyze_pair(
    eval_result: Mapping[str, Any],
    audit_result: Mapping[str, Any],
    *,
    benchmark: str | None = None,
) -> GeneralizationGap:
    """Compute the TRINITY eval→audit gap + GATE-5 verdict for one paired run.

    ``eval_result`` is a ``trinity.eval`` output dict, ``audit_result`` a
    ``scripts/audit_eval.py`` output dict — both carry ``results.TRINITY``. If either
    TRINITY score is missing the verdict is ``"n/a"``.
    """
    bench = str(benchmark or eval_result.get("benchmark") or audit_result.get("benchmark") or "?")
    et, at = _trinity(eval_result), _trinity(audit_result)
    if et is None or at is None:
        return GeneralizationGap(bench, et, at, None, "n/a", 1.0,
                                 _best_single(eval_result), _best_single(audit_result))
    gap = et - at
    verdict, penalty = overfit_verdict(gap)
    return GeneralizationGap(bench, et, at, gap, verdict, penalty,
                             _best_single(eval_result), _best_single(audit_result))


def render(gaps: Sequence[GeneralizationGap]) -> str:
    """Markdown report: a per-benchmark eval→audit gap table + a GATE-5 verdict summary."""
    out = ["# Eval → audit generalization (overfit) gap\n"]
    if not gaps:
        return "".join(out) + "\n_(no paired eval/audit runs found)_\n"

    out.append(f"Flagged against the pr_eval submission gate: penalty at gap > "
               f"{OVERFIT_PENALTY}, hard-reject at gap > {OVERFIT_HARD_REJECT}.\n")
    out.append("| benchmark | eval TRINITY | audit TRINITY | gap | verdict |")
    out.append("|---|---|---|---|---|")
    marks = {"ok": "✅", "penalty": "⚠", "reject": "❌", "n/a": "—"}
    for g in sorted(gaps, key=lambda x: x.benchmark):
        et = f"{g.eval_trinity:.3f}" if g.eval_trinity is not None else "—"
        at = f"{g.audit_trinity:.3f}" if g.audit_trinity is not None else "—"
        gp = f"{g.gap:+.3f}" if g.gap is not None else "—"
        out.append(f"| {g.benchmark} | {et} | {at} | {gp} | {marks.get(g.verdict, '')} {g.verdict} |")

    rejects = [g.benchmark for g in gaps if g.verdict == "reject"]
    penalties = [g.benchmark for g in gaps if g.verdict == "penalty"]
    if rejects:
        out.append(f"\n**❌ would be REJECTED by GATE 5** (gap > {OVERFIT_HARD_REJECT}): "
                   f"{', '.join(rejects)}")
    elif penalties:
        out.append(f"\n**⚠ would be PENALIZED** (gap > {OVERFIT_PENALTY}): {', '.join(penalties)}")
    else:
        out.append("\n**✅ all runs within the overfit tolerance.**")
    return "\n".join(out) + "\n"
