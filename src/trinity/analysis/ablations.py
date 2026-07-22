"""Offline R9 check: does removing each design component hurt accuracy?

``docs/SPEC.md`` §1.3 invariant **R9** — *"Removing SVF / Thinker / tri-role /
penultimate-token all hurt; tri-role + token-choice matter most"* (Table 2) — is a
replication requirement, yet nothing in ``src/`` or ``scripts/`` verifies it. R9 is
what justifies each moving part of the coordinator: SVF adaptation, the Thinker
role, the three-role split, and reading the penultimate token (not the last/EOS
token — SPEC §3.2 notes using the last token collapses LiveCodeBench 61.46 -> 50.85).
If an ablation *didn't* hurt, that component would be dead weight.

This reads the full model's accuracy and each ablation's accuracy and reports, per
ablation, the drop ``full - ablation`` (how much removing it hurt), whether it hurt
at all, and the ranking of drops. R9 holds iff **every** ablation is below the full
model; the ranking surfaces which components "matter most".

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "AblationDrop",
    "analyze",
    "render",
]

# A removal must lower accuracy by more than this to count as "hurt" (so float
# noise on a tie is not read as a real effect).
_TOL = 1e-9
_DEFAULT_FULL = "full"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class AblationDrop:
    """One ablation's effect: how much removing a component hurt the full model."""

    ablation: str
    ablation_accuracy: float
    drop: float          # full_accuracy - ablation_accuracy
    hurt: bool           # drop > tol

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "ablation": self.ablation,
            "ablation_accuracy": self.ablation_accuracy,
            "drop": self.drop,
            "hurt": self.hurt,
        }


def analyze(
    accuracies: Mapping[str, Any],
    *,
    full: str = _DEFAULT_FULL,
    tol: float = _TOL,
) -> dict[str, Any]:
    """Compute the R9 ablation drops.

    Args:
        accuracies: ``{variant: accuracy}`` including the ``full`` model and one
            entry per ablation (e.g. ``no_svf``, ``no_thinker``, ``no_trirole``,
            ``last_token``). Non-numeric entries are dropped.
        full: The key naming the full (un-ablated) model (default ``"full"``).
        tol: A removal counts as ``hurt`` only if it lowers accuracy by more than
            this.

    Returns:
        ``{"full_accuracy": float, "drops": [AblationDrop.to_dict, ...] sorted by
        drop desc, "n_ablations": int, "n_hurt": int, "r9_holds": bool,
        "did_not_hurt": [ablation, ...], "matter_most": [ablation, ...]}``.
        ``r9_holds`` is True iff there is at least one ablation and every one hurt.
        ``matter_most`` is the (up to two) ablations with the largest drops.

    Raises:
        KeyError: If ``full`` is not a numeric entry in ``accuracies``.
    """
    clean = {str(k): float(v) for k, v in accuracies.items() if _is_num(v)}
    if full not in clean:
        raise KeyError(f"full model {full!r} not found among {sorted(clean)}")

    full_acc = clean[full]
    drops = [
        AblationDrop(name, acc, full_acc - acc, (full_acc - acc) > tol)
        for name, acc in clean.items() if name != full
    ]
    drops.sort(key=lambda d: d.drop, reverse=True)

    did_not_hurt = [d.ablation for d in drops if not d.hurt]
    matter_most = [d.ablation for d in drops[:2] if d.hurt]
    return {
        "full_accuracy": full_acc,
        "drops": [d.to_dict() for d in drops],
        "n_ablations": len(drops),
        "n_hurt": sum(d.hurt for d in drops),
        "r9_holds": len(drops) > 0 and all(d.hurt for d in drops),
        "did_not_hurt": did_not_hurt,
        "matter_most": matter_most,
    }


def render(
    accuracies: Mapping[str, Any], *, full: str = _DEFAULT_FULL, tol: float = _TOL,
) -> str:
    """A compact text report of the R9 ablation drops and verdict."""
    report = analyze(accuracies, full=full, tol=tol)
    lines = [f"full model accuracy: {report['full_accuracy']:.3f}",
             "",
             "| ablation (removed) | accuracy | drop | hurt? |",
             "|---|---|---|---|"]
    for d in report["drops"]:
        lines.append(
            f"| {d['ablation']} | {d['ablation_accuracy']:.3f} | "
            f"{d['drop']:+.3f} | {'yes' if d['hurt'] else 'NO'} |"
        )
    verdict = "HOLDS" if report["r9_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R9 (removing each component hurts): {verdict} — "
        f"{report['n_hurt']}/{report['n_ablations']} ablations hurt"
    )
    if report["matter_most"]:
        lines.append(f"matter most (largest drops): {', '.join(report['matter_most'])}")
    if report["did_not_hurt"]:
        lines.append(f"did NOT hurt: {', '.join(report['did_not_hurt'])}")
    return "\n".join(lines)
