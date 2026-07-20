"""Offline R12 check: is TRINITY far more token-efficient than the multi-agent baselines?

``docs/SPEC.md`` §1.3 invariant **R12** — *"TRINITY far more token-efficient than
MoA/Smoothie/MasRouter"* (Table 9) — is a replication requirement, yet nothing in ``src/``
or ``scripts/`` verifies it. :mod:`trinity.efficiency` measures the *competition* efficiency
term (turns per correct answer **inside one head**, the 10% score component); it never
compares TRINITY's token spend against the heavyweight multi-agent routing baselines, which
is the distinct claim R12 makes. R12 is the efficiency half of the thesis: comparable or
better accuracy for far fewer tokens.

The comparison is **tokens per correct answer** — ``tokens_per_query / accuracy`` — not raw
tokens per query, so a baseline cannot look efficient by answering cheaply and wrongly: a
method that halves its tokens but also halves its accuracy is no more efficient per correct
answer. R12 holds when TRINITY's speedup over **every** baseline
(``baseline_tpc / trinity_tpc``) clears a factor — default ``2.0``, since R12 claims *far*
more efficient, not merely more.

A system that is never correct (``accuracy <= 0``) has an undefined, infinitely bad
tokens-per-correct: as a baseline it is trivially worse than TRINITY, and as TRINITY it
fails R12 against any baseline that answers at all.

Pure stdlib over plain numbers — no torch, no network, no GPU. (issue #364)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, TypeGuard

__all__ = [
    "TRINITY_KEY",
    "DEFAULT_FACTOR",
    "SystemEfficiency",
    "BaselineComparison",
    "tokens_per_correct",
    "analyze",
    "render",
]

#: Key identifying the trained router among the systems.
TRINITY_KEY = "TRINITY"

#: Speedup a baseline must be beaten by for R12's "far more" to hold.
DEFAULT_FACTOR = 2.0


def _is_num(x: Any) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def tokens_per_correct(accuracy: float, tokens_per_query: float) -> float:
    """Tokens spent per **correct** answer: ``tokens_per_query / accuracy``.

    Returns ``inf`` when ``accuracy <= 0`` — a system that never answers correctly spends
    unboundedly many tokens per correct answer, which is exactly how it should compare.
    """
    if accuracy <= 0.0:
        return math.inf
    return tokens_per_query / accuracy


def _jsonable(x: float) -> float | None:
    """``x`` if finite, else ``None`` (JSON has no infinity)."""
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class SystemEfficiency:
    """One system's accuracy / token spend, reduced to tokens per correct answer."""

    name: str
    accuracy: float
    tokens_per_query: float
    tokens_per_correct: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (an infinite tokens-per-correct becomes ``None``)."""
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "tokens_per_query": self.tokens_per_query,
            "tokens_per_correct": _jsonable(self.tokens_per_correct),
        }


@dataclass(frozen=True)
class BaselineComparison:
    """TRINITY against one multi-agent baseline."""

    baseline: str
    accuracy: float
    tokens_per_query: float
    tokens_per_correct: float
    speedup: float                    # baseline_tpc / trinity_tpc
    far_more_efficient: bool          # speedup >= factor

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "baseline": self.baseline,
            "accuracy": self.accuracy,
            "tokens_per_query": self.tokens_per_query,
            "tokens_per_correct": _jsonable(self.tokens_per_correct),
            "speedup": _jsonable(self.speedup),
            "far_more_efficient": self.far_more_efficient,
        }


def _system(name: str, entry: Any) -> SystemEfficiency | None:
    """Coerce ``{accuracy, tokens_per_query}`` to a :class:`SystemEfficiency`, or ``None``."""
    if not isinstance(entry, Mapping):
        return None
    acc, tok = entry.get("accuracy"), entry.get("tokens_per_query")
    if not _is_num(acc) or not _is_num(tok):
        return None
    accuracy, tokens = float(acc), float(tok)
    if tokens <= 0.0:
        return None
    return SystemEfficiency(name, accuracy, tokens, tokens_per_correct(accuracy, tokens))


def _speedup(baseline_tpc: float, trinity_tpc: float) -> float:
    """``baseline_tpc / trinity_tpc``, with the degenerate ends pinned sensibly."""
    if not math.isfinite(trinity_tpc):
        return 0.0                    # TRINITY never correct -> no efficiency win at all
    if not math.isfinite(baseline_tpc):
        return math.inf               # baseline never correct -> infinitely worse
    return baseline_tpc / trinity_tpc if trinity_tpc > 0.0 else math.inf


def analyze(
    systems: Mapping[str, Any],
    *,
    trinity_key: str = TRINITY_KEY,
    factor: float = DEFAULT_FACTOR,
) -> dict[str, Any]:
    """Verify R12 over ``{system: {accuracy, tokens_per_query}}``; return a JSON-ready report.

    Every system is reduced to tokens per correct answer, then each baseline is compared to
    ``trinity_key``. R12 **holds** iff TRINITY beats every baseline by at least ``factor``.
    Entries that are not ``{accuracy, tokens_per_query}`` with a positive token count are
    dropped (a malformed row must not silently pass the invariant).

    Raises:
        ValueError: When ``trinity_key`` is absent or malformed — R12 is meaningless
            without the system it is a claim about.
    """
    trinity = _system(trinity_key, systems.get(trinity_key))
    if trinity is None:
        raise ValueError(f"missing or malformed {trinity_key!r} entry "
                         "(need {'accuracy': float, 'tokens_per_query': > 0})")

    comparisons: list[BaselineComparison] = []
    for name in sorted(k for k in systems if k != trinity_key):
        sys_ = _system(name, systems[name])
        if sys_ is None:
            continue
        sp = _speedup(sys_.tokens_per_correct, trinity.tokens_per_correct)
        comparisons.append(BaselineComparison(
            baseline=name, accuracy=sys_.accuracy, tokens_per_query=sys_.tokens_per_query,
            tokens_per_correct=sys_.tokens_per_correct, speedup=sp,
            far_more_efficient=sp >= factor,
        ))

    speedups = [c.speedup for c in comparisons]
    worst = min(zip(speedups, [c.baseline for c in comparisons]), default=None)
    return {
        "invariant": "R12",
        "claim": "TRINITY far more token-efficient than the multi-agent baselines",
        "factor": factor,
        "trinity": trinity.to_dict(),
        "baselines": [c.to_dict() for c in comparisons],
        "n_baselines": len(comparisons),
        "min_speedup": _jsonable(worst[0]) if worst else None,
        "worst_baseline": worst[1] if worst else None,
        "holds": bool(comparisons) and all(c.far_more_efficient for c in comparisons),
    }


def render(
    systems: Mapping[str, Any],
    *,
    trinity_key: str = TRINITY_KEY,
    factor: float = DEFAULT_FACTOR,
) -> str:
    """Markdown: per-baseline tokens-per-correct + speedup, and the R12 verdict."""
    report = analyze(systems, trinity_key=trinity_key, factor=factor)
    tri = report["trinity"]
    out = ["# R12 — TRINITY token efficiency vs the multi-agent baselines\n"]
    out.append(f"Efficiency is **tokens per correct answer** (`tokens_per_query / accuracy`), "
               f"so a cheap-but-wrong baseline cannot look efficient. R12 requires a "
               f"**{factor:g}x** speedup over every baseline.\n")
    tpc = tri["tokens_per_correct"]
    out.append(f"**{trinity_key}**: accuracy {tri['accuracy']:.3f}, "
               f"{tri['tokens_per_query']:.0f} tokens/query -> "
               f"{'∞' if tpc is None else f'{tpc:.0f}'} tokens/correct\n")
    if not report["baselines"]:
        return "\n".join(out) + "\n_(no baselines to compare)_\n"

    out.append("| baseline | accuracy | tokens/query | tokens/correct | speedup | far more? |")
    out.append("|---|---|---|---|---|---|")
    for b in report["baselines"]:
        b_tpc = "∞" if b["tokens_per_correct"] is None else f"{b['tokens_per_correct']:.0f}"
        sp = "∞" if b["speedup"] is None else f"{b['speedup']:.2f}x"
        out.append(f"| {b['baseline']} | {b['accuracy']:.3f} | {b['tokens_per_query']:.0f} "
                   f"| {b_tpc} | {sp} | {'✅' if b['far_more_efficient'] else '❌'} |")
    ms = report["min_speedup"]
    out.append(f"\n- worst case: **{report['worst_baseline']}** at "
               f"{'∞' if ms is None else f'{ms:.2f}x'}")
    if report["holds"]:
        out.append(f"\n**R12 HOLDS** ✅ — {trinity_key} is at least {factor:g}x more "
                   "token-efficient per correct answer than every baseline.")
    else:
        losers = [b["baseline"] for b in report["baselines"] if not b["far_more_efficient"]]
        out.append(f"\n**R12 does NOT hold** ❌ — {trinity_key} fails the {factor:g}x bar "
                   f"against: {', '.join(losers)}.")
    return "\n".join(out) + "\n"
