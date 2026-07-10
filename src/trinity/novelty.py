"""Offline novelty analysis: how different are two heads' routing decisions.

Why this exists
---------------
Novelty is 5% of the competition score (CONTRIBUTING.md / SUBMITTING.md:
"different routing choices from other miners"). The hidden scorer defines it in
``scripts/pr_eval.py::_compute_novelty`` as::

    novelty = 1.0 - agreement_rate

where ``agreement_rate`` is the fraction of reference questions on which the
submitted head and the current king pick the SAME ``(agent, role)``. When there
is no king yet it returns a neutral ``0.5``.

Obtaining the two heads' decisions needs the torch model (``LinearHead.select``
+ the encoder), but the *scoring* — turning two aligned decision sequences into a
novelty number — is pure arithmetic that a contributor cannot currently run
without the leaderboard machinery. This module exposes that pure part, plus the
per-question and per-choice breakdown the single scalar hides and a diversity
measure of a single head's own choices.

Feed it the decisions you already have (e.g. ``head.select(...)[:2]`` per
question for your head and any reference head), and it tells you your novelty and
exactly which questions and choices drive it.

Pure / deterministic / no network / no GPU / no torch.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Sequence

__all__ = [
    "NEUTRAL_NOVELTY",
    "Decision",
    "NoveltyReport",
    "DiversityReport",
    "normalize_decision",
    "agreement_rate",
    "novelty_score",
    "novelty_report",
    "selection_diversity",
]

#: Novelty the hidden scorer returns when there is no reference head to compare to.
NEUTRAL_NOVELTY = 0.5

#: A routing decision: normally ``(agent_index, role)``, but any hashable works.
Decision = Hashable


def normalize_decision(decision: Any) -> Decision:
    """Coerce one decision to a stable, comparable, hashable key.

    Accepts a bare choice, or an ``(agent, role)`` pair where ``role`` may be an
    enum (``Role.WORKER``) or a string. An enum is reduced to its ``name`` (else
    ``value``, else ``str``) so two runs compare equal regardless of object
    identity, and the key round-trips through JSON.
    """
    def _norm_one(x: Any) -> Any:
        if isinstance(x, (str, int, bool)) or x is None:
            return x
        for attr in ("name", "value"):
            if hasattr(x, attr):
                return getattr(x, attr)
        return str(x)

    if isinstance(decision, tuple):
        return tuple(_norm_one(x) for x in decision)
    return _norm_one(decision)


def _aligned(a: Sequence[Any], b: Sequence[Any]) -> tuple[list[Decision], list[Decision]]:
    if len(a) != len(b):
        raise ValueError(
            f"decision sequences must be aligned by question; got {len(a)} vs {len(b)}"
        )
    return [normalize_decision(x) for x in a], [normalize_decision(x) for x in b]


def agreement_rate(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Fraction of positions where the two heads pick the same decision.

    The two sequences must be aligned by question (same length). An empty input
    is ``1.0`` (two heads that decide nothing trivially agree), matching the
    ``novelty == 0`` that ``novelty_score`` then returns.
    """
    na, nb = _aligned(a, b)
    if not na:
        return 1.0
    matches = sum(1 for x, y in zip(na, nb) if x == y)
    return matches / len(na)


def novelty_score(a: Sequence[Any], b: Sequence[Any]) -> float:
    """``1.0 - agreement_rate(a, b)`` — the hidden scorer's novelty definition."""
    return 1.0 - agreement_rate(a, b)


@dataclass(frozen=True)
class NoveltyReport:
    """Novelty of ``head`` vs ``reference``, with the breakdown the scalar hides.

    ``differing_indices`` are the question positions where the two heads chose
    differently — the ones that generate novelty. ``switched_from_to`` counts each
    ``(reference_choice -> head_choice)`` transition, so a contributor can see
    *which* routing changes drive (or fail to drive) their novelty.
    """

    n_questions: int
    n_agree: int
    agreement_rate: float
    novelty: float
    differing_indices: list[int]
    switched_from_to: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_questions": self.n_questions,
            "n_agree": self.n_agree,
            "agreement_rate": self.agreement_rate,
            "novelty": self.novelty,
            "differing_indices": list(self.differing_indices),
            "switched_from_to": dict(self.switched_from_to),
        }


def novelty_report(
    head: Sequence[Any],
    reference: Sequence[Any] | None,
    *,
    neutral: float = NEUTRAL_NOVELTY,
) -> NoveltyReport:
    """Full novelty breakdown of ``head`` against ``reference``.

    ``reference is None`` (no king yet) yields the neutral novelty the scorer
    uses, with an empty breakdown — matching ``pr_eval`` returning ``0.5`` when no
    king exists.

    Args:
        head: This head's per-question decisions.
        reference: The reference (king) head's decisions, or ``None``.
        neutral: Novelty to report when there is no reference.

    Returns:
        A :class:`NoveltyReport`.
    """
    if reference is None:
        return NoveltyReport(
            n_questions=len(head), n_agree=0, agreement_rate=1.0 - neutral,
            novelty=neutral, differing_indices=[], switched_from_to={},
        )

    na, nb = _aligned(head, reference)
    differing: list[int] = []
    switches: Counter[str] = Counter()
    for i, (h, r) in enumerate(zip(na, nb)):
        if h != r:
            differing.append(i)
            switches[f"{r} -> {h}"] += 1

    n = len(na)
    n_agree = n - len(differing)
    agree = n_agree / n if n else 1.0
    return NoveltyReport(
        n_questions=n,
        n_agree=n_agree,
        agreement_rate=agree,
        novelty=1.0 - agree,
        differing_indices=differing,
        switched_from_to=dict(switches),
    )


@dataclass(frozen=True)
class DiversityReport:
    """How spread out a SINGLE head's own routing choices are.

    A head that always picks the same worker has no room to be novel and cannot
    exploit routing headroom. ``normalized_entropy`` is ``1.0`` for a uniform
    spread and ``0.0`` for a head that always makes the same choice.
    """

    n_questions: int
    n_distinct: int
    counts: dict[str, int]
    top_choice: str | None
    top_share: float
    normalized_entropy: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_questions": self.n_questions,
            "n_distinct": self.n_distinct,
            "counts": dict(self.counts),
            "top_choice": self.top_choice,
            "top_share": self.top_share,
            "normalized_entropy": self.normalized_entropy,
        }


def selection_diversity(decisions: Sequence[Any]) -> DiversityReport:
    """Distribution, dominant-choice share, and normalized entropy of choices.

    ``normalized_entropy`` is the Shannon entropy of the choice distribution
    divided by ``log(n_distinct)``, so it is ``1.0`` for a perfectly uniform head
    and ``0.0`` for a head that always makes the same choice (or an empty input).
    """
    norm = [normalize_decision(d) for d in decisions]
    counts: Counter[Any] = Counter(norm)
    n = len(norm)
    if n == 0:
        return DiversityReport(0, 0, {}, None, 0.0, 0.0)

    top, top_n = counts.most_common(1)[0]
    n_distinct = len(counts)
    if n_distinct <= 1:
        norm_entropy = 0.0
    else:
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
        norm_entropy = entropy / math.log(n_distinct)

    return DiversityReport(
        n_questions=n,
        n_distinct=n_distinct,
        counts={str(k): v for k, v in counts.most_common()},
        top_choice=str(top),
        top_share=top_n / n,
        normalized_entropy=norm_entropy,
    )
