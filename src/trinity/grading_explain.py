"""Explain WHY a candidate answer grades correct or incorrect.

Why this exists
---------------
``trinity.orchestration.reward`` grades an answer, but it is a black box: it
returns ``1.0`` or ``0.0`` with no trace. ``docs/JOURNAL.md`` repeatedly finds
that lost points are a *format* problem, not a reasoning one — a correct answer
in the wrong shape (``$18.90`` vs ``\$18.90``, ``2{,}048`` vs ``2048``, a choice
letter buried in prose) scores 0, and there is no offline way to see which step
of the grade failed.

This module runs the SAME extract / normalize / compare pipeline the grader uses
(reusing ``reward``'s public helpers — it changes nothing) and records each step,
so a contributor can see exactly why an answer scored what it did:

  * which extractor fired and what it pulled out of the candidate;
  * the normalized candidate and reference forms that were compared;
  * whether the match was exact, numeric, or symbolic — or why it failed.

It is a read-only diagnostic. The final ``score`` it reports is taken from
``reward.score_text`` itself, so the explanation can never disagree with the real
grade.

Pure / deterministic / no network / no GPU / no torch (sympy is used by the
grader only if importable, exactly as in production).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["GradeExplanation", "explain_grade"]


@dataclass
class GradeExplanation:
    """A step-by-step account of one grade.

    ``score`` is the authoritative grade from ``reward.score_text``; ``steps`` are
    the human-readable trace; ``detail`` carries the structured intermediate
    values (extracted / normalized forms) for programmatic use.
    """

    benchmark: str
    kind: str                       # "math" | "choice" | "code" | "unknown"
    candidate: str
    reference: str
    score: float
    correct: bool
    steps: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "kind": self.kind,
            "candidate": self.candidate,
            "reference": self.reference,
            "score": self.score,
            "correct": self.correct,
            "steps": list(self.steps),
            "detail": dict(self.detail),
        }


def _explain_math(R: Any, candidate: str, reference: str,
                  steps: list[str], detail: dict[str, Any]) -> None:
    boxed = R.extract_boxed(candidate)
    if boxed is not None:
        extracted, how = boxed, "boxed"
    else:
        num = R.extract_last_number(candidate)
        if num is not None:
            extracted, how = num, "last-number"
        else:
            extracted, how = candidate, "whole-candidate (no boxed/number found)"
    steps.append(f"extract: {how} -> {extracted!r}")

    ref_boxed = R.extract_boxed(reference)
    ref_val = ref_boxed if ref_boxed is not None else reference
    if ref_boxed is not None:
        steps.append(f"reference was boxed -> {ref_val!r}")

    ncand, nref = R.normalize_math_answer(extracted), R.normalize_math_answer(ref_val)
    steps.append(f"normalize: candidate {ncand!r} vs reference {nref!r}")
    detail.update(extracted=extracted, extractor=how, normalized_candidate=ncand,
                  normalized_reference=nref)

    if R.math_equal(extracted, ref_val):
        if ncand == nref and ncand != "":
            steps.append("match: exact after normalization")
        else:
            steps.append("match: numeric/symbolic equivalence (normalized forms differ)")
    else:
        steps.append("no match: normalized forms are not equal and no numeric/symbolic "
                     "equivalence was found")


def _explain_choice(R: Any, candidate: str, reference: str,
                    steps: list[str], detail: dict[str, Any]) -> None:
    got = R.extract_choice_letter(candidate)
    ref_letter = R.extract_choice_letter(reference) or reference.strip().upper()[:1]
    steps.append(f"extract choice: candidate -> {got!r}; reference -> {ref_letter!r}")
    detail.update(extracted_letter=got, reference_letter=ref_letter)
    if got is None:
        steps.append("no match: no choice letter (A-D) could be extracted from the candidate")
    elif got == ref_letter:
        steps.append("match: extracted letter equals the reference letter")
    else:
        steps.append(f"no match: extracted {got!r} != reference {ref_letter!r}")


def explain_grade(benchmark: str, candidate: str, reference: object) -> GradeExplanation:
    """Grade ``candidate`` against ``reference`` and explain each step.

    Args:
        benchmark: Benchmark id (case-insensitive), e.g. ``"math500"`` / ``"mmlu"``.
        candidate: The model's answer text.
        reference: The gold answer (string for math/choice; a test spec for code).

    Returns:
        A :class:`GradeExplanation` whose ``score`` matches ``reward.score_text``.
    """
    from trinity.orchestration import reward as R

    key = (benchmark or "").strip().lower()
    ref_str = reference if isinstance(reference, str) else str(reference)
    steps: list[str] = []
    detail: dict[str, Any] = {}

    if key in R.MATH_BENCHMARKS:
        kind = "math"
        _explain_math(R, candidate, ref_str, steps, detail)
    elif key in R.CHOICE_BENCHMARKS:
        kind = "choice"
        _explain_choice(R, candidate, ref_str, steps, detail)
    elif key in R.CODE_BENCHMARKS:
        kind = "code"
        has = R.has_answer(key, candidate)
        steps.append(f"code: extractable code present = {has} "
                     "(functional correctness is decided by running the tests)")
        detail["has_code"] = has
    else:
        kind = "unknown"
        steps.append(f"unknown benchmark {benchmark!r}: no grading pipeline")

    # The authoritative score comes from the grader itself, so the explanation can
    # never contradict the real grade.
    try:
        score = R.score_text(key, candidate, reference)
    except ValueError:
        score = 0.0
        steps.append("score_text rejected the benchmark; reporting 0.0")

    return GradeExplanation(
        benchmark=key or str(benchmark), kind=kind, candidate=candidate,
        reference=ref_str, score=score, correct=score > 0.0, steps=steps, detail=detail,
    )
