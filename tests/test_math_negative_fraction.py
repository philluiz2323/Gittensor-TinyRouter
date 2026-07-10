"""Regression tests: negated LaTeX fractions must grade correctly.

`normalize_math_answer` rewrites ``\\frac{a}{b}`` -> ``(a)/(b)``, so a *negated*
fraction ``-\\frac{3}{4}`` becomes ``-(3)/(4)`` with the minus OUTSIDE the paren.
The integer-ratio canonicalizer (and the ``_as_number`` fallback) previously only
read a sign INSIDE the paren, so ``-\\frac{3}{4}`` never canonicalized to ``-3/4``
and, with ``sympy`` unavailable, was graded WRONG against ``-3/4`` / ``-0.75``.
Negative fractions are common MATH500/AIME answers, so this silently deflated
measured accuracy. These tests pin the fix (offline, no sympy required).
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import normalize_math_answer, score_text


@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{-\frac{3}{4}}", "-3/4"),        # the core bug
        (r"\boxed{-\frac{3}{4}}", "-0.75"),       # vs decimal reference
        (r"\boxed{-\frac{1}{2}}", "-1/2"),
        (r"\boxed{-\frac{2}{4}}", "-1/2"),        # reduces
        (r"\boxed{-\frac{6}{8}}", "-3/4"),        # reduces
        (r"\boxed{-\frac{100}{200}}", "-0.5"),
        (r"\boxed{-\frac{-3}{4}}", "3/4"),        # double negative -> positive
        (r"\boxed{-\frac{4}{2}}", "-2"),          # improper -> integer
    ],
)
def test_negated_fraction_scores_correct(candidate, reference):
    assert score_text("math500", candidate, reference) == 1.0


@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{-3/4}", r"-\frac{3}{4}"),       # reference is the LaTeX fraction
        (r"\boxed{-0.75}", r"-\frac{3}{4}"),
    ],
)
def test_negated_fraction_symmetric(candidate, reference):
    assert score_text("math500", candidate, reference) == 1.0


@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{-\frac{3}{4}}", "3/4"),         # sign mismatch
        (r"\boxed{\frac{3}{4}}", "-3/4"),
        (r"\boxed{-\frac{1}{3}}", "-1/2"),        # genuinely unequal
        (r"\boxed{-\frac{4}{2}}", "2"),           # sign mismatch on integer
    ],
)
def test_sign_correctness_preserved(candidate, reference):
    # The fix must NOT introduce false positives: wrong answers stay 0.
    assert score_text("math500", candidate, reference) == 0.0


@pytest.mark.parametrize(
    "candidate, reference",
    [
        (r"\boxed{\frac{3}{4}}", "3/4"),          # positive fractions unaffected
        (r"\boxed{\frac{3}{4}}", "0.75"),
        (r"\boxed{42}", "42"),                    # integers unaffected
        (r"\boxed{-5}", "-5"),
    ],
)
def test_positive_and_integer_cases_unaffected(candidate, reference):
    assert score_text("math500", candidate, reference) == 1.0


def test_normalize_folds_outer_sign():
    # The minus that lands outside the paren is now read into the numerator.
    assert normalize_math_answer(r"-\frac{3}{4}") == "-3/4"
    assert normalize_math_answer(r"-\frac{-3}{4}") == "3/4"     # double negative
    # sign inside the paren already worked; keep it working.
    assert normalize_math_answer(r"\frac{-3}{4}") == "-3/4"
    assert normalize_math_answer(r"\frac{3}{-4}") == "-3/4"
    assert normalize_math_answer(r"\frac{3}{4}") == "3/4"
