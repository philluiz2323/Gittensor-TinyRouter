"""Strip a single-letter ``x=`` prefix from solve-for-x answers (issue #348).

MATH-500 references are often bare values (``5``) while models box
``x=5``. The normalizer already strips a leading ``=``; this extends it to a
single leading letter assignment so correct answers stop scoring 0.0.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_single_letter_assignment_normalizes_to_rhs():
    assert normalize_math_answer("x=5") == "5"
    assert normalize_math_answer("x = 5") == "5"
    assert normalize_math_answer("n=10") == "10"


def test_boxed_equation_form_grades_correct():
    assert score_text("math500", r"\boxed{x=5}", "5") == 1.0
    assert score_text("math500", r"\boxed{x = 5}", "5") == 1.0
    assert score_text("math500", r"\boxed{n=10}", "10") == 1.0


def test_bare_value_and_wrong_assignment_unchanged():
    assert score_text("math500", r"\boxed{5}", "5") == 1.0
    assert math_equal("x=5", "6") is False


def test_multi_char_lhs_is_not_stripped():
    # Conservative: only a single leading letter is removed.
    assert normalize_math_answer("log=2") == "log=2"
