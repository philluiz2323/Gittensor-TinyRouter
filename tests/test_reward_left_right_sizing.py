"""LaTeX sizing delimiters must not change a math answer's value (issue #432)."""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_left_right_frac_matches_bare_half():
    boxed = r"Final answer: \boxed{\left(\frac{1}{2}\right)}"
    assert score_text("math500", boxed, "1/2") == 1.0
    assert score_text("math500", r"\boxed{\frac{1}{2}}", r"\left(\frac{1}{2}\right)") == 1.0


def test_bigl_bigr_frac_matches_bare_half():
    assert score_text("math500", r"\boxed{\bigl(\frac{1}{2}\bigr)}", "1/2") == 1.0
    assert "bigl" not in normalize_math_answer(r"\bigl(\frac{1}{2}\bigr)")


def test_structured_tuple_parens_are_not_peeled_away():
    # Outer parens on a multi-element answer are value, not sizing noise.
    assert math_equal("(5, 120)", "(5, 120)")
    assert normalize_math_answer("(5, 120)").startswith("(")
