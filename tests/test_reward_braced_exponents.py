"""Braced LaTeX exponents must match bare caret powers (issue #434)."""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_braced_power_matches_bare_caret():
    assert normalize_math_answer(r"2^{10}") == normalize_math_answer(r"2^10")
    assert math_equal(r"2^{10}", r"2^10")
    assert score_text("math500", r"\boxed{2^{10}}", r"2^10") == 1.0


def test_braced_negative_exponent():
    assert math_equal(r"10^{-2}", r"10^-2")
