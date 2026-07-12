r"""A radical normalizes across ``\sqrt{x}`` and the unbraced ``\sqrt x``.

MATH500 answers are frequently radicals (``\sqrt{2}``, ``2\sqrt{3}``,
``\frac{\sqrt{3}}{2}``). A model and the reference routinely spell the same value
two ways — the braced ``\sqrt{2}`` and the bare ``\sqrt2`` — and
``normalize_math_answer`` previously left both intact, so ``\sqrt{2}`` scored 0
against ``\sqrt2`` (a false negative). It also broke the sympy fallback: the bare
backslash in ``\sqrt`` makes ``parse_expr`` raise, so ``2\sqrt{3}`` never matched
``2sqrt(3)``. These tests pin both spellings to sympy's canonical ``sqrt(x)`` form.

This mirrors the existing ``\frac{a}{b}`` / ``\frac a b`` brace-agnostic handling.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_braced_and_bare_sqrt_normalize_equally():
    assert normalize_math_answer(r"\sqrt{2}") == "sqrt(2)"
    assert normalize_math_answer(r"\sqrt2") == "sqrt(2)"
    assert normalize_math_answer(r"\sqrt 2") == "sqrt(2)"
    assert normalize_math_answer(r"2\sqrt{3}") == "2sqrt(3)"
    assert normalize_math_answer(r"2\sqrt3") == "2sqrt(3)"


def test_sqrt_forms_compare_equal_regardless_of_braces():
    assert math_equal(r"\sqrt{2}", r"\sqrt2")
    assert math_equal(r"2\sqrt{3}", r"2\sqrt3")
    # Folding to sqrt(...) also lets the plain sympy spelling match the LaTeX one.
    assert math_equal(r"\sqrt2", "sqrt(2)")


def test_sqrt_answer_scores_correct_in_either_form():
    assert score_text("math500", r"The side is \boxed{2\sqrt{3}}.", r"2\sqrt3") == 1.0
    assert score_text("math500", r"\boxed{\sqrt{5}}", r"\sqrt{5}") == 1.0
    # A radical still compares by value, so a different radicand is wrong.
    assert score_text("math500", r"\boxed{\sqrt{2}}", r"\sqrt{3}") == 0.0


def test_sqrt_over_a_scalar_denominator_is_bridged_symbolically():
    # \sqrt{3}/2 vs \sqrt3/2 — the denominator keeps them out of the pure-radical
    # exact match, so this exercises the sqrt(...) -> sympy bridge.
    assert math_equal(r"\sqrt{3}/2", r"\sqrt3/2")
