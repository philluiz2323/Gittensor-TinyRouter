r"""The constant pi normalizes across ``\pi``, the Unicode ``π`` and plain ``pi``.

MATH500 answers are frequently multiples or fractions of pi (``2\pi``,
``\frac{\pi}{3}``, ``\pi/2``). A model and the reference routinely spell the same
value three different ways — the LaTeX ``\pi`` command, the Unicode glyph ``π``
(U+03C0), or a bare ``pi`` — and ``normalize_math_answer`` previously left each
form intact, so ``2\pi`` scored 0 against ``2π`` (a false negative). It also broke
the sympy fallback: ``parse_expr`` chokes on the stray backslash in ``\pi``, so
``\pi/2`` never matched ``\frac{\pi}{2}``. These tests pin every spelling to the
same canonical ``pi`` token and confirm value-based grading still rejects a
genuinely different answer.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


def test_all_pi_spellings_normalize_equally():
    assert normalize_math_answer(r"2\pi") == "2pi"
    assert normalize_math_answer("2π") == "2pi"
    assert normalize_math_answer("2 pi") == "2pi"
    assert normalize_math_answer(r"\pi") == "pi"


def test_pi_forms_compare_equal_regardless_of_spelling():
    assert math_equal(r"2\pi", "2π")
    assert math_equal(r"2\pi", "2pi")
    assert math_equal("2π", "2pi")


def test_pi_fraction_matches_across_frac_and_slash_forms():
    # \frac{\pi}{2} folds to (pi)/(2); the plain \pi/2 folds to pi/2. They are the
    # same value, bridged by the symbolic fallback.
    assert math_equal(r"\frac{\pi}{2}", r"\pi/2")
    assert math_equal(r"\frac{\pi}{3}", r"\frac{\pi}{3}")


def test_boxed_pi_answer_scores_correct_in_either_form():
    assert score_text("math500", r"The area is \boxed{2\pi}.", "2π") == 1.0
    assert score_text("math500", r"The area is \boxed{2\pi}.", r"2\pi") == 1.0
    # A pi answer still compares by value, so a different multiple is wrong.
    assert score_text("math500", r"\boxed{3\pi}", r"2\pi") == 0.0


def test_capital_pi_product_symbol_is_not_folded_to_the_constant():
    # \Pi is the product operator, not the constant pi. The fold is case-sensitive
    # (it runs before the lowercasing step), so \Pi is never collapsed to the bare
    # ``pi`` constant token the way \pi is.
    assert normalize_math_answer(r"\Pi") != "pi"


def test_pi_is_not_stripped_from_a_longer_macro_prefix():
    # The negative lookahead keeps \pi a whole command: a hypothetical \pix macro
    # must not become "pix".
    assert normalize_math_answer(r"\pix") == r"\pix"
