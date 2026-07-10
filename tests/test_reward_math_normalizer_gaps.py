"""Two math-answer normalizer gaps: \\tfrac and escaped set-braces.

``normalize_math_answer`` already treats the fraction commands as a family
(\\frac and \\dfrac both normalize to ``a/b``) and already strips a single outer
brace pair. These tests pin the two cases it missed:

* ``\\tfrac{a}{b}`` (textstyle fraction) must normalize like ``\\frac``/``\\dfrac``.
* ``\\{1,2\\}`` (escaped set braces) must strip to ``1,2`` with no stray backslash.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import normalize_math_answer, score_text


# ---------------------------------------------------------------------------
# \tfrac joins the fraction family (\frac / \dfrac / \tfrac all -> a/b)
# ---------------------------------------------------------------------------
def test_tfrac_normalizes_like_frac_and_dfrac():
    assert normalize_math_answer(r"\frac{3}{4}") == "3/4"
    assert normalize_math_answer(r"\dfrac{3}{4}") == "3/4"
    assert normalize_math_answer(r"\tfrac{3}{4}") == "3/4"


def test_tfrac_answer_scores_correct():
    # Two models giving the same value must earn the same reward regardless of
    # whether they wrote \dfrac or \tfrac.
    assert score_text("math500", r"The answer is \boxed{\tfrac{3}{4}}.", "3/4") == 1.0
    assert score_text("math500", r"The answer is \boxed{\dfrac{3}{4}}.", "3/4") == 1.0
    # A wrong \tfrac value is still wrong.
    assert score_text("math500", r"\boxed{\tfrac{1}{2}}", "3/4") == 0.0


def test_cfrac_is_left_untouched():
    # \cfrac (continued fraction) is NOT a plain a/b and must not be rewritten.
    assert normalize_math_answer(r"\cfrac{3}{4}") == r"\cfrac{3}{4}"


# ---------------------------------------------------------------------------
# Escaped set-braces strip cleanly (no stray trailing backslash)
# ---------------------------------------------------------------------------
def test_escaped_set_braces_strip_without_stray_backslash():
    assert normalize_math_answer(r"\{1,2\}") == "1,2"
    assert normalize_math_answer(r"\{\}") == ""


def test_plain_braces_still_strip():
    assert normalize_math_answer(r"{5}") == "5"
    assert normalize_math_answer(r"{1,2}") == "1,2"


def test_escaped_set_answer_matches_plain_reference():
    # candidate uses escaped braces, reference is plainly braced -> must match.
    assert score_text("math500", r"\boxed{\{1,2,3\}}", "{1,2,3}") == 1.0
