"""Degree symbol normalizes in both brace forms: ^\\circ and ^{\\circ}.

``normalize_math_answer`` already strips ``^\\circ`` (unbraced), but the braced
``^{\\circ}`` form — common LaTeX that models frequently emit — was left intact,
so ``\\boxed{90^{\\circ}}`` scored 0 against a plain ``90`` while ``\\boxed{90^\\circ}``
scored 1. These tests pin both forms to the same normalized value.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import normalize_math_answer, score_text


def test_both_degree_brace_forms_normalize_equally():
    assert normalize_math_answer(r"90^\circ") == "90"
    assert normalize_math_answer(r"90^{\circ}") == "90"
    assert normalize_math_answer(r"45^{\circ}") == "45"


def test_degree_answer_scores_correct_in_either_form():
    assert score_text("math500", r"The angle is \boxed{90^{\circ}}.", "90") == 1.0
    assert score_text("math500", r"The angle is \boxed{90^\circ}.", "90") == 1.0
    # A degree answer still compares by value, so a wrong angle is wrong.
    assert score_text("math500", r"\boxed{45^{\circ}}", "90") == 0.0


def test_degree_matches_a_degreed_reference_regardless_of_brace_style():
    # candidate braced, reference unbraced (and vice versa) must agree.
    assert score_text("math500", r"\boxed{60^{\circ}}", r"60^\circ") == 1.0
    assert score_text("math500", r"\boxed{60^\circ}", r"60^{\circ}") == 1.0


def test_bare_circ_without_caret_is_left_untouched():
    # \circ without a caret (e.g. function composition) is not a degree symbol.
    assert normalize_math_answer(r"f\circ g") == r"f\circg"


def test_degree_keyword_still_stripped():
    assert normalize_math_answer(r"90\degree") == "90"
