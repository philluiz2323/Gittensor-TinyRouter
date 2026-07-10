"""Offline unit tests for math answer parsing helpers in ``orchestration.reward``.

``extract_boxed``, ``extract_last_number``, and ``normalize_math_answer`` drive
math grading and ``has_answer`` but had no dedicated pytest coverage beyond
indirect ``score_text`` calls.
"""
from __future__ import annotations

from trinity.orchestration import reward as R


# --- extract_boxed ---


def test_extract_boxed_returns_none_for_empty_or_missing():
    assert R.extract_boxed("") is None
    assert R.extract_boxed("no boxed answer here") is None


def test_extract_boxed_handles_nested_braces():
    text = r"First \boxed{1}. Final answer: \boxed{\frac{1}{2}}."
    assert R.extract_boxed(text) == r"\frac{1}{2}"


def test_extract_boxed_takes_last_match():
    text = r"\boxed{old} ... \boxed{new}"
    assert R.extract_boxed(text) == "new"


# --- extract_last_number ---


def test_extract_last_number_takes_last_literal():
    assert R.extract_last_number("first 1/2 then finally 42") == "42"
    assert R.extract_last_number("answer: 1/2") == "1/2"


def test_extract_last_number_strips_thousands_commas():
    assert R.extract_last_number("The total is 1,234.50 dollars.") == "1234.50"


def test_extract_last_number_returns_none_when_absent():
    assert R.extract_last_number("no digits") is None


# --- normalize_math_answer ---


def test_normalize_math_answer_strips_escaped_dollar_before_bare_dollar():
    # Regression guard: reversing this order leaves "\18.90" (see docs/JOURNAL.md).
    assert R.normalize_math_answer(r"\$18.90") == "18.90"


def test_normalize_math_answer_canonicalizes_integer_fraction():
    assert R.normalize_math_answer(r"\frac{3}{4}") == "3/4"
