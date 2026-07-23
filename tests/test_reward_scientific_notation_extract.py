"""Unboxed scientific notation must be extracted whole (issue #436)."""
from __future__ import annotations

from trinity.orchestration.reward import extract_last_number, score_text


def test_one_e_three_not_read_as_three():
    assert extract_last_number("the answer is 1e3") == "1e3"
    assert score_text("math500", "the answer is 1e3", "1000") == 1.0
    assert score_text("math500", "the answer is 1e3", "3") == 0.0


def test_signed_and_decimal_exponents():
    assert extract_last_number("approx 2.5e-2") == "2.5e-2"
