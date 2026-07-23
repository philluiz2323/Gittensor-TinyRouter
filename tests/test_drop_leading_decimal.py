"""DROP metric: a leading-decimal number token (``.5``) must keep its value.

``_normalize_token`` stripped edge punctuation — a set that includes ``.`` —
*before* trying the numeric parse, so ``".5"`` lost its decimal point and
normalized to ``"5.0"``: equal to a gold ``"5"`` (a wrong answer graded
correct) and unequal to the value-identical gold ``"0.5"`` (a correct answer
graded wrong). The official DROP ``_remove_punc`` tests ``_is_number`` first
and leaves numeric tokens untouched, which the module docstring promises to
match. These tests pin the number-first order and guard the surrounding
behaviors (currency/edge punctuation, signs, thousands commas) unchanged.

No network, no GPU, pure text.
"""
from __future__ import annotations

from trinity.adapters.drop import _normalize_tokens, score_drop


# --- normalization: numbers are recognised before punctuation is stripped ---


def test_leading_decimal_keeps_its_value():
    assert _normalize_tokens(".5") == ["0.5"]
    assert _normalize_tokens("0.5") == ["0.5"]


def test_negative_leading_decimal():
    assert _normalize_tokens("-.5") == ["-0.5"]


def test_edge_punctuation_still_stripped_from_non_bare_numbers():
    # Currency and sentence-final punctuation still normalize to the value.
    assert _normalize_tokens("$16") == ["16.0"]
    assert _normalize_tokens("16.") == ["16.0"]
    assert _normalize_tokens("1,234.5") == ["1234.5"]
    # Signs and words are unaffected.
    assert _normalize_tokens("-5") == ["-5.0"]
    assert _normalize_tokens("touchdown") == ["touchdown"]


# --- grading: end-to-end through score_drop ---


def test_point_five_matches_zero_point_five_gold():
    # Correct answer, shorthand spelling: was 0.0 before the fix.
    assert score_drop("Answer: .5", {"gold_answers": ["0.5"]}) == 1.0


def test_point_five_no_longer_matches_gold_five():
    # Wrong answer, ten times off: was 1.0 before the fix.
    assert score_drop("Answer: .5", {"gold_answers": ["5"]}) == 0.0


def test_plain_integer_grading_unchanged():
    assert score_drop("Answer: 5", {"gold_answers": ["5"]}) == 1.0
    assert score_drop("Answer: 4", {"gold_answers": ["5"]}) == 0.0
