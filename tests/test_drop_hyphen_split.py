"""Regression: DROP normalization must tokenize on hyphens, like the official metric.

DROP's official ``_normalize`` tokenizes on ``" |-"`` (space OR hyphen), so a hyphenated
gold span/range and the same answer written with spaces are equal. This module split on
whitespace only, so ``"1994-1995"`` collapsed to the single token ``"19941995"`` and a
model that wrote ``"1994 1995"`` (tokens ``1994 1995``) scored 0.0 — a correct answer
graded wrong. The repo's deliberate leading-sign strictness (``-5`` must not match ``5``)
is preserved: only INTERNAL hyphens split. No torch.
"""
from __future__ import annotations

from trinity.adapters.drop import _normalize_tokens, score_drop


# --------------------------------------------------------------------------- #
# the bug: a hyphenated gold answer vs the same answer written with spaces
# --------------------------------------------------------------------------- #
def test_hyphenated_range_matches_spaced_answer():
    # A year range: gold "1994-1995", model wrote "1994 1995" (equal under DROP).
    assert score_drop("Answer: 1994 1995", {"gold_answers": ["1994-1995"]}) == 1.0
    # And the reverse: gold spaced, model hyphenated.
    assert score_drop("Answer: 1994-1995", {"gold_answers": ["1994 1995"]}) == 1.0


def test_hyphenated_compound_span_matches_spaced():
    # A common football-DROP span: "20-yard" vs "20 yard".
    assert score_drop("Answer: 20 yard", {"gold_answers": ["20-yard"]}) == 1.0
    assert score_drop("Answer: 20-yard", {"gold_answers": ["20 yard"]}) == 1.0
    # a purely lexical compound normalizes the same way
    assert score_drop("Answer: well known", {"gold_answers": ["well-known"]}) == 1.0


def test_hyphen_tokens_normalize_like_spaces():
    assert _normalize_tokens("1994-1995") == _normalize_tokens("1994 1995")
    assert _normalize_tokens("20-yard") == _normalize_tokens("20 yard")


# --------------------------------------------------------------------------- #
# no regression: a wrong hyphenated answer stays wrong, and a leading minus sign
# is still NOT split away (a negative must not match its positive)
# --------------------------------------------------------------------------- #
def test_wrong_hyphenated_answer_still_zero():
    assert score_drop("Answer: 1994 1996", {"gold_answers": ["1994-1995"]}) == 0.0


def test_leading_sign_not_split_by_hyphen_rule():
    assert score_drop("Answer: -5", {"gold_answers": ["-5"]}) == 1.0
    assert score_drop("Answer: 5", {"gold_answers": ["-5"]}) == 0.0
    assert score_drop("Answer: -5", {"gold_answers": ["5"]}) == 0.0
    assert score_drop("Answer: -1,000", {"gold_answers": ["-1000"]}) == 1.0
