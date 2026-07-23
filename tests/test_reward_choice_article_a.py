"""English article "a" / pronoun "I" must not be graded as MMLU choices (issue #413)."""
from __future__ import annotations

from trinity.orchestration.reward import extract_choice_letter, score_text


def test_article_phrase_is_not_choice_a():
    assert extract_choice_letter("The answer is a decrease in pressure.") is None
    assert score_text("mmlu", "The answer is a decrease in pressure.", "A") == 0.0


def test_pronoun_i_think_is_not_choice_i():
    assert extract_choice_letter("The answer is I think option B is correct") == "B"
    assert extract_choice_letter("The answer is I believe option C") == "C"


def test_trailing_article_does_not_override_boxed():
    text = "\\boxed{C}\n\nSo the answer is a straightforward application of the rule."
    assert extract_choice_letter(text) == "C"
    assert score_text("mmlu", text, "C") == 1.0


def test_genuine_uppercase_a_still_counts():
    assert extract_choice_letter("The answer is A because the derivative is positive.") == "A"
    assert extract_choice_letter("The answer is a.") == "A"
    assert extract_choice_letter("the answer is c.") == "C"
    assert extract_choice_letter("The answer is I.") == "I"
