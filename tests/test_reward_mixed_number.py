"""Mixed numbers must not collapse into concatenated fractions (issue #438)."""
from __future__ import annotations

from trinity.orchestration.reward import normalize_math_answer, score_text


def test_one_and_a_half():
    assert normalize_math_answer("1 1/2") == "3/2"
    assert score_text("math500", r"\boxed{1 1/2}", "3/2") == 1.0
    assert score_text("math500", r"\boxed{1 1/2}", "11/2") == 0.0
