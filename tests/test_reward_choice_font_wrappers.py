r"""A boxed, font-wrapped choice letter extracts like a bare one.

Models routinely box a *formatted* answer letter on MMLU / GPQA / MMLU-Pro:
``\boxed{\text{B}}``, ``\boxed{\textbf{D}}``, ``\mathbf{C}``. The math normalizer
already strips exactly these font commands (see test_reward_math_font_commands),
but ``extract_choice_letter`` never did, so a wrapped letter yielded ``None`` and
the answer scored 0 despite being correct. These tests pin the wrapped forms to
the same letter a bare one produces.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

from trinity.orchestration.reward import extract_choice_letter, score_text


def test_boxed_text_wrapped_letter_is_extracted():
    assert extract_choice_letter(r"Therefore the answer is \boxed{\text{B}}.") == "B"
    assert extract_choice_letter(r"\boxed{\textbf{D}}") == "D"
    assert extract_choice_letter(r"Final: $\boxed{\text{J}}$") == "J"


def test_bare_font_wrapped_letter_is_extracted():
    assert extract_choice_letter(r"\textbf{C}") == "C"
    assert extract_choice_letter(r"The answer is \mathbf{A}.") == "A"


def test_nested_font_wrappers_fully_collapse():
    assert extract_choice_letter(r"\boxed{\textbf{\text{F}}}") == "F"


def test_wrapped_letter_scores_correct_across_choice_benchmarks():
    assert score_text("mmlu", r"Therefore \boxed{\text{B}}.", "B") == 1.0
    assert score_text("gpqa", r"\boxed{\textbf{D}}", "D") == 1.0
    assert score_text("mmlu_pro", r"Final answer: \boxed{\text{J}}", "J") == 1.0
    # A wrapped WRONG letter is still wrong — unwrapping changes shape, not value.
    assert score_text("mmlu", r"\boxed{\text{A}}", "B") == 0.0


def test_plain_letters_still_extract_unchanged():
    # The unwrap is a no-op on text with no font commands (regression guard).
    assert extract_choice_letter("The answer is (B).") == "B"
    assert extract_choice_letter("Answer: C") == "C"
    assert extract_choice_letter(r"\boxed{G}") == "G"
