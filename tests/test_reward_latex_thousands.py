"""Regression: LaTeX digit-grouping thousands (``1{,}000``) must grade equal.

MATH-500 answers commonly write thousands with LaTeX's ``{,}`` group (which
renders as a comma), e.g. ``\\boxed{2{,}048}``. The grader stripped *bare* commas
("1,000" -> "1000") but not the braced form, so ``\\boxed{2{,}048}`` never matched
``2048`` — a false negative on correct answers. Both the boxed path
(``normalize_math_answer``) and the fallback path (``extract_last_number``) must
handle it, and it must not create false positives on comma-separated lists.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import pytest

from trinity.orchestration import reward as R


@pytest.mark.parametrize(
    "candidate,reference",
    [
        (r"\boxed{1{,}000}", "1000"),
        (r"\boxed{2{,}048}", "2048"),
        (r"\boxed{12{,}345}", "12345"),
        (r"\boxed{1{,}234{,}567}", "1234567"),
        ("The answer is 1{,}000", "1000"),          # fallback (no \boxed)
        (r"\boxed{1{,}000}", "1,000"),               # reference also braced/comma
    ],
)
def test_latex_grouped_thousands_grade_correct(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 1.0


def test_bare_comma_and_plain_still_work():
    # Regression guard: the pre-existing bare-comma handling is unaffected.
    assert R.score_text("math500", r"\boxed{1,000}", "1000") == 1.0
    assert R.score_text("math500", r"\boxed{42}", "42") == 1.0


def test_no_false_positive_on_wrong_or_list_answers():
    # A comma-separated list is not a single number; a wrong number stays wrong.
    assert R.score_text("math500", r"\boxed{1,2,3}", "6") == 0.0
    assert R.score_text("math500", r"\boxed{2{,}048}", "2049") == 0.0


def test_extract_last_number_reads_grouped_form():
    assert R.extract_last_number("the total is 1{,}000") == "1000"
    assert R.extract_last_number("cost 12{,}345 dollars") == "12345"
