"""Offline tests: the math grader must not accept wrong large-magnitude answers.

`math_equal`'s numeric fallback used a *relative* tolerance `rel_tol * max(1,|a|,|b|)`
that grew with the answer's magnitude, so once `|value| >= 1e6` the threshold reached
`>= 1.0` and an off-by-one (or larger) integer was graded correct (issue #141). The
fallback is now an ABSOLUTE tolerance, so genuinely different numbers never merge while
rounded float-vs-fraction forms of the same value still match. No network, no GPU.
"""
from __future__ import annotations

import pytest

from trinity.orchestration import reward as R


# --- the bug: different large integers must NOT be equal ---


@pytest.mark.parametrize(
    "candidate,reference",
    [
        ("1000001", "1000000"),        # off-by-one at 1e6
        ("5000003", "5000000"),        # off-by-three at 5e6
        ("10000005", "10000000"),      # off-by-five at 1e7
        (r"\boxed{1000001}", "1000000"),
        ("-1000001", "-1000000"),      # negative magnitude too
    ],
)
def test_wrong_large_integer_scores_zero(candidate, reference):
    assert R.score_text("math500", candidate, reference) == 0.0
    assert R.math_equal(candidate, reference) is False


def test_math_equal_does_not_merge_off_by_one_millions():
    assert R.math_equal("1000001", "1000000") is False
    assert R.math_equal("999999999", "1000000000") is False


# --- preserved behaviour: exact and rounded-representation matches still hold ---


def test_correct_large_integer_still_scores_one():
    assert R.score_text("math500", "1000000", "1000000") == 1.0
    assert R.score_text("math500", r"\boxed{1000000}", "1000000") == 1.0
    # integer-valued float form of the same number
    assert R.math_equal("1000000.0", "1000000") is True


def test_thousands_separated_large_answer_still_matches():
    assert R.math_equal("1,000,000", "1000000") is True


def test_rounded_decimal_vs_fraction_still_matches():
    # The absolute tolerance still bridges a truncated decimal to its fraction.
    assert R.math_equal("0.333333", "1/3") is True
    assert R.math_equal("0.5", "1/2") is True


def test_small_wrong_answers_still_rejected():
    assert R.score_text("math500", "43", "42") == 0.0
    assert R.score_text("math500", r"\boxed{2{,}048}", "2049") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
