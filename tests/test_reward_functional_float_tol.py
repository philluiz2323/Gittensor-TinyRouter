"""Regression: the LiveCodeBench functional grader must compare return values
with float tolerance and sequence-type equivalence, not a bare ``==`` (issue #332).

LeetCode / LiveCodeBench functional problems whose answer is a float accept
results "within 1e-5". The child harness previously asserted ``_got == _expected``,
so a representationally-correct float (``2.5000000001`` vs ``2.5``) or a nested
float list graded WRONG -- a false negative on the in-scope LiveCodeBench slice,
while the math path already bridges float reps (``math_equal`` abs_tol=1e-6).
The tolerant comparison must NOT credit genuinely-different values, must not let a
``bool`` bridge to ``1``/``0``, and must leave exact int/string answers unchanged.

Pure / offline -- pass@1 runs candidate code in the sandboxed subprocess the
reward checker uses; no network, no GPU.
"""
from __future__ import annotations

from trinity.orchestration.reward import run_pass_at_1


def _fn(ret: str) -> str:
    return f"class Solution:\n    def f(self, x):\n        return {ret}\n"


def _test(output: str) -> dict:
    return {"testtype": "functional", "input": "0", "output": output}


# --- floats within tolerance now pass (were false negatives) ----------------


def test_float_within_tolerance_passes():
    assert run_pass_at_1(_fn("2.5000000001"), [_test("2.5")], fn_name="f") is True


def test_nested_float_list_within_tolerance_passes():
    assert run_pass_at_1(
        _fn("[1.5000000001, 2.5]"), [_test("[1.5, 2.5]")], fn_name="f"
    ) is True


def test_tuple_return_matches_list_expected():
    # Same ordered sequence, different container type.
    assert run_pass_at_1(_fn("(0, 1)"), [_test("[0, 1]")], fn_name="f") is True


# --- genuinely-wrong answers must still fail --------------------------------


def test_float_outside_tolerance_fails():
    assert run_pass_at_1(_fn("2.6"), [_test("2.5")], fn_name="f") is False


def test_wrong_list_element_fails():
    assert run_pass_at_1(_fn("[1, 2]"), [_test("[1, 3]")], fn_name="f") is False


def test_length_mismatch_fails():
    assert run_pass_at_1(_fn("[0]"), [_test("[0, 1]")], fn_name="f") is False


# --- bool must not numerically bridge to 1/0 --------------------------------


def test_true_does_not_bridge_to_one():
    assert run_pass_at_1(_fn("True"), [_test("1")], fn_name="f") is False


def test_true_matches_true():
    assert run_pass_at_1(_fn("True"), [_test("true")], fn_name="f") is True


# --- exact answers unaffected -----------------------------------------------


def test_exact_int_unaffected():
    assert run_pass_at_1(_fn("42"), [_test("42")], fn_name="f") is True


def test_exact_string_unaffected():
    assert run_pass_at_1(_fn('"abc"'), [_test('"abc"')], fn_name="f") is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
