"""LiveCodeBench 'functional' (LeetCode-style) tests are scored by calling the
entry point, not by running the candidate as a stdin/stdout program.

Before this, every functional test dict ``{"input", "output"}`` was executed as
a stdin/stdout test, so a correct function-call solution produced no stdout and
scored 0. These tests exercise the call-based harness end-to-end (real
subprocess sandbox) and confirm the stdin path is unchanged.

Pure / offline — pass@1 runs candidate code in the same sandboxed subprocess the
reward checker uses; no network, no GPU.
"""
from __future__ import annotations

from trinity.orchestration.reward import (
    _coerce_test_spec,
    _parse_functional_args,
    run_pass_at_1,
    score_text,
)

# --- LeetCode-style solutions (Solution class method) -----------------------
_SOLUTION_TWO_SUM = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, n in enumerate(nums):\n"
    "            if target - n in seen:\n"
    "                return [seen[target - n], i]\n"
    "            seen[n] = i\n"
    "        return []\n"
)
_TWO_SUM_SPEC = {
    "fn_name": "twoSum",
    "tests": [
        {"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]", "testtype": "functional"},
        {"input": "[3, 2, 4]\n6", "output": "[1, 2]", "testtype": "functional"},
    ],
}

# --- bare top-level function ------------------------------------------------
_BARE_ADD = "def add(a, b):\n    return a + b\n"
_ADD_SPEC = {
    "fn_name": "add",
    "tests": [{"input": "2\n3", "output": "5", "testtype": "functional"}],
}


def test_functional_solution_method_scores_correct():
    assert run_pass_at_1(_SOLUTION_TWO_SUM, _TWO_SUM_SPEC["tests"],
                         fn_name="twoSum") is True


def test_functional_wrong_answer_fails():
    wrong = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        return [0, 0]\n"
    )
    assert run_pass_at_1(wrong, _TWO_SUM_SPEC["tests"], fn_name="twoSum") is False


def test_functional_bare_function_scores_correct():
    assert run_pass_at_1(_BARE_ADD, _ADD_SPEC["tests"], fn_name="add") is True


def test_score_text_functional_end_to_end():
    good = f"Here is my solution:\n```python\n{_SOLUTION_TWO_SUM}```"
    bad = "```python\nclass Solution:\n    def twoSum(self, nums, target):\n        return []\n```"
    assert score_text("livecodebench", good, _TWO_SUM_SPEC) == 1.0
    assert score_text("livecodebench", bad, _TWO_SUM_SPEC) == 0.0


def test_missing_entry_point_fails_cleanly():
    # fn_name that the candidate does not define -> fail, not crash.
    assert run_pass_at_1(_BARE_ADD, _ADD_SPEC["tests"], fn_name="nonexistent") is False


# --- regressions: stdin path and spec coercion ------------------------------
def test_stdin_tests_still_work_without_fn_name():
    code = "n = int(input())\nprint(n * n)\n"
    stdin_tests = [{"input": "3\n", "output": "9"}, {"input": "5\n", "output": "25"}]
    assert run_pass_at_1(code, stdin_tests) is True
    bad = "n = int(input())\nprint(n + 1)\n"
    assert run_pass_at_1(bad, stdin_tests) is False


def test_functional_type_without_fn_name_falls_back_to_stdin():
    # A functional-typed case with no fn_name must not crash; it degrades to the
    # stdin path (returns whatever that comparison yields), never an exception.
    code = "print('x')\n"
    tests = [{"input": "", "output": "x", "testtype": "functional"}]
    assert run_pass_at_1(code, tests) is True


def test_coerce_test_spec_returns_fn_name():
    tests, timeout_s, fn_name = _coerce_test_spec(_TWO_SUM_SPEC)
    assert fn_name == "twoSum"
    assert timeout_s == 10
    assert len(tests) == 2
    # A bare list spec has no fn_name.
    _, _, none_fn = _coerce_test_spec([{"input": "1\n", "output": "1"}])
    assert none_fn is None


def test_parse_functional_args_one_value_per_line():
    assert _parse_functional_args("[2, 7, 11, 15]\n9") == [[2, 7, 11, 15], 9]
    assert _parse_functional_args('"abc"\n3') == ["abc", 3]
    assert _parse_functional_args("") == []


def test_mixed_stdin_and_functional_cases_in_one_spec():
    # A solution that works as BOTH a callable and a stdin program passes a spec
    # that mixes the two test flavors.
    code = (
        "import sys\n"
        "def solve(a, b):\n"
        "    return a + b\n"
        "if __name__ == '__main__':\n"
        "    data = sys.stdin.read().split()\n"
        "    if data:\n"
        "        print(solve(int(data[0]), int(data[1])))\n"
    )
    spec = {
        "fn_name": "solve",
        "tests": [
            {"input": "2\n3", "output": "5", "testtype": "functional"},
            {"input": "4 5\n", "output": "9", "testtype": "stdin"},
        ],
    }
    assert score_text("livecodebench", f"```python\n{code}```", spec) == 1.0


def test_functional_handles_nested_and_string_args():
    code = (
        "class Solution:\n"
        "    def joinRows(self, grid, sep):\n"
        "        return sep.join(''.join(map(str, r)) for r in grid)\n"
    )
    spec = {
        "fn_name": "joinRows",
        "tests": [
            {"input": '[[1, 2], [3, 4]]\n"-"', "output": '"12-34"',
             "testtype": "functional"},
        ],
    }
    assert run_pass_at_1(code, spec["tests"], fn_name="joinRows") is True


# --- loaders: testtype is carried and func_name resolves from metadata ------
def test_parse_lcb_tests_carries_testtype():
    from trinity.adapters.loaders import _parse_lcb_tests

    row = {
        "public_test_cases": [
            {"input": "1\n", "output": "1", "testtype": "stdin"},
            {"input": "[1, 2]\n", "output": "3", "testtype": "functional"},
        ]
    }
    tests = _parse_lcb_tests(row)
    assert [t["testtype"] for t in tests] == ["stdin", "functional"]
    assert tests[1]["input"] == "[1, 2]\n"


def test_lcb_fn_name_prefers_top_level_then_metadata():
    import json

    from trinity.adapters.loaders import _lcb_fn_name

    assert _lcb_fn_name({"fn_name": "twoSum"}) == "twoSum"
    assert _lcb_fn_name({"func_name": "maxProfit"}) == "maxProfit"
    # Falls back to func_name nested in the JSON metadata blob.
    assert _lcb_fn_name({"metadata": json.dumps({"func_name": "lengthOfLIS"})}) == \
        "lengthOfLIS"
    # A stdin-style row (no entry point) resolves to None.
    assert _lcb_fn_name({"metadata": json.dumps({"difficulty": "easy"})}) is None
    assert _lcb_fn_name({}) is None
