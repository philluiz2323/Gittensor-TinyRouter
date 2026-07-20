"""format_hint must resolve benchmark aliases like the grader (issue #389).

``livecodebench_v6`` items must get the code-fence hint, not the math
``\\boxed{}`` default — otherwise Fugu workers emit ungradeable answers.
"""
from __future__ import annotations

from trinity.fugu.workflow import format_hint


def test_livecodebench_v6_gets_code_hint_not_math():
    v6 = format_hint("livecodebench_v6")
    canon = format_hint("livecodebench")
    assert v6 == canon
    assert "Python code block" in v6
    assert "boxed" not in v6.lower()


def test_canonical_hints_unchanged():
    assert "Answer: X" in format_hint("mmlu")
    assert "boxed" in format_hint("math500").lower()
