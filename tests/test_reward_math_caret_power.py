"""Caret ``^`` is exponentiation in math answers, not bitwise XOR (issue #342).

``_sympy_equal`` previously omitted sympy's ``convert_xor`` transform, so
``2^6`` parsed as ``2 XOR 6 = 4`` — a false-positive grade against reference
``4``, and a false-negative against ``64``.

Pure / offline — no torch, no network. Requires sympy (base dependency).
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import math_equal, score_text

pytest.importorskip("sympy")


def test_caret_power_false_positives_rejected():
    # XOR coincidences that must NOT grade equal (powers ≠ XOR results).
    assert math_equal("2^6", "4") is False   # 64 vs 4 (was True: 2 XOR 6)
    assert math_equal("5^1", "4") is False   # 5 vs 4 (was True: 5 XOR 1)


def test_caret_power_true_positives_accepted():
    assert math_equal("2^3", "8") is True
    assert math_equal("2^6", "64") is True
    assert math_equal("10^6", "1000000") is True


def test_boxed_caret_power_end_to_end():
    assert score_text("math500", r"\boxed{2^6}", "4") == 0.0
    assert score_text("math500", r"\boxed{2^3}", "8") == 1.0
    assert score_text("math500", r"\boxed{2^6}", "64") == 1.0
