"""Offline tests for the multi-agent ensemble (plurality) baseline — SPEC R3.

Uses a 'toy' benchmark (unknown -> answers_agree is exact string match) + an injected
exact-match score_fn for deterministic control, plus real-grader answers_agree checks.
No torch, no network.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis import analyze_ensemble  # re-export check
from trinity.analysis.ensemble import analyze, answers_agree, plurality_answer, render

_REPO = Path(__file__).resolve().parents[1]


def _exact(benchmark, candidate, reference):
    return 1.0 if str(candidate).strip() == str(reference).strip() else 0.0


def _item(answers, correct="X", benchmark="toy"):
    return {"benchmark": benchmark, "correct_answer": correct, "model_answers": answers}


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.analysis.ensemble; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


def test_reexported_from_package():
    assert analyze_ensemble is analyze


# --------------------------------------------------------------------------- #
# answers_agree (real grader)
# --------------------------------------------------------------------------- #
def test_answers_agree_math_equivalent_forms():
    assert answers_agree("math500", "1/2", "0.5") is True     # math_equal
    assert answers_agree("math500", "1/2", "1/3") is False


def test_answers_agree_choice_and_fallback():
    assert answers_agree("mmlu", "B", "B") is True
    assert answers_agree("mmlu", "A", "B") is False
    assert answers_agree("toy", "X", "X") is True             # unknown -> exact text
    assert answers_agree("toy", "X", "Y") is False


def test_answers_agree_resolves_versioned_code_alias():
    # A versioned identity (livecodebench_v6) must grade as CODE, like the scorer
    # does — two models emitting the same code agree even with different prose,
    # instead of falling through to the exact-text default.
    m2 = "Solution:\n```python\ndef solve():\n    return 42\n```"
    m3 = "```python\ndef solve():\n    return 42\n```"        # same code, no prose
    assert answers_agree("livecodebench", m2, m3) is True
    assert answers_agree("livecodebench_v6", m2, m3) is True  # was False (alias unresolved)
    # distinct code still disagrees under the alias.
    m4 = "```python\ndef solve():\n    return 0\n```"
    assert answers_agree("livecodebench_v6", m2, m4) is False


def test_plurality_clusters_same_code_under_versioned_alias():
    # End-to-end: two v6 models with identical code but different prose form the
    # plurality over a lone dissenter, instead of a 3-way no-majority split.
    answers = {
        "m1": "```python\ndef solve():\n    return 0\n```",           # dissenter
        "m2": "Solution:\n```python\ndef solve():\n    return 42\n```",
        "m3": "```python\ndef solve():\n    return 42\n```",
    }
    assert plurality_answer("livecodebench_v6", answers) == answers["m2"]


# --------------------------------------------------------------------------- #
# plurality_answer
# --------------------------------------------------------------------------- #
def test_plurality_picks_majority():
    assert plurality_answer("toy", {"a": "X", "b": "X", "c": "Y"}) == "X"


def test_plurality_tie_breaks_to_earliest_model():
    # all distinct -> 3 singleton clusters -> earliest model 'a' wins deterministically.
    assert plurality_answer("toy", {"a": "X", "b": "Y", "c": "Z"}) == "X"


def test_plurality_skips_empty_and_none():
    assert plurality_answer("toy", {"a": None, "b": "", "c": "Y"}) == "Y"
    assert plurality_answer("toy", {}) is None


# --------------------------------------------------------------------------- #
# analyze — the ensemble beats the best single when models complement
# --------------------------------------------------------------------------- #
def test_ensemble_beats_best_single_via_complementarity():
    items = [
        _item({"a": "X", "b": "X", "c": "Y"}),   # a,b right
        _item({"a": "Y", "b": "X", "c": "X"}),   # b,c right
        _item({"a": "X", "b": "Y", "c": "Z"}),   # only a right; plurality tie -> a's "X"
    ]
    s = analyze(items, benchmark="toy", score_fn=_exact)
    assert s.ensemble_accuracy == pytest.approx(1.0)
    assert s.best_single == pytest.approx(2 / 3) and s.best_single_model == "a"
    assert s.oracle_any == pytest.approx(1.0)
    assert s.ensemble_vs_best_single == pytest.approx(1 / 3)


def test_plurality_can_be_wrong():
    # two models agree on a WRONG answer -> plurality is wrong though a specialist was right.
    s = analyze([_item({"a": "W", "b": "W", "c": "X"})], benchmark="toy", score_fn=_exact)
    assert s.ensemble_accuracy == pytest.approx(0.0)
    assert s.oracle_any == pytest.approx(1.0) and s.best_single == pytest.approx(1.0)


def test_analyze_empty():
    s = analyze([], benchmark="toy", score_fn=_exact)
    assert s.n_questions == 0 and s.best_single_model is None


# --------------------------------------------------------------------------- #
# render + R3 verdict
# --------------------------------------------------------------------------- #
def test_render_r3_verdict():
    s = analyze([_item({"a": "W", "b": "W", "c": "X"})], benchmark="toy", score_fn=_exact)
    md = render(s, trinity_accuracy=0.5)
    assert "ensemble baseline (SPEC R3)" in md and "ensemble (plurality vote)" in md
    assert "**R3** (TRINITY > best multi-agent baseline): ✅ HOLDS (0.500 vs ensemble 0.000)" in md
    # without a TRINITY score, the verdict is deferred, not fabricated.
    assert "must beat this ensemble baseline" in render(s)
    assert render(analyze([], benchmark="toy")).strip().endswith("(no cached-answer items found)_")


# --------------------------------------------------------------------------- #
# Ragged pool: reject (mirrors analysis.agreement) so best_single <= oracle holds
# --------------------------------------------------------------------------- #
def test_ragged_pool_is_rejected():
    # Model 'c' appears only in the last item (where it is right); grading it over
    # 1 item while the oracle spans 3 would give best_single (1.0) > oracle_any.
    items = [
        _item({"a": "X", "b": "W"}),
        _item({"a": "W", "b": "W"}),
        _item({"c": "X"}),
    ]
    with pytest.raises(ValueError, match="ragged pool|expected"):
        analyze(items, benchmark="toy", score_fn=_exact)


def test_best_single_never_exceeds_oracle_on_uniform_pool():
    items = [
        _item({"a": "X", "b": "W", "c": "W"}),
        _item({"a": "W", "b": "X", "c": "W"}),
        _item({"a": "X", "b": "W", "c": "X"}),
    ]
    s = analyze(items, benchmark="toy", score_fn=_exact)
    assert s.best_single <= s.oracle_any + 1e-9
    # Every model is graded over all 3 items (uniform denominators).
    assert len(s.models) == 3


def test_empty_answer_items_do_not_make_pool_ragged():
    # An item with empty model_answers is skipped and must not trip the guard.
    items = [
        _item({"a": "X", "b": "X"}),
        _item({}),
        _item({"a": "X", "b": "W"}),
    ]
    s = analyze(items, benchmark="toy", score_fn=_exact)
    assert s.n_questions == 2
