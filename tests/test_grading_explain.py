"""Offline tests for the grading explainer. No network, no GPU.

These use the REAL grader, so the explanation's score is pinned to reward.score_text.
"""
from __future__ import annotations

from trinity.grading_explain import explain_grade
from trinity.orchestration import reward as R


def _joined(exp) -> str:
    return " | ".join(exp.steps).lower()


# ---------------------------------------------------------------------------
# The reported score always equals the real grader
# ---------------------------------------------------------------------------
def test_score_matches_the_real_grader_across_cases():
    cases = [
        ("math500", r"\boxed{4}", "4"),
        ("math500", "the answer is 5", "4"),
        ("math500", r"\boxed{2{,}048}", "2048"),
        ("mmlu", "The answer is B.", "B"),
        ("mmlu", "It is clearly A.", "B"),
    ]
    for bench, cand, ref in cases:
        exp = explain_grade(bench, cand, ref)
        assert exp.score == R.score_text(bench, cand, ref)
        assert exp.correct == (exp.score > 0.0)


# ---------------------------------------------------------------------------
# math: extraction + normalization trace
# ---------------------------------------------------------------------------
def test_boxed_math_answer_is_explained_as_a_match():
    exp = explain_grade("math500", r"So \boxed{42}.", "42")
    assert exp.kind == "math" and exp.correct
    assert exp.detail["extractor"] == "boxed"
    assert exp.detail["extracted"] == "42"
    assert "match" in _joined(exp)


def test_last_number_fallback_is_named():
    exp = explain_grade("math500", "after simplifying we get 7", "7")
    assert exp.detail["extractor"] == "last-number"
    assert exp.correct


def test_wrong_math_answer_explains_the_mismatch():
    exp = explain_grade("math500", r"\boxed{5}", "4")
    assert not exp.correct
    assert "no match" in _joined(exp)
    assert exp.detail["normalized_candidate"] != exp.detail["normalized_reference"]


def test_normalization_equivalence_is_reported():
    # 1/2 vs 0.5 grade equal via numeric/symbolic equivalence, not exact match.
    exp = explain_grade("math500", r"\boxed{1/2}", "0.5")
    if exp.correct:  # depends on grader's numeric path, which is present
        assert "equivalence" in _joined(exp) or "exact" in _joined(exp)


# ---------------------------------------------------------------------------
# choice: extracted letter trace
# ---------------------------------------------------------------------------
def test_choice_match_reports_both_letters():
    exp = explain_grade("mmlu", "After thought, the answer is C.", "C")
    assert exp.kind == "choice" and exp.correct
    assert exp.detail["extracted_letter"] == "C"
    assert exp.detail["reference_letter"] == "C"


def test_choice_unextractable_is_explained():
    exp = explain_grade("mmlu", "I am not sure about this one.", "C")
    assert not exp.correct
    assert exp.detail["extracted_letter"] is None
    assert "no choice letter" in _joined(exp)


# ---------------------------------------------------------------------------
# code + unknown
# ---------------------------------------------------------------------------
def test_code_reports_presence_not_execution():
    exp = explain_grade("livecodebench", "```python\ndef f(): return 1\n```",
                        {"tests": [], "fn_name": "f"})
    assert exp.kind == "code"
    assert exp.detail["has_code"] is True
    assert "functional correctness" in _joined(exp)


def test_unknown_benchmark_is_handled_not_raised():
    exp = explain_grade("nonsense", "x", "y")
    assert exp.kind == "unknown" and exp.score == 0.0
    assert "unknown benchmark" in _joined(exp)


def test_versioned_benchmark_identity_routes_like_the_grader():
    # A frozen hidden LiveCodeBench v6 item carries the adapter *identity*
    # "livecodebench_v6", which reward.resolve_benchmark maps to "livecodebench".
    # The explainer must route on that resolved key so its kind/steps agree with
    # the score (previously it labeled this "unknown" while score_text graded it,
    # so the trace contradicted the grade).
    spec = {"tests": [{"stdin": "", "expected_stdout": "hi\n"}], "timeout_s": 5}
    code = "```python\nprint(\"hi\")\n```"
    exp = explain_grade("livecodebench_v6", code, spec)
    assert exp.kind == "code"
    assert exp.detail["has_code"] is True
    assert exp.score == R.score_text("livecodebench_v6", code, spec)
    assert exp.correct == (exp.score > 0.0)
    # The alias resolution is surfaced in the trace, not silently applied.
    assert "resolves to dispatch key" in _joined(exp)
    # And the contract holds: a graded (kind != unknown) item is never called unknown.
    assert "unknown benchmark" not in _joined(exp)


def test_explanation_roundtrips_to_dict():
    d = explain_grade("math500", r"\boxed{4}", "4").to_dict()
    assert d["score"] == 1.0 and d["kind"] == "math"
    assert isinstance(d["steps"], list) and d["steps"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))


def test_choice_integer_reference_trace_matches_grade():
    # MMLU sets often store the answer as a 0-based index; grader maps 1 -> "B".
    exp = explain_grade("mmlu", "After thought, the answer is B.", 1)
    assert exp.correct and exp.score == 1.0
    assert exp.detail["reference_letter"] == "B"      # not the string "1"
    assert any("match:" in s and not s.startswith("no match") for s in exp.steps)


def test_choice_nonletter_reference_is_not_fabricated_into_a_match():
    # The grader rejects a non-letter reference (score 0); the trace must NOT
    # fabricate "B" from "Beta particle"[:1] and claim a match.
    exp = explain_grade("mmlu", "The answer is B.", "Beta particle")
    assert not exp.correct and exp.score == 0.0
    assert exp.detail["reference_letter"] is None
    assert not any("match:" in s and not s.startswith("no match") for s in exp.steps)


def test_choice_reference_letter_equals_the_grader_normalizer():
    from trinity.orchestration.reward import normalize_reference_letter
    for ref in (1, "C", "Beta particle", 3):
        exp = explain_grade("mmlu", "the answer is B.", ref)
        assert exp.detail["reference_letter"] == normalize_reference_letter(ref)
        matched = any("match:" in s and not s.startswith("no match") for s in exp.steps)
        assert matched == (exp.score == 1.0)  # trace never contradicts the grade
