"""Offline unit tests for verifier verdict parsing and transcript post-processing.

These pure helpers sit on the live orchestration hot path (``session.py``) but had
no dedicated pytest coverage. Pinning their edge cases prevents silent regressions
in verdict extraction and head+tail truncation.
"""
from __future__ import annotations

from trinity.roles import postprocess, verifier
from trinity.roles.postprocess import ELISION_MARKER
from trinity.types import Role


# --- parse_verdict ---


def test_parse_verdict_returns_none_for_empty_input():
    assert verifier.parse_verdict("") is None
    assert verifier.parse_verdict("   ") is None


def test_parse_verdict_accepts_case_insensitive_last_match():
    text = "VERDICT: REVISE\nSome discussion.\nverdict: accept"
    assert verifier.parse_verdict(text) == "ACCEPT"


def test_parse_verdict_returns_none_when_absent():
    assert verifier.parse_verdict("Looks fine but no verdict line.") is None


# --- extract_diagnosis ---


def test_extract_diagnosis_returns_full_text_without_verdict():
    text = "  Needs more work on step 2.  "
    assert verifier.extract_diagnosis(text) == "Needs more work on step 2."


def test_extract_diagnosis_strips_text_before_last_verdict():
    text = "First pass.\nVERDICT: REVISE\nReworked.\nVERDICT: ACCEPT"
    assert verifier.extract_diagnosis(text) == "First pass.\nVERDICT: REVISE\nReworked."


def test_extract_diagnosis_empty_for_blank_input():
    assert verifier.extract_diagnosis("") == ""


# --- postprocess ---


def test_postprocess_passthrough_when_under_budget():
    raw = "  short answer  "
    assert postprocess.postprocess(raw, Role.WORKER, max_chars=100) == "short answer"


def test_postprocess_passthrough_when_max_chars_non_positive():
    raw = "x" * 50
    assert postprocess.postprocess(raw, Role.WORKER, max_chars=0) == raw
    assert postprocess.postprocess(raw, Role.VERIFIER, max_chars=-1) == raw


def test_postprocess_head_tail_truncation_preserves_verdict_line():
    prefix = "A" * 40
    verdict = "VERDICT: ACCEPT"
    raw = f"{prefix}{'B' * 200}{verdict}"
    out = postprocess.postprocess(raw, Role.VERIFIER, max_chars=80)
    assert ELISION_MARKER in out
    assert out.endswith(verdict)
    assert out.startswith("A")
    assert len(out) <= 80


def test_postprocess_hard_truncates_when_budget_smaller_than_marker():
    raw = "abcdefghijklmnop"
    out = postprocess.postprocess(raw, Role.THINKER, max_chars=5)
    assert out == "abcde"
    assert ELISION_MARKER not in out
