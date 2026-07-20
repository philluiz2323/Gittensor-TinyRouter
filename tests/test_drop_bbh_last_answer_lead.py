"""Regression: DROP/BBH answer extraction must read the LAST answer lead.

A chain-of-thought that contains its own "the answer is ..." phrasing and then
commits the real answer as a trailing "Answer: X" must be graded on X. The prior
`_ANSWER_LEAD` used a greedy `(.+)` under DOTALL, so `finditer` collapsed every
lead into a single first-lead match and `matches[-1]` still read the reasoning —
scoring a correct answer 0.0 (the bug #340 intended, but did not, fix). No torch.
"""
from __future__ import annotations

from trinity.adapters.bbh import _final_answer_segment as bbh_seg
from trinity.adapters.bbh import score_bbh
from trinity.adapters.drop import _final_answer_segment as drop_seg
from trinity.adapters.drop import score_drop


def _bbh(gold, atype="exact_match"):
    return {"answer": gold, "answer_type": atype}


# --------------------------------------------------------------------------- #
# the bug: chain-of-thought "answer is" before the committed final answer
# --------------------------------------------------------------------------- #
def test_bbh_takes_last_lead_over_chain_of_thought():
    out = "Let me reason. The answer is tricky here.\nAnswer: True"
    assert bbh_seg(out) == "True"
    assert score_bbh(out, _bbh("True")) == 1.0


def test_drop_takes_last_lead_over_chain_of_thought():
    out = "Working it out: the answer is not obvious.\nAnswer: 21"
    assert drop_seg(out) == "21"
    assert score_drop(out, {"gold_answers": ["21"]}) == 1.0


def test_bbh_takes_last_of_two_real_answer_leads():
    # A self-correction: the final "Answer:" wins.
    out = "Answer: 3\nOn reflection that is wrong. Answer: 7"
    assert bbh_seg(out) == "7"
    assert score_bbh(out, _bbh("7")) == 1.0


# --------------------------------------------------------------------------- #
# a trailing "the answer is:" fragment with no content must not win
# --------------------------------------------------------------------------- #
def test_bbh_skips_trailing_empty_lead():
    out = "Answer: 4\nIn summary, the answer is:"
    assert bbh_seg(out) == "4"
    assert score_bbh(out, _bbh("4")) == 1.0


def test_drop_skips_trailing_empty_lead():
    out = "Answer: 12\nThat is my final answer is:"
    assert drop_seg(out) == "12"
    assert score_drop(out, {"gold_answers": ["12"]}) == 1.0


# --------------------------------------------------------------------------- #
# no behavioural change for the ordinary cases
# --------------------------------------------------------------------------- #
def test_plain_answer_and_no_lead_unchanged():
    assert bbh_seg("Answer: True") == "True"
    assert bbh_seg("The answer is: 42") == "42"
    assert bbh_seg("just 5") == "just 5"          # no lead -> last non-empty line
    assert score_bbh("Answer: False", _bbh("True")) == 0.0  # wrong stays wrong


def test_multiple_choice_path_unaffected():
    # MC routes through extract_choice_letter on the raw candidate, not the segment.
    assert score_bbh("I think the answer is (B).", _bbh("(B)", "multiple_choice")) == 1.0
    assert score_bbh("Answer: (A)", _bbh("(B)", "multiple_choice")) == 0.0
