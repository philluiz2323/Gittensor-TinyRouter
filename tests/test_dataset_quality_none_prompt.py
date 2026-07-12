"""Regression tests: a None question_text must count as a missing prompt (issue #225).

The existing suite only exercises an empty-string / absent prompt. A present
``question_text: None`` used to slip through because ``str(None)`` is the truthy
``"None"``, hiding the missing prompt and colliding None items into a spurious
duplicate. Pure Python — no network, no GPU, no torch.
"""
from __future__ import annotations

from trinity.dataset_quality import audit_dataset


def _item(qid, text, gold="4", benchmark="m"):
    return {"question_id": qid, "question_text": text,
            "correct_answer": gold, "benchmark": benchmark}


def test_none_prompt_counts_as_missing_not_duplicate():
    report = audit_dataset([_item("a", None), _item("b", None)])
    q = report.per_benchmark["m"]
    assert q.missing_prompt == 2
    assert q.duplicate_questions == 0
    assert not q.ok


def test_none_prompt_matches_empty_string_behaviour():
    none_q = audit_dataset([_item("a", None)]).per_benchmark["m"]
    empty_q = audit_dataset([_item("a", "")]).per_benchmark["m"]
    assert none_q.missing_prompt == empty_q.missing_prompt == 1
    assert none_q.duplicate_questions == empty_q.duplicate_questions == 0


def test_real_duplicate_text_still_flagged_alongside_none():
    # A genuine duplicate is still caught; None items don't inflate the count.
    report = audit_dataset([
        _item("a", "1+1?"), _item("b", "1+1?"),   # real duplicate
        _item("c", None), _item("d", None),        # two missing prompts
    ])
    q = report.per_benchmark["m"]
    assert q.duplicate_questions == 1
    assert q.missing_prompt == 2


def test_none_prompt_missing_answer_still_independent():
    # None prompt AND None answer: both defects counted on their own axes.
    report = audit_dataset([{"question_id": "a", "question_text": None,
                             "correct_answer": None, "benchmark": "m"}])
    q = report.per_benchmark["m"]
    assert q.missing_prompt == 1
    assert q.missing_answer == 1
