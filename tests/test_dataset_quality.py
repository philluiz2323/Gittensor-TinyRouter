"""Offline tests for the benchmark data-quality audit. No network, no GPU."""
from __future__ import annotations

from trinity.dataset_quality import audit_dataset, normalize_question


def _item(qid, text="q?", gold="4", benchmark="math500"):
    return {"question_id": qid, "question_text": text,
            "correct_answer": gold, "benchmark": benchmark}


def test_a_clean_dataset_has_no_problems():
    report = audit_dataset([_item("a", "1+1?"), _item("b", "2+2?")])
    assert report.ok and report.n_problems == 0
    assert report.n_items == 2
    assert report.per_benchmark["math500"].ok


# ---------------------------------------------------------------------------
# duplicate ids
# ---------------------------------------------------------------------------
def test_duplicate_ids_are_flagged():
    report = audit_dataset([_item("dup", "x"), _item("dup", "y"), _item("ok", "z")])
    q = report.per_benchmark["math500"]
    assert q.duplicate_ids == ["dup"]
    assert not q.ok and not report.ok


def test_same_id_across_benchmarks_is_not_a_collision():
    report = audit_dataset([_item("q0", benchmark="math500"), _item("q0", benchmark="mmlu")])
    assert report.ok
    assert report.per_benchmark["math500"].duplicate_ids == []
    assert report.per_benchmark["mmlu"].duplicate_ids == []


# ---------------------------------------------------------------------------
# duplicate questions
# ---------------------------------------------------------------------------
def test_duplicate_question_text_is_flagged_after_normalization():
    report = audit_dataset([
        _item("a", "What is 2+2?"),
        _item("b", "  what is 2+2? "),   # same after normalization
        _item("c", "different"),
    ])
    assert report.per_benchmark["math500"].duplicate_questions == 1


def test_triplicate_question_counts_two_extras():
    report = audit_dataset([_item("a", "same"), _item("b", "same"), _item("c", "same")])
    assert report.per_benchmark["math500"].duplicate_questions == 2


def test_normalize_question_collapses_case_and_whitespace():
    assert normalize_question("  Hello   World ") == "hello world"


# ---------------------------------------------------------------------------
# missing fields
# ---------------------------------------------------------------------------
def test_missing_prompt_and_answer_are_flagged():
    report = audit_dataset([
        _item("a", text="", gold="4"),      # missing prompt
        _item("b", text="q", gold=""),      # missing answer
        _item("c", text="q", gold=None),    # missing answer (None)
    ])
    q = report.per_benchmark["math500"]
    assert q.missing_prompt == 1
    assert q.missing_answer == 2


def test_blank_questions_are_not_counted_as_duplicates():
    # Two blank prompts are missing_prompt, not a duplicate-question pair.
    report = audit_dataset([_item("a", text=""), _item("b", text="")])
    q = report.per_benchmark["math500"]
    assert q.missing_prompt == 2
    assert q.duplicate_questions == 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def test_report_aggregates_across_benchmarks():
    report = audit_dataset([
        _item("a", "x", benchmark="math500"),
        _item("a", "y", benchmark="math500"),   # dup id in math500
        _item("z", "w", benchmark="mmlu"),
    ])
    assert report.n_items == 3
    assert set(report.per_benchmark) == {"math500", "mmlu"}
    assert report.n_problems == 1
    assert not report.ok


def test_empty_dataset_is_ok():
    report = audit_dataset([])
    assert report.ok and report.n_items == 0
    assert report.to_dict()["per_benchmark"] == {}


def test_report_to_dict_shape():
    d = audit_dataset([_item("a")]).to_dict()
    assert d["n_items"] == 1 and d["ok"] is True
    assert "math500" in d["per_benchmark"]
    assert d["per_benchmark"]["math500"]["n_items"] == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
