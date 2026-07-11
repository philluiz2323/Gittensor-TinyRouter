"""Offline coverage for the pure helpers in adapters/swebench.py.

`swebench.py` had 70% coverage: the HuggingFace row parser, the FAIL_TO_PASS
list coercion, patch normalization, the placeholder exact-match scorer, and
several adapter methods were unexercised. All of these are pure (no torch, no
network — `datasets` is imported lazily), so they test offline directly.
"""
from __future__ import annotations

import json
import sys
import types

from trinity.adapters import swebench as sb
from trinity.adapters.base import ScoringMode, TaskType


# --------------------------------------------------------------------------- #
# PatchReference
# --------------------------------------------------------------------------- #
def test_patch_reference_round_trips_through_dict():
    ref = sb.PatchReference(
        repo="octo/calc", base_commit="abc", gold_patch="diff", test_patch="t",
        fail_to_pass=["a::b"], pass_to_pass=["c::d"],
        environment_setup_commit="env", version="1.0",
    )
    assert sb.PatchReference.from_dict(ref.to_dict()) == ref


def test_patch_reference_from_dict_defaults_missing_fields():
    ref = sb.PatchReference.from_dict({"repo": "r", "base_commit": "b"})
    assert ref.gold_patch == "" and ref.fail_to_pass == [] and ref.version is None


def test_is_valid_requires_repo_commit_and_a_test():
    assert sb.PatchReference("r", "b", "g", fail_to_pass=["t"]).is_valid()
    assert not sb.PatchReference("", "b", "g", fail_to_pass=["t"]).is_valid()
    assert not sb.PatchReference("r", "", "g", fail_to_pass=["t"]).is_valid()
    assert not sb.PatchReference("r", "b", "g", fail_to_pass=[]).is_valid()


# --------------------------------------------------------------------------- #
# _as_list
# --------------------------------------------------------------------------- #
def test_as_list_none_is_empty():
    assert sb._as_list(None) == []


def test_as_list_json_string():
    assert sb._as_list(json.dumps(["a", "b"])) == ["a", "b"]


def test_as_list_already_a_list_is_stringified():
    assert sb._as_list([1, 2]) == ["1", "2"]


def test_as_list_tuple():
    assert sb._as_list(("x", "y")) == ["x", "y"]


def test_as_list_bad_json_nonempty_string_wraps():
    assert sb._as_list("not json {") == ["not json {"]


def test_as_list_bad_json_empty_string_is_empty():
    assert sb._as_list("") == []


def test_as_list_unexpected_type_is_empty():
    assert sb._as_list(42) == []


# --------------------------------------------------------------------------- #
# build_patch_prompt
# --------------------------------------------------------------------------- #
def test_build_patch_prompt_includes_repo_commit_and_issue():
    p = sb.build_patch_prompt("Fix the bug.", "octo/calc", "deadbeef")
    assert "octo/calc" in p and "deadbeef" in p
    assert "Fix the bug." in p
    assert "unified diff" in p


def test_build_patch_prompt_includes_hints_only_when_present():
    assert "## Hints" not in sb.build_patch_prompt("x", "r", "c")
    assert "## Hints" in sb.build_patch_prompt("x", "r", "c", hints="use git")


# --------------------------------------------------------------------------- #
# normalize_patch + score_patch
# --------------------------------------------------------------------------- #
def test_normalize_patch_empty():
    assert sb.normalize_patch("") == ""


def test_normalize_patch_strips_fences_index_and_hunk_headers():
    raw = (
        "```diff\n"
        "diff --git a/f b/f\n"
        "index 1234abc..5678def 100644\n"
        "@@ -1,3 +1,3 @@ def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "\n"
        "```"
    )
    out = sb.normalize_patch(raw)
    assert "index 1234abc" not in out
    assert "@@" not in out
    assert "```" not in out
    assert "-    return 1" in out and "+    return 2" in out
    assert "" not in out.split("\n")  # no blank lines


def test_score_patch_exact_normalized_match():
    gold = "diff --git a/f b/f\n@@ -1 +1 @@\n-x\n+y\n"
    cand = "```\ndiff --git a/f b/f\nindex aaa..bbb\n@@ -9 +9 @@ ctx\n-x\n+y\n```"
    assert sb.score_patch(cand, {"gold_patch": gold}) == 1.0


def test_score_patch_mismatch_is_zero():
    assert sb.score_patch("-x\n+y", {"gold_patch": "-x\n+z"}) == 0.0


def test_score_patch_no_gold_or_bad_reference_is_zero():
    assert sb.score_patch("anything", {"gold_patch": ""}) == 0.0
    assert sb.score_patch("anything", {}) == 0.0
    assert sb.score_patch("anything", "not-a-dict") == 0.0


# --------------------------------------------------------------------------- #
# _row_get
# --------------------------------------------------------------------------- #
def test_row_get_first_present_then_default():
    assert sb._row_get({"b": 1}, "a", "b") == 1
    assert sb._row_get({}, "a", default="d") == "d"
    assert sb._row_get(7, "a", default="d") == "d"  # non-mapping -> TypeError guard


# --------------------------------------------------------------------------- #
# _hf_swebench (via a faked datasets module)
# --------------------------------------------------------------------------- #
def _fake_datasets(monkeypatch, handler):
    module = types.ModuleType("datasets")
    module.load_dataset = handler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_hf_swebench_parses_rows(monkeypatch):
    rows = [
        {
            "instance_id": "octo__calc-1", "problem_statement": "add is wrong",
            "repo": "octo/calc", "base_commit": "c0", "patch": "GOLD",
            "test_patch": "TP", "FAIL_TO_PASS": json.dumps(["t::a"]),
            "PASS_TO_PASS": ["t::b"], "version": "1.0", "hints_text": "hint",
        },
        {"problem_statement": "", "repo": "x"},        # no problem -> skip
        {"problem_statement": "p", "repo": ""},        # no repo    -> skip
    ]
    _fake_datasets(monkeypatch, lambda path, split=None: rows)

    tasks = sb._hf_swebench("test")
    assert len(tasks) == 1
    t = tasks[0]
    assert t.task_id == "octo__calc-1"
    assert t.benchmark == sb.BENCHMARK
    assert t.answer["gold_patch"] == "GOLD"
    assert t.answer["fail_to_pass"] == ["t::a"]
    assert t.answer["pass_to_pass"] == ["t::b"]
    assert t.meta["repo"] == "octo/calc"
    assert "octo/calc" in t.prompt  # build_patch_prompt was applied


def test_hf_swebench_none_when_datasets_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    assert sb._hf_swebench("test") is None


def test_hf_swebench_none_on_load_error(monkeypatch):
    def boom(path, split=None):
        raise RuntimeError("gated")

    _fake_datasets(monkeypatch, boom)
    assert sb._hf_swebench("test") is None


def test_hf_swebench_all_rows_skipped_returns_none(monkeypatch):
    _fake_datasets(monkeypatch, lambda path, split=None: [{"problem_statement": ""}])
    assert sb._hf_swebench("test") is None


# --------------------------------------------------------------------------- #
# load_swebench_tasks
# --------------------------------------------------------------------------- #
def test_load_falls_back_to_toy_set_offline(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    tasks = sb.load_swebench_tasks("test", max_items=None, seed=0)
    assert len(tasks) == 1
    assert tasks[0].benchmark == sb.BENCHMARK
    assert tasks[0].answer["repo"] == "octo/calc"


def test_load_shuffle_and_truncate_are_deterministic(monkeypatch):
    # 20 upstream rows: the logical "test" split keeps the ~25% holdout (>= 3),
    # so max_items=3 still truncates to exactly 3. See the SWE-bench holdout carve
    # (single-split policy, mirrors GPQA #95).
    rows = [
        {"instance_id": f"i{i}", "problem_statement": "p", "repo": "r", "base_commit": "c"}
        for i in range(20)
    ]
    _fake_datasets(monkeypatch, lambda path, split=None: rows)
    a = sb.load_swebench_tasks("test", max_items=3, seed=1)
    _fake_datasets(monkeypatch, lambda path, split=None: rows)
    b = sb.load_swebench_tasks("test", max_items=3, seed=1)
    assert len(a) == 3
    assert [t.task_id for t in a] == [t.task_id for t in b]


# --------------------------------------------------------------------------- #
# SweBenchAdapter methods
# --------------------------------------------------------------------------- #
def _toy_task():
    return sb._toy_swebench()[0]


def test_adapter_metadata_methods():
    adapter = sb.SweBenchAdapter()
    task = _toy_task()
    assert adapter.build_prompt(task) == task.prompt
    assert adapter.task_type() is TaskType.PATCH
    assert adapter.scoring_modes() == frozenset({ScoringMode.CACHED, ScoringMode.EXECUTION})


def test_adapter_serialize_task_shape():
    adapter = sb.SweBenchAdapter()
    task = _toy_task()
    s = adapter.serialize_task(task)
    assert s["task_id"] == task.task_id
    assert s["reference"] == task.answer
    assert s["task_type"] == TaskType.PATCH.value
    assert s["meta"] == dict(task.meta)


def test_adapter_score_output_uses_cached_exact_match_without_a_provider():
    adapter = sb.SweBenchAdapter()  # no repo_provider
    task = _toy_task()
    assert adapter.score_output(task.answer["gold_patch"], task.answer) == 1.0
    assert adapter.score_output("wrong patch", task.answer) == 0.0


def test_adapter_score_execution_uses_a_callable_context():
    adapter = sb.SweBenchAdapter()
    assert adapter.score_execution("out", {"gold_patch": "g"}, context=lambda o, r: 1.0) == 1.0
    # A None result and a non-callable context both fall back to None.
    assert adapter.score_execution("out", {}, context=lambda o, r: None) is None
    assert adapter.score_execution("out", {}, context=None) is None


def test_adapter_score_trajectory_grades_the_final_answer():
    adapter = sb.SweBenchAdapter()
    task = _toy_task()

    class _Traj:
        final_answer = task.answer["gold_patch"]

    _Traj.task = task
    assert adapter.score_trajectory(_Traj()) == 1.0
