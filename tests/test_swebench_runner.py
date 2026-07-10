"""Sandboxed SWE-bench patch evaluator (issue #18).

Exercises the real grading mechanism — ``git apply`` + a subprocess pytest run —
against a throwaway local git repo built in a tempdir. No network and no
SWE-bench download: :func:`prepare_repo` (the network seam) is not called; the
test injects its own prepared work-tree.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from trinity.adapters.swebench import SweBenchAdapter
from trinity.adapters.swebench_runner import (
    PatchEvalResult,
    evaluate_patch,
    extract_patch,
    score_swebench,
)


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


@contextmanager
def _repo():
    """A tiny git repo whose `add` is buggy (a-b); yields (dir, gold, wrong, ref)."""
    d = tempfile.mkdtemp(prefix="swebench-test-")
    try:
        _run(["git", "init", "-q"], d)
        _run(["git", "config", "user.email", "t@t"], d)
        _run(["git", "config", "user.name", "t"], d)
        buggy = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
        (Path(d) / "calc.py").write_text(buggy)
        (Path(d) / "test_calc.py").write_text(
            "from calc import add, sub\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n\n"
            "def test_sub():\n    assert sub(10, 4) == 6\n"
        )
        _run(["git", "add", "-A"], d)
        _run(["git", "commit", "-q", "-m", "init"], d)
        base = _run(["git", "rev-parse", "HEAD"], d).stdout.strip()

        # Gold patch: fix `add` to a + b (derived via git diff so it applies cleanly).
        (Path(d) / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
        gold = _run(["git", "diff"], d).stdout
        _run(["git", "checkout", "--", "calc.py"], d)

        # Wrong patch: applies (adds a comment) but does not fix the bug.
        (Path(d) / "calc.py").write_text("# helper module\n" + buggy)
        wrong = _run(["git", "diff"], d).stdout
        _run(["git", "checkout", "--", "calc.py"], d)

        ref = {
            "repo": "octo/calc",
            "base_commit": base,
            "gold_patch": gold,
            "fail_to_pass": ["test_calc.py::test_add"],
            "pass_to_pass": ["test_calc.py::test_sub"],
        }
        yield d, gold, wrong, ref
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _reset(d, base):
    _run(["git", "-C", d, "reset", "--hard", "-q", base], ".")
    _run(["git", "-C", d, "clean", "-fdq"], ".")


# --------------------------------------------------------------------------- #
# extract_patch
# --------------------------------------------------------------------------- #
def test_extract_patch_from_fence():
    text = "Here is the fix:\n```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@\n-1\n+2\n```\nDone."
    out = extract_patch(text)
    assert out.startswith("diff --git a/x b/x")
    assert "prose" not in out and "Done." not in out


def test_extract_patch_bare_and_none():
    assert extract_patch("diff --git a/x b/x\n--- a/x\n+++ b/x\n").startswith("diff --git")
    assert extract_patch("just some prose, no diff here") == ""
    assert extract_patch("") == ""


# --------------------------------------------------------------------------- #
# evaluate_patch (real git apply + subprocess pytest)
# --------------------------------------------------------------------------- #
def test_gold_patch_resolves():
    with _repo() as (d, gold, _wrong, ref):
        _reset(d, ref["base_commit"])
        res = evaluate_patch(d, gold, ref)
        assert isinstance(res, PatchEvalResult)
        assert res.passed is True
        assert res.reason == "resolved"
        assert res.reward == 1.0


def test_wrong_patch_applies_but_fails_tests():
    with _repo() as (d, _gold, wrong, ref):
        _reset(d, ref["base_commit"])
        res = evaluate_patch(d, wrong, ref)
        assert res.applied is True          # the comment patch applies...
        assert res.passed is False          # ...but FAIL_TO_PASS still fails
        assert res.reason == "tests_failed"


def test_non_applying_patch_is_clean_failure():
    with _repo() as (d, _gold, _wrong, ref):
        _reset(d, ref["base_commit"])
        bogus = "diff --git a/nope.py b/nope.py\n--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n"
        res = evaluate_patch(d, bogus, ref)
        assert res.passed is False
        assert res.reason == "patch_did_not_apply"


def test_no_patch_found():
    with _repo() as (d, _gold, _wrong, ref):
        res = evaluate_patch(d, "I could not find a fix.", ref)
        assert res.passed is False
        assert res.reason == "no_patch_found"


def test_score_swebench_none_without_repo():
    # No work-tree -> cannot execute -> None (caller falls back to placeholder).
    assert score_swebench("diff --git ...", {"gold_patch": "x"}) is None


def test_score_swebench_binary_with_repo():
    with _repo() as (d, gold, wrong, ref):
        _reset(d, ref["base_commit"])
        assert score_swebench(gold, ref, repo_dir=d) == 1.0
        _reset(d, ref["base_commit"])
        assert score_swebench(wrong, ref, repo_dir=d) == 0.0


# --------------------------------------------------------------------------- #
# Adapter integration: repo_provider routes scoring through the runner
# --------------------------------------------------------------------------- #
def test_adapter_uses_runner_when_provider_set():
    with _repo() as (d, gold, wrong, ref):
        @contextmanager
        def provider(reference):
            _reset(d, reference["base_commit"])   # fresh work-tree per grade
            yield d

        adapter = SweBenchAdapter(repo_provider=provider)
        assert adapter.score_output(gold, ref) == 1.0
        assert adapter.score_output(wrong, ref) == 0.0


def test_adapter_defaults_to_exact_match_placeholder():
    # No repo_provider -> unchanged offline behaviour (exact normalized-patch match).
    adapter = SweBenchAdapter()
    ref = {"gold_patch": "diff --git a/x b/x\n+fix\n"}
    assert adapter.score_output("diff --git a/x b/x\n+fix\n", ref) == 1.0
    assert adapter.score_output("diff --git a/x b/x\n+nope\n", ref) == 0.0
