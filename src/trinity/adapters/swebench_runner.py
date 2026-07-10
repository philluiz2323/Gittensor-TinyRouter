"""Sandboxed SWE-bench patch evaluator (issue #18).

The SWE-bench adapter (#17) turns a repository issue into a task and lets a model
emit a unified diff; this module is the *dedicated runner* that actually grades
that diff by applying it to a checked-out repo and running the task's tests. It
is the execution counterpart to the adapter's cheap exact-match placeholder.

Grading follows SWE-bench's own rule: a patch **resolves** an instance iff, after
it is applied, every ``FAIL_TO_PASS`` test passes **and** every ``PASS_TO_PASS``
test still passes. The result is a single binary reward.

Isolation & safety, mirroring the repo's existing ``reward.run_pass_at_1``
sandbox: the patch is applied with ``git apply`` inside a caller-provided
work-tree, tests run in a **subprocess with a wall-clock timeout**, and this
module never ``exec``s model output in-process. Preparing the work-tree (cloning
the repo at ``base_commit``) is a network operation kept behind :func:`prepare_repo`
and injected, so the pure grading mechanism (:func:`evaluate_patch`) is fully
offline-testable against a throwaway local git repo.

This module is imported only when a patch is actually executed, and it touches no
other benchmark's scoring path.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

__all__ = [
    "PatchEvalResult",
    "extract_patch",
    "apply_patch",
    "run_tests",
    "evaluate_patch",
    "score_swebench",
    "prepare_repo",
]

#: Default per-step wall-clock limits (seconds). Bounded so an adversarial patch
#: or a hanging test cannot stall the evaluator.
_APPLY_TIMEOUT = 60
_TEST_TIMEOUT = 900

_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DIFF_START_RE = re.compile(r"^(diff --git |--- |\+\+\+ |Index: )", re.MULTILINE)


@dataclass
class PatchEvalResult:
    """Outcome of grading one candidate patch.

    ``passed`` is the binary reward. ``reason`` is a stable machine-readable tag
    (``patch_did_not_apply`` / ``tests_failed`` / ``resolved`` / ...), and
    ``detail`` carries human-readable context (git/pytest stderr) for clean
    failure reporting without leaking into the reward.
    """

    passed: bool
    reason: str
    applied: bool = False
    detail: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def reward(self) -> float:
        return 1.0 if self.passed else 0.0


def extract_patch(text: str) -> str:
    """Pull the unified diff out of a model's (possibly chatty) output.

    Prefers a fenced ```` ```diff ```` block; otherwise takes everything from the
    first diff header (``diff --git`` / ``---`` / ``Index:``) onward. Returns an
    empty string if no diff-looking content is present.
    """
    if not text:
        return ""
    fence = _FENCE_RE.search(text)
    if fence:
        candidate = fence.group(1)
        if _DIFF_START_RE.search(candidate):
            return candidate.strip("\n") + "\n"
    m = _DIFF_START_RE.search(text)
    if m:
        return text[m.start():].strip("\n") + "\n"
    return ""


def apply_patch(repo_dir: str | Path, patch: str, *, timeout: int = _APPLY_TIMEOUT) -> tuple[bool, str]:
    """Apply ``patch`` inside ``repo_dir`` with ``git apply``.

    Returns ``(ok, detail)``. ``git apply`` is used (not ``patch``) so malformed
    or fuzzy diffs are rejected rather than partially applied. ``--whitespace=nowarn``
    keeps benign whitespace differences from failing an otherwise-valid patch.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "git apply timed out"
    except FileNotFoundError:
        return False, "git not available"
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def run_tests(
    repo_dir: str | Path,
    node_ids: Sequence[str],
    *,
    timeout: int = _TEST_TIMEOUT,
) -> tuple[bool, str]:
    """Run the given pytest node ids in ``repo_dir`` in a subprocess.

    Returns ``(all_passed, detail)``. The tests run in a fresh ``python -m pytest``
    process (never in-process), with the cache disabled and a wall-clock timeout,
    so a hanging or crashing test is contained. ``all_passed`` is ``True`` only if
    pytest exits 0 (every selected test passed).
    """
    if not node_ids:
        return False, "no tests specified"
    try:
        proc = subprocess.run(
            ["python", "-m", "pytest", "-p", "no:cacheprovider", "-q", "--no-header",
             "--tb=no", *node_ids],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "tests timed out"
    except FileNotFoundError:
        return False, "python/pytest not available"
    tail = "\n".join((proc.stdout or "").splitlines()[-15:])
    return proc.returncode == 0, tail


def evaluate_patch(
    repo_dir: str | Path,
    candidate: str,
    reference: Any,
    *,
    apply_timeout: int = _APPLY_TIMEOUT,
    test_timeout: int = _TEST_TIMEOUT,
) -> PatchEvalResult:
    """Grade ``candidate`` against ``reference`` in a **prepared** ``repo_dir``.

    ``repo_dir`` must already be a checked-out work-tree at the instance's
    ``base_commit`` (see :func:`prepare_repo`). Steps: extract the diff, ``git
    apply`` it, then run ``FAIL_TO_PASS`` + ``PASS_TO_PASS`` and require all to
    pass. Never raises for a bad patch — a failure to apply or a failing test is a
    clean ``passed=False`` result with a reason.
    """
    ref = reference if isinstance(reference, dict) else {}
    fail_to_pass = list(ref.get("fail_to_pass", []) or [])
    pass_to_pass = list(ref.get("pass_to_pass", []) or [])

    patch = extract_patch(candidate)
    if not patch.strip():
        return PatchEvalResult(False, "no_patch_found", applied=False)

    applied, apply_detail = apply_patch(repo_dir, patch, timeout=apply_timeout)
    if not applied:
        return PatchEvalResult(False, "patch_did_not_apply", applied=False, detail=apply_detail)

    nodes = fail_to_pass + pass_to_pass
    if not nodes:
        return PatchEvalResult(False, "no_tests_specified", applied=True)

    ok, detail = run_tests(repo_dir, nodes, timeout=test_timeout)
    if ok:
        return PatchEvalResult(True, "resolved", applied=True, detail=detail)
    return PatchEvalResult(False, "tests_failed", applied=True, detail=detail, failures=nodes)


def score_swebench(
    candidate: str,
    reference: Any,
    *,
    repo_dir: str | Path | None = None,
    apply_timeout: int = _APPLY_TIMEOUT,
    test_timeout: int = _TEST_TIMEOUT,
) -> float | None:
    """Binary reward for a candidate patch, or ``None`` if it cannot be executed.

    When ``repo_dir`` (a prepared work-tree) is supplied, this returns
    ``1.0``/``0.0`` from :func:`evaluate_patch`. With no work-tree there is nothing
    to run against, so it returns ``None`` — the caller (the adapter) then falls
    back to its cheap exact-match placeholder rather than guessing.
    """
    if repo_dir is None:
        return None
    return evaluate_patch(
        repo_dir, candidate, reference,
        apply_timeout=apply_timeout, test_timeout=test_timeout,
    ).reward


@contextmanager
def prepare_repo(reference: Any, *, workdir: str | Path | None = None) -> Iterator[str]:
    """Clone the instance's repo at ``base_commit`` into a temp work-tree.

    **Network operation** (``git clone``) — the production seam that turns a
    reference into the checked-out ``repo_dir`` that :func:`evaluate_patch` grades
    against. It is deliberately separate from the pure grading mechanism so the
    latter stays offline-testable; tests inject their own prepared repo instead of
    calling this. Yields the repo path and cleans up the clone on exit.

    Raises:
        RuntimeError: If ``reference`` lacks ``repo``/``base_commit``, or the
            clone/checkout fails.
    """
    import tempfile

    ref = reference if isinstance(reference, dict) else {}
    repo = ref.get("repo")
    base_commit = ref.get("base_commit")
    if not repo or not base_commit:
        raise RuntimeError("reference is missing 'repo' or 'base_commit'")

    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="swebench-"))
    dest = root / "repo"
    url = repo if "://" in str(repo) else f"https://github.com/{repo}.git"
    try:
        clone = subprocess.run(
            ["git", "clone", "--quiet", url, str(dest)],
            capture_output=True, text=True, timeout=600,
        )
        if clone.returncode != 0:
            raise RuntimeError(f"git clone failed: {clone.stderr.strip()}")
        checkout = subprocess.run(
            ["git", "-C", str(dest), "checkout", "--quiet", str(base_commit)],
            capture_output=True, text=True, timeout=120,
        )
        if checkout.returncode != 0:
            raise RuntimeError(f"git checkout failed: {checkout.stderr.strip()}")
        yield str(dest)
    finally:
        if workdir is None:
            shutil.rmtree(root, ignore_errors=True)
