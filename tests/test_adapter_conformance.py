"""Offline tests for the benchmark-adapter conformance auditor.

Audits the real registry (must be green today) and, via a synthetic broken adapter,
proves each contract check actually FIRES on a violation. No torch/network — adapters use
their offline toy fallback.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.adapters.base import BenchmarkAdapter, ScoringMode, TaskType
from trinity.adapters.conformance import (
    ADAPTER_CASES,
    ConformanceReport,
    audit_adapter,
    audit_all,
    render,
)
from trinity.types import Task

_REPO = Path(__file__).resolve().parents[1]


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.adapters.conformance; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# real registry: every adapter conforms today
# --------------------------------------------------------------------------- #
def test_real_registry_is_conformant():
    report = audit_all()
    assert report.ok, f"adapter contract failures: {[r.to_dict() for r in report.failures()]}"
    assert len(report.adapters) >= 9                       # the full registered suite
    # the math/MCQ adapters get a self-consistency check; all get the generic contracts.
    checked = {r.contract for r in report.results}
    assert {"task_type", "scoring_modes", "load_tasks", "load_determinism",
            "serialize_task", "score_output_binary", "self_consistency"} <= checked


def test_self_consistency_cases_cover_the_math_mcq_adapters():
    report = audit_all()
    audited = set(report.adapters)
    for name in ADAPTER_CASES:
        assert name in audited
        sc = [r for r in report.results if r.adapter == name and r.contract == "self_consistency"]
        assert sc and sc[0].ok


# --------------------------------------------------------------------------- #
# synthetic broken adapter: each contract check must FIRE on its violation
# --------------------------------------------------------------------------- #
class _FakeAdapter(BenchmarkAdapter):
    """A conformant baseline adapter with one togglable contract violation."""

    name = "fake"

    def __init__(self, **breaks):
        self.b = {"binary": True, "json_safe": True, "deterministic": True,
                  "valid_type": True, "all_keys": True, "self_consistent": True, **breaks}
        self._loads = 0

    def load_tasks(self, split, max_items, seed=0):
        self._loads += 1
        idx = 0 if self.b["deterministic"] else self._loads
        return [Task(task_id=f"fake-{idx}", benchmark="fake", prompt="Q", answer="B", meta={})]

    def build_prompt(self, task):
        return task.prompt

    def score_output(self, output, reference):
        if not self.b["binary"]:
            return 0.5
        if not self.b["self_consistent"]:
            return 0.0                                  # never credits the correct answer
        return 1.0 if str(reference) in output else 0.0

    def task_type(self):
        return TaskType.MCQ if self.b["valid_type"] else "mcq"   # str, not a TaskType

    def scoring_modes(self):
        return frozenset({ScoringMode.CACHED})

    def serialize_task(self, task):
        d = {"task_id": task.task_id, "benchmark": task.benchmark, "prompt": task.prompt,
             "reference": task.answer, "task_type": "mcq", "meta": {}}
        if not self.b["all_keys"]:
            d.pop("reference")
        if not self.b["json_safe"]:
            d["oops"] = {1, 2, 3}                        # a set is not JSON-native
        return d


_CASE = ("says B", "says A", "B")


def _result(adapter, contract):
    rows = audit_adapter("fake", adapter, case=_CASE)
    return next(r for r in rows if r.contract == contract)


def test_clean_fake_passes_every_contract():
    rows = audit_adapter("fake", _FakeAdapter(), case=_CASE)
    assert all(r.ok for r in rows) and len(rows) == 7


@pytest.mark.parametrize("break_kw,contract", [
    ({"binary": False}, "score_output_binary"),
    ({"json_safe": False}, "serialize_task"),
    ({"all_keys": False}, "serialize_task"),
    ({"deterministic": False}, "load_determinism"),
    ({"valid_type": False}, "task_type"),
    ({"self_consistent": False}, "self_consistency"),
])
def test_each_violation_is_caught(break_kw, contract):
    res = _result(_FakeAdapter(**break_kw), contract)
    assert not res.ok, f"{contract} should have failed for {break_kw}"


def test_report_is_not_ok_when_a_contract_fails():
    report = ConformanceReport(results=audit_adapter("fake", _FakeAdapter(binary=False), case=_CASE))
    assert not report.ok and report.failures()


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_pass_and_fail():
    ok_md = render(audit_all())
    assert "adapter conformance" in ok_md.lower() and "ALL PASS" in ok_md
    bad = ConformanceReport(results=audit_adapter("fake", _FakeAdapter(binary=False), case=_CASE))
    bad_md = render(bad)
    assert "FAILURE" in bad_md and "score_output_binary" in bad_md
    assert render(ConformanceReport()).strip().endswith("(no adapters registered)_")
