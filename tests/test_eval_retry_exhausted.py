"""Tests that one retry-exhausted trajectory degrades to 0.0 instead of aborting the eval.

`_score_policy` and `_score_single_model` fan every task out through `asyncio.gather`.
A single trajectory that exhausts its retries (e.g. a persistent `httpx.ReadTimeout`)
used to propagate out of `gather`, out of the scorer, and out of `evaluate()` — throwing
away the TRINITY score and every single-model baseline already computed this run. These
tests pin the pessimistic-degrade contract that training (`trinity.optim.fitness`) already
honours: a failed task counts as 0.0 and stays in the denominator, while an all-failed run
raises rather than reporting a meaningless 0.0.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity import eval as te
from trinity.types import Task


class _Adapter:
    """Adapter stub: a trajectory is its own score, an output is its own float."""

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_trajectory(self, traj: float) -> float:
        return float(traj)

    def score_output(self, output: str, reference: object) -> float:
        return float(output)


@dataclass
class _ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _Pool:
    """Single-model pool whose `chat` raises for the task ids in `fail_ids`."""

    def __init__(self, fail_ids: set[str]) -> None:
        self.fail_ids = fail_ids

    async def chat(self, model, messages, *, max_tokens=0, temperature=0.0, reasoning=None,
                   client=None):
        blob = " ".join(m["content"] for m in messages)
        if any(blob == f"Q{tid}" for tid in self.fail_ids):
            raise RuntimeError("httpx.ReadTimeout: retries exhausted")
        return _ChatResult(text="1.0")


def _tasks(n: int) -> list[Task]:
    return [Task(task_id=str(i), benchmark="math500", prompt=f"Q{i}", answer="1.0")
            for i in range(n)]


def _fake_run_trajectory(fail_ids: set[str]):
    """Return a `run_trajectory` stub that raises for `fail_ids` and returns 1.0 otherwise."""

    async def _run(task, *a, **kw):
        if task.task_id in fail_ids:
            raise RuntimeError("httpx.ReadTimeout: retries exhausted")
        return 1.0

    return _run


def test_one_failed_trajectory_degrades_to_zero_not_abort(monkeypatch):
    """3 of 4 tasks answer correctly, one exhausts retries -> score 0.75, no abort."""
    monkeypatch.setattr(te, "run_trajectory", _fake_run_trajectory({"2"}))
    score = asyncio.run(te._score_policy(
        _tasks(4), policy=None, pool=None, pool_models=[], adapter=_Adapter(), sample=False))
    assert score == pytest.approx(0.75)


def test_clean_run_matches_plain_mean(monkeypatch):
    """No failures -> identical to the old plain-mean path (happy path unchanged)."""
    monkeypatch.setattr(te, "run_trajectory", _fake_run_trajectory(set()))
    score = asyncio.run(te._score_policy(
        _tasks(4), policy=None, pool=None, pool_models=[], adapter=_Adapter(), sample=False))
    assert score == pytest.approx(1.0)


def test_all_failed_raises_rather_than_reporting_zero(monkeypatch):
    """A dead API must raise, not report 0.0 accuracy that looks like a real measurement."""
    monkeypatch.setattr(te, "run_trajectory", _fake_run_trajectory({"0", "1", "2", "3"}))
    with pytest.raises(RuntimeError, match="all 4 trajectories failed"):
        asyncio.run(te._score_policy(
            _tasks(4), policy=None, pool=None, pool_models=[], adapter=_Adapter(), sample=False))


def test_single_model_baseline_survives_one_failure(monkeypatch):
    """The single-model baseline degrades one failed task to 0.0 as well."""
    monkeypatch.setattr("trinity.roles.prompts.build_messages",
                        lambda role, prompt, history: [{"role": "user", "content": prompt}])
    score = asyncio.run(te._score_single_model(
        _tasks(4), _Pool(fail_ids={"2"}), model="m0", adapter=_Adapter(),
        max_tokens=16, reasoning=None))
    assert score == pytest.approx(0.75)


def test_failure_count_is_printed(monkeypatch, capsys):
    """A degraded number is announced so it is never mistaken for a clean one."""
    monkeypatch.setattr(te, "run_trajectory", _fake_run_trajectory({"2"}))
    asyncio.run(te._score_policy(
        _tasks(4), policy=None, pool=None, pool_models=[], adapter=_Adapter(), sample=False,
        label="TRINITY"))
    out = capsys.readouterr().out
    assert "1/4 trajectories failed" in out
    assert "TRINITY" in out
