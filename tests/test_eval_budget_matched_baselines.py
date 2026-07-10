"""The R1/R2 single-model baselines must be budget-matched to TRINITY.

SPEC §1.3.4:

    Budget-matched single-model baselines (R1/R2): run each single model at
    max_tokens = 20,480 (5x) so the single-vs-TRINITY comparison is fair,
    matching the paper's 5x protocol.

`evaluate` passed `args.max_tokens` (4096) to each baseline's single turn, while
TRINITY ran up to `max_turns=5` turns at 4096 each. The baselines therefore got a
fifth of TRINITY's budget, and R1 ("TRINITY avg > best single model avg") was
partly decided by token budget rather than by routing.

Offline: no network, no `httpx` request. The pool records the `max_tokens` it is
asked for.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from trinity import eval as trinity_eval
from trinity.eval import single_model_budget


@pytest.fixture(autouse=True)
def _stub_httpx(monkeypatch):
    """`_score_single_model` opens an `httpx.AsyncClient`; no request is issued."""

    class _AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    module = types.ModuleType("httpx")
    module.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)


# --------------------------------------------------------------------------- #
# single_model_budget
# --------------------------------------------------------------------------- #
def test_default_budget_is_the_spec_20480():
    """5 turns x 4096 tokens — the exact number SPEC §1.3.4 names."""
    assert single_model_budget(4096, 5) == 20_480


def test_budget_is_the_product_of_turns_and_tokens():
    assert single_model_budget(1000, 3) == 3000


def test_budget_tracks_a_changed_max_turns():
    """Derived from max_turns, not hard-coded 5, so the match survives --max-turns."""
    assert single_model_budget(4096, 2) == 8192
    assert single_model_budget(4096, 8) == 32_768


def test_single_turn_trinity_needs_no_multiplier():
    assert single_model_budget(4096, 1) == 4096


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_max_tokens_rejected(bad):
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        single_model_budget(bad, 5)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_max_turns_rejected(bad):
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        single_model_budget(4096, bad)


# --------------------------------------------------------------------------- #
# The baseline actually receives the matched budget
# --------------------------------------------------------------------------- #
class _Result:
    text = "ok"


class _RecordingPool:
    def __init__(self) -> None:
        self.max_tokens_seen: list[int] = []

    async def chat(self, model, messages, *, max_tokens, **kw):
        self.max_tokens_seen.append(max_tokens)
        return _Result()


class _Adapter:
    def build_prompt(self, task):
        return task.prompt

    def score_output(self, text, answer):
        return 1.0


class _Task:
    def __init__(self, tid):
        self.task_id = tid
        self.prompt = "q"
        self.answer = "a"


def _run_baseline(pool, budget):
    return asyncio.run(
        trinity_eval._score_single_model(
            [_Task("t0"), _Task("t1")], pool, "m", _Adapter(),
            max_tokens=budget, reasoning=None,
        )
    )


def test_baseline_requests_the_matched_budget():
    """The regression: the baseline used to be handed 4096, not 20,480."""
    pool = _RecordingPool()
    _run_baseline(pool, single_model_budget(4096, 5))

    assert pool.max_tokens_seen == [20_480, 20_480]


def test_baseline_is_not_handed_the_per_turn_cap():
    pool = _RecordingPool()
    _run_baseline(pool, single_model_budget(4096, 5))

    assert 4096 not in pool.max_tokens_seen


def test_every_task_gets_the_same_budget():
    pool = _RecordingPool()
    _run_baseline(pool, single_model_budget(1024, 3))

    assert set(pool.max_tokens_seen) == {3072}


def test_trinity_per_turn_cap_is_unchanged():
    """TRINITY still spends `max_tokens` PER TURN; only the baseline is scaled."""
    per_turn, turns = 4096, 5
    assert single_model_budget(per_turn, turns) == per_turn * turns
    # The routed path receives the per-turn cap, never the total.
    run_kwargs = dict(max_turns=turns, max_tokens=per_turn)
    assert run_kwargs["max_tokens"] == per_turn
