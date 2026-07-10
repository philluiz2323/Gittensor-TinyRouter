"""Offline tests for the efficiency / composite-score analysis. No network, no GPU."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from trinity.efficiency import (
    DEFAULT_MAX_TURNS,
    SCORE_WEIGHTS,
    TurnRecord,
    avg_turns,
    composite_score,
    summarize_efficiency,
    trajectory_turn_records,
    turn_efficiency,
)

_REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# turn_efficiency
# ---------------------------------------------------------------------------
def test_single_turn_correct_earns_the_full_live_acc():
    assert turn_efficiency(1.0, 0.8) == pytest.approx(0.8)


def test_using_the_whole_budget_earns_zero():
    assert turn_efficiency(DEFAULT_MAX_TURNS, 0.8) == 0.0


def test_efficiency_is_zero_when_nothing_is_correct():
    assert turn_efficiency(1.0, 0.0) == 0.0


def test_efficiency_never_goes_negative_past_the_budget():
    # A defensive guard: avg_turns above the budget clamps at 0, not negative.
    assert turn_efficiency(DEFAULT_MAX_TURNS + 3, 0.9) == 0.0


def test_efficiency_is_linear_between_the_endpoints():
    # avg_turns=3 on a 1..5 budget -> (5-3)/4 = 0.5 of live_acc.
    assert turn_efficiency(3.0, 0.6) == pytest.approx(0.5 * 0.6)


# ---------------------------------------------------------------------------
# The formula must match the hidden scorer (pr_eval._compute_score)
# ---------------------------------------------------------------------------
def _load_pr_eval():
    spec = importlib.util.spec_from_file_location(
        "pr_eval", _REPO / "scripts" / "pr_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pr_eval", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hidden,live,turns,novelty", [
    (0.82, 0.70, 2.4, 0.10),
    (0.5, 0.5, 1.0, 0.0),
    (0.9, 0.0, 5.0, 0.3),      # live_acc 0 -> efficiency 0
    (1.0, 1.0, 5.0, 1.0),      # max turns -> efficiency 0
    (0.33, 0.66, 3.7, 0.05),
])
def test_composite_matches_pr_eval_when_importable(hidden, live, turns, novelty):
    try:
        pr_eval = _load_pr_eval()
    except Exception:
        pytest.skip("pr_eval not importable in this environment")
    expected = pr_eval._compute_score(hidden, live, turns, novelty)
    got = composite_score(
        hidden_acc=hidden, live_acc=live, avg_turns_used=turns, novelty=novelty
    ).total
    assert got == pytest.approx(expected)


def test_composite_matches_the_documented_formula_directly():
    # Independent of pr_eval: 0.70h + 0.15l + 0.10*eff + 0.05*nov.
    b = composite_score(hidden_acc=0.8, live_acc=0.6, avg_turns_used=3.0, novelty=0.2)
    eff = (5 - 3) / 4 * 0.6
    expected = 0.70 * 0.8 + 0.15 * 0.6 + 0.10 * eff + 0.05 * 0.2
    assert b.total == pytest.approx(expected)
    assert b.efficiency == pytest.approx(eff)
    assert sum(b.weighted.values()) == pytest.approx(b.total)


def test_weights_sum_to_one():
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# avg_turns and the missing-turn penalty
# ---------------------------------------------------------------------------
def test_avg_turns_penalizes_a_missing_turn_count_as_max():
    # A trajectory that never terminated (turns<=0) is charged the full budget,
    # matching pr_eval charging max_turns on a failed/errored trajectory.
    recs = [TurnRecord(correct=True, turns=2), TurnRecord(correct=False, turns=0)]
    assert avg_turns(recs, max_turns=5) == pytest.approx((2 + 5) / 2)
    assert avg_turns(recs, max_turns=5, penalize_missing=False) == pytest.approx((2 + 0) / 2)


def test_avg_turns_of_nothing_is_zero():
    assert avg_turns([]) == 0.0


# ---------------------------------------------------------------------------
# summarize_efficiency
# ---------------------------------------------------------------------------
def test_summary_counts_and_per_correct_costs():
    recs = [
        TurnRecord(correct=True, turns=1, llm_calls=2, cost_usd=0.01),
        TurnRecord(correct=True, turns=3, llm_calls=4, cost_usd=0.03),
        TurnRecord(correct=False, turns=5, llm_calls=6, cost_usd=0.05),
    ]
    s = summarize_efficiency(recs, max_turns=5)
    assert s.n_tasks == 3 and s.n_correct == 2
    assert s.accuracy == pytest.approx(2 / 3)
    assert s.avg_turns == pytest.approx((1 + 3 + 5) / 3)
    assert s.avg_turns_correct == pytest.approx((1 + 3) / 2)
    # Total turns/calls/cost across ALL tasks, divided by correct answers.
    assert s.turns_per_correct == pytest.approx((1 + 3 + 5) / 2)
    assert s.calls_per_correct == pytest.approx((2 + 4 + 6) / 2)
    assert s.cost_per_correct == pytest.approx((0.01 + 0.03 + 0.05) / 2)


def test_per_correct_is_infinite_when_nothing_solved():
    recs = [TurnRecord(correct=False, turns=5, llm_calls=3, cost_usd=0.02)]
    s = summarize_efficiency(recs)
    assert s.n_correct == 0 and s.accuracy == 0.0
    assert s.turns_per_correct == float("inf")
    assert s.calls_per_correct == float("inf")
    assert s.cost_per_correct == float("inf")
    assert s.efficiency == 0.0


def test_calls_and_cost_are_none_when_not_supplied_for_every_task():
    recs = [
        TurnRecord(correct=True, turns=1, llm_calls=2),
        TurnRecord(correct=True, turns=1),           # no llm_calls
    ]
    s = summarize_efficiency(recs)
    assert s.calls_per_correct is None
    assert s.cost_per_correct is None


def test_summary_of_nothing_is_all_zero():
    s = summarize_efficiency([])
    assert s.n_tasks == 0 and s.efficiency == 0.0
    assert s.to_dict()["n_tasks"] == 0


# ---------------------------------------------------------------------------
# trajectory adapter
# ---------------------------------------------------------------------------
class _Traj:
    def __init__(self, n_turns: int) -> None:
        self.n_turns = n_turns


def test_trajectory_records_use_explicit_correctness():
    trajs = [_Traj(2), _Traj(4)]
    recs = trajectory_turn_records(trajs, correctness=[1, 0])
    assert [(r.correct, r.turns) for r in recs] == [(True, 2), (False, 4)]


def test_trajectory_records_use_a_custom_score_fn():
    trajs = [_Traj(1), _Traj(3)]
    recs = trajectory_turn_records(trajs, score_fn=lambda t: 1.0 if t.n_turns == 1 else 0.0)
    assert [r.correct for r in recs] == [True, False]


def test_trajectory_records_reject_mismatched_correctness_length():
    with pytest.raises(ValueError, match="expected"):
        trajectory_turn_records([_Traj(1), _Traj(2)], correctness=[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
