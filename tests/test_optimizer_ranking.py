"""Offline tests for the R8 (sep-CMA-ES > SFT > RS > REINFORCE) verifier.

No network, no GPU, no torch — plain numbers only.
"""
from __future__ import annotations

from trinity.analysis.optimizer_ranking import (
    EXPECTED_ORDER,
    analyze_task,
    analyze_tasks,
    canonical_optimizer,
    render,
)


# --------------------------------------------------------------------------- #
# canonical_optimizer
# --------------------------------------------------------------------------- #
def test_canonicalizes_aliases_and_rejects_unknown():
    assert canonical_optimizer("Sep-CMA-ES") == "sep_cmaes"
    assert canonical_optimizer("cma") == "sep_cmaes"
    assert canonical_optimizer("random search") == "rs"
    assert canonical_optimizer("random_search") == "rs"
    assert canonical_optimizer("policy_gradient") == "reinforce"
    assert canonical_optimizer("SFT") == "sft"
    assert canonical_optimizer("adam") is None
    assert canonical_optimizer(42) is None


def test_expected_order_is_the_spec_chain():
    assert EXPECTED_ORDER == ("sep_cmaes", "sft", "rs", "reinforce")


# --------------------------------------------------------------------------- #
# analyze_task
# --------------------------------------------------------------------------- #
def test_correct_full_chain_holds():
    r = analyze_task(
        {"sep_cmaes": 0.72, "sft": 0.66, "rs": 0.60, "reinforce": 0.55}, task="math500"
    )
    assert r.holds
    assert r.observed == ["sep_cmaes", "sft", "rs", "reinforce"]
    assert r.inversions == []


def test_alias_keys_are_accepted():
    r = analyze_task(
        {"Sep-CMA-ES": 0.72, "SFT": 0.66, "Random Search": 0.60, "REINFORCE": 0.55}
    )
    assert r.holds and r.observed == ["sep_cmaes", "sft", "rs", "reinforce"]


def test_an_inversion_is_reported_and_breaks_r8():
    # RS beats SFT -> the (sft, rs) expected pair inverts.
    r = analyze_task({"sep_cmaes": 0.72, "sft": 0.58, "rs": 0.60, "reinforce": 0.55})
    assert not r.holds
    assert ("sft", "rs") in r.inversions
    assert r.observed[0] == "sep_cmaes"          # cma still best
    assert r.observed.index("rs") < r.observed.index("sft")  # rs measured above sft


def test_tie_is_not_a_strict_pass():
    r = analyze_task({"sep_cmaes": 0.60, "sft": 0.60, "rs": 0.50, "reinforce": 0.40})
    assert not r.holds
    assert ("sep_cmaes", "sft") in r.inversions


def test_subset_of_optimizers_checks_only_present_chain():
    # Only cma / rs / reinforce present -> chain checked among them, in expected order.
    r = analyze_task({"cma": 0.7, "rs": 0.6, "reinforce": 0.5})
    assert r.holds
    assert r.expected == ["sep_cmaes", "rs", "reinforce"]


def test_fewer_than_two_recognized_optimizers_cannot_hold():
    assert analyze_task({"sep_cmaes": 0.7}).holds is False
    assert analyze_task({"adam": 0.9, "sgd": 0.8}).holds is False   # none recognized


def test_non_numeric_and_unknown_entries_are_dropped():
    r = analyze_task({"sep_cmaes": 0.7, "sft": "n/a", "rs": 0.6, "adam": 0.99})
    assert set(r.scores) == {"sep_cmaes", "rs"}
    assert r.holds and r.expected == ["sep_cmaes", "rs"]


# --------------------------------------------------------------------------- #
# analyze_tasks: across-tasks verdict
# --------------------------------------------------------------------------- #
def test_r8_holds_only_when_every_scored_task_holds():
    report = analyze_tasks({
        "math500": {"sep_cmaes": 0.72, "sft": 0.66, "rs": 0.60, "reinforce": 0.55},
        "mmlu": {"sep_cmaes": 0.68, "sft": 0.60, "rs": 0.58, "reinforce": 0.50},
    })
    assert report["r8_holds"] is True
    assert report["n_tasks_scored"] == 2
    assert report["violations"] == []


def test_one_violating_task_fails_the_whole_invariant():
    report = analyze_tasks({
        "math500": {"sep_cmaes": 0.72, "sft": 0.66, "rs": 0.60, "reinforce": 0.55},
        "livecodebench": {"sep_cmaes": 0.50, "sft": 0.66, "rs": 0.60, "reinforce": 0.55},
    })
    assert report["r8_holds"] is False
    assert report["violations"] == ["livecodebench"]


def test_all_unscorable_does_not_hold():
    report = analyze_tasks({"math500": {"sep_cmaes": 0.7}})   # only one optimizer
    assert report["r8_holds"] is False
    assert report["n_tasks_scored"] == 0


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_reports_holds_and_order():
    md = render({"math500": {"sep_cmaes": 0.72, "sft": 0.66, "rs": 0.60, "reinforce": 0.55}})
    assert "R8 (sep-CMA-ES > SFT > RS > REINFORCE): HOLDS" in md
    assert "sep_cmaes > sft > rs > reinforce" in md


def test_render_flags_a_violation():
    md = render({"t": {"sep_cmaes": 0.72, "sft": 0.58, "rs": 0.60, "reinforce": 0.55}})
    assert "VIOLATED" in md and "violations: t" in md
    assert "inverts sft<=rs" in md
