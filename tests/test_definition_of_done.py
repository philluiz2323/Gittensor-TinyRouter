"""Offline tests for the SPEC definition-of-done roll-up.

The SPEC's bar is *"R1–R4 and R8 hold on at least 2 of our chosen in-distribution tasks;
… the optimizer drives J(θ) upward"*. These tests pin the three things that make the
verdict trustworthy: each invariant is read the way the SPEC words it (R2 is strictly
"beats EVERY single model", not just the best-scoring one), the ≥2-task rule is a *count*
of tasks rather than an average, and missing evidence is reported as "not measured" and
never counted as a pass. Synthetic per-item vectors, numpy only — no torch/network.
"""
import json
import subprocess
import sys
from pathlib import Path

from trinity.analysis import definition_of_done as dod_pkg  # re-export check
from trinity.analysis.definition_of_done import (
    DOD_INVARIANTS,
    DOD_MIN_TASKS,
    assess,
    assess_task,
    render,
)

_REPO = Path(__file__).resolve().parents[1]
_BOOT = 200  # keep the paired bootstrap cheap in tests


def _per_item(trinity, singles, random_routing):
    d = {"task_ids": [f"q{i}" for i in range(len(trinity))],
         "TRINITY": trinity, "random_routing": random_routing}
    for name, vec in singles.items():
        d[f"single::{name}"] = vec
    return d


def _winning_task(name="math500", **kw):
    return assess_task(
        name,
        _per_item([1, 1, 1, 1, 1, 1, 0, 1],
                  {"alpha": [1, 0, 1, 0, 1, 0, 0, 1], "beta": [0, 1, 0, 1, 0, 1, 0, 0]},
                  [1, 0, 0, 1, 0, 0, 0, 1]),
        n_boot=_BOOT, **kw,
    )


# --------------------------------------------------------------------------- #
# per-task invariant reading
# --------------------------------------------------------------------------- #
def test_winning_task_holds_r1_r2_r4():
    t = _winning_task()
    assert t.holds["R1"] is True
    assert t.holds["R2"] is True
    assert t.holds["R4"] is True
    assert "TRINITY" in t.detail["R1"]


def test_r2_requires_beating_every_single_not_just_the_best():
    # TRINITY (0.500) beats `weak` (0.250) but LOSES to `strong` (0.875).
    t = assess_task(
        "mmlu",
        _per_item([1, 1, 1, 1, 0, 0, 0, 0],
                  {"strong": [1, 1, 1, 1, 1, 1, 0, 1], "weak": [1, 1, 0, 0, 0, 0, 0, 0]},
                  [0, 0, 0, 0, 0, 0, 0, 1]),
        n_boot=_BOOT,
    )
    assert t.holds["R2"] is False
    assert "strong" in t.detail["R2"]
    assert t.holds["R1"] is False          # best single is `strong`, so R1 fails too


def test_r3_compares_against_the_plurality_ensemble():
    assert _winning_task(ensemble_accuracy=0.10).holds["R3"] is True
    assert _winning_task(ensemble_accuracy=0.99).holds["R3"] is False


def test_r8_requires_sep_cmaes_to_rank_first():
    top = _winning_task(r8_ranking=[{"trainer": "sep_cmaes"}, {"trainer": "sft"}])
    bad = _winning_task(r8_ranking=[{"trainer": "random_search"}, {"trainer": "sep_cmaes"}])
    assert top.holds["R8"] is True and bad.holds["R8"] is False
    assert "sep_cmaes" in bad.detail["R8"]


def test_missing_evidence_is_not_measured_never_a_pass():
    t = assess_task("gpqa")                # no per_item, no ensemble, no ranking
    assert all(t.holds[k] is None for k in DOD_INVARIANTS)
    assert t.measured == []
    assert "not measured" in t.detail["R3"]


# --------------------------------------------------------------------------- #
# the >=2-task rule
# --------------------------------------------------------------------------- #
def test_two_of_three_tasks_meets_the_bar():
    win_a, win_b = _winning_task("math500"), _winning_task("gpqa")
    lose = assess_task(
        "mmlu",
        _per_item([0, 0, 0, 0, 1, 0, 0, 0],
                  {"strong": [1, 1, 1, 1, 1, 1, 0, 1]}, [1, 1, 1, 1, 1, 0, 1, 1]),
        n_boot=_BOOT,
    )
    v = assess([win_a, lose, win_b], drives_J_upward=True, min_tasks=2)
    assert v.task_counts["R1"] == 2 and v.invariants_met["R1"] is True
    assert v.task_counts["R2"] == 2 and v.invariants_met["R2"] is True


def test_one_task_short_of_the_bar_fails():
    v = assess([_winning_task("math500")], drives_J_upward=True, min_tasks=2)
    assert v.task_counts["R1"] == 1
    assert v.invariants_met["R1"] is False
    assert v.passed is False
    assert "R1" in v.unmet


def test_rule_counts_tasks_not_averages():
    # One blowout win + one loss must NOT pass a 2-task bar, even though the MEAN
    # across tasks would favour TRINITY. This is the rule results_table does not apply.
    blowout = assess_task(
        "math500",
        _per_item([1] * 8, {"alpha": [0] * 8}, [0] * 8), n_boot=_BOOT,
    )
    loss = assess_task(
        "mmlu",
        _per_item([0] * 7 + [1], {"alpha": [1] * 8}, [1] * 8), n_boot=_BOOT,
    )
    v = assess([blowout, loss], drives_J_upward=True, min_tasks=2)
    assert v.task_counts["R1"] == 1 and v.passed is False


def test_optimizer_clause_is_required():
    tasks = [_winning_task("math500", ensemble_accuracy=0.1,
                           r8_ranking=[{"trainer": "sep_cmaes"}]),
             _winning_task("gpqa", ensemble_accuracy=0.1,
                           r8_ranking=[{"trainer": "sep_cmaes"}])]
    assert assess(tasks, drives_J_upward=True).passed is True
    assert assess(tasks, drives_J_upward=False).passed is False
    assert assess(tasks, drives_J_upward=None).passed is False   # unmeasured != satisfied


def test_spec_default_is_two_tasks():
    assert DOD_MIN_TASKS == 2
    assert DOD_INVARIANTS == ("R1", "R2", "R3", "R4", "R8")


def test_module_is_reachable_through_the_analysis_package():
    assert dod_pkg.DOD_MIN_TASKS == DOD_MIN_TASKS
    assert hasattr(dod_pkg, "assess") and hasattr(dod_pkg, "render")


# --------------------------------------------------------------------------- #
# rendering + serialization
# --------------------------------------------------------------------------- #
def test_render_matrix_tally_and_verdict():
    tasks = [_winning_task("math500", ensemble_accuracy=0.1,
                           r8_ranking=[{"trainer": "sep_cmaes"}]),
             _winning_task("gpqa", ensemble_accuracy=0.1,
                           r8_ranking=[{"trainer": "sep_cmaes"}])]
    md = render(assess(tasks, drives_J_upward=True))
    assert "definition of done" in md.lower()
    assert "| math500 |" in md and "| gpqa |" in md
    assert "holds on" in md and "PASS" in md
    assert "H200" in md                    # the non-offline clause is surfaced, not dropped


def test_render_reports_not_met_and_serializes():
    v = assess([_winning_task("math500")], drives_J_upward=None, min_tasks=2)
    assert "NOT MET" in render(v)
    d = v.to_dict()
    assert d["passed"] is False and d["min_tasks"] == 2
    assert "not_offline_checkable" in d
    json.dumps(d)                          # must be JSON-serializable


def test_render_empty_is_graceful():
    assert "no tasks assessed" in render(assess([], drives_J_upward=True))


# --------------------------------------------------------------------------- #
# report script end-to-end
# --------------------------------------------------------------------------- #
def test_report_script_reads_eval_jsons(tmp_path):
    for bench in ("math500", "gpqa"):
        (tmp_path / f"eval_{bench}.json").write_text(json.dumps({
            "benchmark": bench,
            "per_item": _per_item([1, 1, 1, 1, 1, 1, 0, 1],
                                  {"alpha": [1, 0, 1, 0, 1, 0, 0, 1]},
                                  [1, 0, 0, 1, 0, 0, 0, 1]),
        }))
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "definition_of_done_report.py"),
         "--root", str(tmp_path), "--n-boot", "200"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0, out.stderr
    assert "| math500 |" in out.stdout and "| gpqa |" in out.stdout
    # R8 / J(theta) have no run artifacts here -> must NOT pass
    assert "NOT MET" in out.stdout


def test_report_script_no_files_is_graceful():
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "definition_of_done_report.py")],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0 and "no eval JSONs" in out.stdout
