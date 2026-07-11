"""Offline tests for the training-convergence / R8 report (trinity.analysis.convergence).

Synthetic run artifacts in both schemas (sep-CMA-ES generation history and the RS
baseline trial history). No torch, no network — reads dicts directly.
"""
import sys

import pytest

from trinity.analysis import analyze_run as analyze_run_pkg  # re-export check
from trinity.analysis.convergence import analyze_run, analyze_runs, render


def test_no_torch_imported():
    assert "torch" not in sys.modules


def test_reexported_from_package():
    assert analyze_run_pkg is analyze_run


def _cma(bests, means=None):
    means = means if means is not None else bests
    return [{"generation": i, "gen_mean_fitness": means[i], "gen_max_fitness": bests[i],
             "best_fitness": bests[i], "seconds": 1.0} for i in range(len(bests))]


# --------------------------------------------------------------------------- #
# analyze_run
# --------------------------------------------------------------------------- #
def test_healthy_run_drives_j_upward():
    r = analyze_run({"benchmark": "math500", "best_fitness": 1.0},
                    _cma([0.90, 0.90, 0.95, 1.00], means=[0.70, 0.68, 0.80, 0.79]))
    assert r.trainer == "sep_cmaes" and r.benchmark == "math500"
    assert r.n_iters == 4 and r.initial == 0.90 and r.final == 1.00
    assert r.net_gain == pytest.approx(0.10) and r.improved is True and r.degenerate is False
    assert r.best_monotone is True and r.iters_to_best == 3 and r.tail_plateau == 1
    assert r.trend_slope > 0.0


def test_rs_baseline_schema_and_trainer():
    hist = [{"trial": 0, "fitness": 0.50, "best_fitness": 0.50},
            {"trial": 1, "fitness": 0.40, "best_fitness": 0.55}]
    r = analyze_run({"trainer": "random_search", "benchmark": "math500", "best_fitness": 0.55}, hist)
    assert r.trainer == "random_search" and r.final == 0.55 and r.improved is True


def test_degenerate_run_no_improvement():
    r = analyze_run({"benchmark": "mmlu", "best_fitness": 0.5}, _cma([0.5, 0.5, 0.5]))
    assert r.improved is False and r.degenerate is True and r.net_gain == pytest.approx(0.0)


def test_empty_history_is_degenerate_zero():
    r = analyze_run({"benchmark": "gpqa"}, [])
    assert r.n_iters == 0 and r.degenerate is True


def test_tail_plateau_and_iters_to_best():
    r = analyze_run({"benchmark": "math500"}, _cma([0.5, 0.9, 0.9, 0.9]))
    assert r.iters_to_best == 1 and r.tail_plateau == 3


def test_non_monotone_best_is_flagged():
    r = analyze_run({"benchmark": "math500"}, _cma([0.9, 0.8]))
    assert r.best_monotone is False


def test_overfit_gap_from_val_fitness():
    r = analyze_run({"benchmark": "math500", "best_fitness": 0.9, "val_fitness": 0.7}, _cma([0.8, 0.9]))
    assert r.overfit_gap == pytest.approx(0.2)
    r2 = analyze_run({"benchmark": "math500", "best_fitness": 0.9}, _cma([0.8, 0.9]))
    assert r2.overfit_gap is None


def test_run_id_from_run_dir():
    r = analyze_run({"benchmark": "math500", "run_dir": "/x/experiments/math500/warm_shaped"},
                    _cma([0.8, 0.9]))
    assert r.run_id == "warm_shaped"


# --------------------------------------------------------------------------- #
# analyze_runs (cross-run R8 + DoD)
# --------------------------------------------------------------------------- #
def _runs():
    cma = analyze_run({"benchmark": "math500", "best_fitness": 1.0}, _cma([0.9, 1.0]))
    rs = analyze_run({"trainer": "random_search", "benchmark": "math500", "best_fitness": 0.6},
                     [{"trial": 0, "fitness": 0.4, "best_fitness": 0.4},
                      {"trial": 1, "fitness": 0.6, "best_fitness": 0.6}])
    return [cma, rs]


def test_cross_run_ranking_and_dod():
    cross = analyze_runs(_runs())
    ranked = [e["trainer"] for e in cross["rankings"]["math500"]]
    assert ranked == ["sep_cmaes", "random_search"]       # by final fitness desc
    assert cross["observed_optimizer_order"] == ["sep_cmaes", "random_search"]
    assert cross["dod_drives_J_upward"] is True            # the sep-CMA-ES run improved
    assert cross["degenerate_runs"] == []
    assert cross["r8_expected_order"][0] == "sep_cmaes"


def test_dod_fails_when_sepcmaes_run_is_degenerate():
    flat = analyze_run({"benchmark": "math500", "best_fitness": 0.5}, _cma([0.5, 0.5]))
    cross = analyze_runs([flat])
    assert cross["dod_drives_J_upward"] is False and cross["degenerate_runs"]


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_report():
    md = render(_runs())
    assert "Training convergence" in md and "R8 optimizer comparison" in md
    assert "Definition of done" in md and "HOLDS" in md
    assert "sep_cmaes" in md and "random_search" in md
    assert render([]).strip().endswith("(no training runs found)_")
