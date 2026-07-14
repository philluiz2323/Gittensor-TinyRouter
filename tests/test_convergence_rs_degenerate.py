"""Regression: an improving Random-Search run must not be flagged degenerate.

convergence.analyze_run treated ``signal[-1] < signal[0]`` as a regression for
EVERY trainer. But the Random-Search ``signal`` is the i.i.d. fitness of each
independently sampled theta -- not a monotone population objective -- so its last
draw being below its first is noise, not a "converged to a bad policy" collapse.
An improving RS run was therefore flagged degenerate, contradicting improved=True.
No network, no GPU.
"""
from __future__ import annotations

from trinity.analysis.convergence import analyze_run, analyze_runs


def _rs_run():
    # best-so-far climbs 0.5 -> 0.8 (improved), but the LAST random draw (0.3) sits
    # below the FIRST (0.5) -- exactly the case the bug mis-flagged.
    return analyze_run(
        {"trainer": "random_search", "benchmark": "math500"},
        [{"trial": 0, "fitness": 0.5, "best_fitness": 0.5},
         {"trial": 1, "fitness": 0.8, "best_fitness": 0.8},
         {"trial": 2, "fitness": 0.3, "best_fitness": 0.8}],
    )


def test_improving_random_search_is_not_degenerate():
    r = _rs_run()
    assert r.improved is True
    assert r.degenerate is False
    assert analyze_runs([r])["degenerate_runs"] == []


def test_flat_random_search_is_still_degenerate():
    # No improvement over start -> degenerate via `not improved`, regardless of draws.
    r = analyze_run(
        {"trainer": "random_search", "benchmark": "math500"},
        [{"trial": 0, "fitness": 0.5, "best_fitness": 0.5},
         {"trial": 1, "fitness": 0.4, "best_fitness": 0.5}],
    )
    assert r.improved is False and r.degenerate is True


def test_sep_cmaes_population_collapse_is_still_degenerate():
    # The population-mean objective ending below its start IS a real collapse,
    # even when best-so-far rose -- this must keep firing for sep-CMA-ES.
    r = analyze_run(
        {"trainer": "sep_cmaes", "benchmark": "math500"},
        [{"generation": 0, "gen_mean_fitness": 0.6, "best_fitness": 0.6},
         {"generation": 1, "gen_mean_fitness": 0.2, "best_fitness": 0.7}],
    )
    assert r.improved is True          # best-so-far rose
    assert r.degenerate is True        # but the population mean collapsed


def test_healthy_sep_cmaes_is_not_degenerate():
    r = analyze_run(
        {"trainer": "sep_cmaes", "benchmark": "math500"},
        [{"generation": 0, "gen_mean_fitness": 0.4, "best_fitness": 0.4},
         {"generation": 1, "gen_mean_fitness": 0.6, "best_fitness": 0.6}],
    )
    assert r.degenerate is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
