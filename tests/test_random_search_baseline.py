"""Offline, numpy-only tests for the Random-Search baseline (docs/SPEC.md R8, M4).

Covers the pure-numpy core of ``trinity.optim.baselines`` — the sampler,
budget-match arithmetic, keep-best, and the synthetic-objective driver — plus the
``RandomSearchTrainer`` construction/guards. No torch, no pool, no GPU: the RS core
injects its fitness exactly like ``sep_cmaes.run`` (see test_sep_cmaes_seed.py).
"""
import asyncio
import sys

import numpy as np
import pytest

from trinity.optim import RandomSearchTrainer as RS_from_pkg  # re-export check
from trinity.optim.base import BaseTrainer
from trinity.optim.baselines import (
    RS_SAMPLE_HIGH,
    RS_SAMPLE_LOW,
    RS_TRIALS_PER_CANDIDATE,
    RandomSearchTrainer,
    budget_matched_candidates,
    run_random_search,
    sample_candidates,
    select_best,
)


def test_no_torch_imported():
    assert "torch" not in sys.modules, "RS baseline must import cleanly without torch"


def test_reexported_from_package():
    assert RS_from_pkg is RandomSearchTrainer
    assert issubclass(RandomSearchTrainer, BaseTrainer)


# --------------------------------------------------------------------------- #
# budget_matched_candidates
# --------------------------------------------------------------------------- #
def test_budget_match_spec_numbers():
    # docs/SPEC.md: λ=33, m_cma=16, T=60 -> 31,680 env budget; m_RS=32 -> 990 draws.
    assert budget_matched_candidates(33, 16, 60, 32) == 990
    assert RS_TRIALS_PER_CANDIDATE == 32


def test_budget_match_default_trials():
    assert budget_matched_candidates(33, 16, 60) == budget_matched_candidates(33, 16, 60, 32)


def test_budget_match_floors():
    assert budget_matched_candidates(1, 1, 10, 3) == 3  # 10 // 3


def test_budget_match_rejects_bad_trials():
    with pytest.raises(ValueError):
        budget_matched_candidates(33, 16, 60, 0)
    with pytest.raises(ValueError):
        budget_matched_candidates(-1, 16, 60, 32)


# --------------------------------------------------------------------------- #
# sample_candidates
# --------------------------------------------------------------------------- #
def test_sample_shape_and_bounds():
    c = sample_candidates(7, 20, seed=0)
    assert c.shape == (20, 7)
    assert np.all(c >= RS_SAMPLE_LOW) and np.all(c < RS_SAMPLE_HIGH)


def test_sample_is_seed_reproducible():
    assert np.array_equal(sample_candidates(5, 8, seed=3), sample_candidates(5, 8, seed=3))


def test_sample_distinct_seeds_differ():
    assert not np.array_equal(sample_candidates(5, 8, seed=0), sample_candidates(5, 8, seed=1))


def test_sample_does_not_touch_global_rng():
    # default_rng must not perturb the legacy global stream sep_cmaes relies on.
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    sample_candidates(4, 4, seed=999)
    after = np.random.random()
    assert before == after


def test_sample_custom_range():
    c = sample_candidates(3, 50, seed=1, low=-2.0, high=-1.0)
    assert np.all(c >= -2.0) and np.all(c < -1.0)


@pytest.mark.parametrize(
    "n,num,low,high",
    [(0, 5, -0.5, 0.5), (5, 0, -0.5, 0.5), (5, 5, 0.5, -0.5), (5, 5, 0.3, 0.3)],
)
def test_sample_validation(n, num, low, high):
    with pytest.raises(ValueError):
        sample_candidates(n, num, seed=0, low=low, high=high)


# --------------------------------------------------------------------------- #
# select_best
# --------------------------------------------------------------------------- #
def test_select_best_picks_argmax():
    cands = np.array([[0.0], [1.0], [2.0]])
    x, f, i = select_best(cands, [0.1, 0.9, 0.4])
    assert i == 1 and f == pytest.approx(0.9) and x.tolist() == [1.0]


def test_select_best_returns_a_copy():
    cands = np.array([[5.0, 6.0], [7.0, 8.0]])
    x, _, _ = select_best(cands, [0.0, 1.0])
    x[0] = -999.0
    assert cands[1, 0] == 7.0  # input untouched


def test_select_best_ties_take_first():
    _, _, i = select_best(np.array([[0.0], [1.0], [2.0]]), [0.5, 0.5, 0.5])
    assert i == 0


def test_select_best_validation():
    with pytest.raises(ValueError):
        select_best(np.zeros((0, 3)), [])
    with pytest.raises(ValueError):
        select_best(np.zeros((3, 2)), [1.0, 2.0])  # length mismatch


# --------------------------------------------------------------------------- #
# run_random_search (the offline driver, synthetic objective)
# --------------------------------------------------------------------------- #
def _sphere(target):
    return lambda x: -float(np.dot(x - target, x - target))


def test_run_best_is_monotone_and_maximal():
    target = np.full(6, 0.25)
    best_x, best_f, hist = run_random_search(_sphere(target), 6, num_candidates=100, seed=0)
    # best-so-far never decreases
    assert all(hist[i]["best_fitness"] <= hist[i + 1]["best_fitness"] + 1e-12 for i in range(len(hist) - 1))
    # returned best equals the max over all trials
    assert best_f == pytest.approx(max(h["fitness"] for h in hist))
    assert len(hist) == 100
    # best_x actually achieves best_f
    assert _sphere(target)(best_x) == pytest.approx(best_f)


def test_run_is_seed_reproducible():
    obj = _sphere(np.zeros(6))
    a = run_random_search(obj, 6, num_candidates=50, seed=0)
    b = run_random_search(obj, 6, num_candidates=50, seed=0)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_run_diverges_across_seeds():
    obj = _sphere(np.zeros(6))
    a = run_random_search(obj, 6, num_candidates=50, seed=0)
    b = run_random_search(obj, 6, num_candidates=50, seed=1)
    assert not np.array_equal(a[0], b[0])


def test_run_more_trials_never_worse():
    # RS keep-best is monotone in the sample: a superset of draws can only improve.
    target = np.full(8, 0.1)
    _, f_small, _ = run_random_search(_sphere(target), 8, num_candidates=20, seed=0)
    _, f_big, _ = run_random_search(_sphere(target), 8, num_candidates=200, seed=0)
    assert f_big >= f_small - 1e-12


# --------------------------------------------------------------------------- #
# RandomSearchTrainer construction + guards (offline; no pool touched)
# --------------------------------------------------------------------------- #
def test_trainer_records_config():
    t = RandomSearchTrainer(trials_per_candidate=16, low=-1.0, high=1.0, seed=7)
    assert (t.trials_per_candidate, t.low, t.high, t.seed) == (16, -1.0, 1.0, 7)


def test_trainer_defaults():
    t = RandomSearchTrainer()
    assert t.trials_per_candidate == RS_TRIALS_PER_CANDIDATE
    assert t.low == RS_SAMPLE_LOW and t.high == RS_SAMPLE_HIGH


@pytest.mark.parametrize("kw", [{"trials_per_candidate": 0}, {"low": 0.5, "high": -0.5}])
def test_trainer_validation(kw):
    with pytest.raises(ValueError):
        RandomSearchTrainer(**kw)


def test_train_requires_spec_and_run_dir(tmp_path):
    t = RandomSearchTrainer()
    # spec missing -> guard fires before any pool/policy use.
    with pytest.raises(ValueError, match="spec"):
        asyncio.run(t.train(None, None, [], spec=None, run_dir=tmp_path))
    # run_dir missing -> guard fires too.
    with pytest.raises(ValueError, match="run_dir"):
        asyncio.run(t.train(None, None, [], spec=object(), run_dir=None))
