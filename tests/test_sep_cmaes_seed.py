"""Seed determinism for the separable CMA-ES wrapper.

Regression cover for the bug where ``SepCMAES(seed=0)`` -- the default at every
level of the repo -- sampled from a wall-clock seed. pycma documents its ``seed``
option as: *"`None` and `0` equate to `time`, `np.nan` means 'do nothing'"*, so
forwarding a literal ``0`` opted the "deterministic wrapper" out of determinism.

The fix seeds ``numpy.random`` directly and passes NaN to pycma. Because pycma
implements an honoured integer seed as exactly ``np.random.seed(seed)``, streams
for previously-working seeds must be unchanged -- that invariant is pinned by
``test_nonzero_seed_stream_matches_pycmas_own_seeding`` below.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.optim.sep_cmaes import SepCMAES, run

cma = pytest.importorskip("cma", reason="pycma is required to build SepCMAES")

_N = 6


def _first_population(seed: int) -> np.ndarray:
    return np.asarray(SepCMAES(n=_N, sigma0=0.1, seed=seed, popsize=5).ask())


# --------------------------------------------------------------------------- #
# The bug: seed=0 must be reproducible
# --------------------------------------------------------------------------- #
def test_default_seed_is_reproducible():
    """The regression: seed 0 previously drew a clock-seeded population."""
    assert np.allclose(_first_population(0), _first_population(0))


def test_default_seed_is_actually_zero():
    """Guard the default, since the bug only bit because the default was 0."""
    assert SepCMAES(n=_N).seed == 0


def test_constructor_default_matches_explicit_zero():
    a = np.asarray(SepCMAES(n=_N, sigma0=0.1, popsize=5).ask())
    b = _first_population(0)
    assert np.allclose(a, b)


@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 2**32 - 1])
def test_every_seed_is_reproducible(seed):
    assert np.allclose(_first_population(seed), _first_population(seed))


def test_distinct_seeds_give_distinct_populations():
    """Determinism must not collapse the seed space onto one stream."""
    assert not np.allclose(_first_population(0), _first_population(1))
    assert not np.allclose(_first_population(1), _first_population(2))


def test_zero_and_one_are_not_aliased():
    """A naive `seed or 1` fix would silently alias these two."""
    assert not np.allclose(_first_population(0), _first_population(1))


# --------------------------------------------------------------------------- #
# Backwards compatibility: seeds pycma already honoured keep their exact stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 7, 12345])
def test_nonzero_seed_stream_matches_pycmas_own_seeding(seed):
    """pycma's honoured `seed=k` is `np.random.seed(k)`; we must reproduce it.

    This pins the claim that the fix is behavior-preserving for any seed that
    already worked -- archived fitness curves stay reproducible.
    """
    np.random.seed(seed)
    reference = cma.CMAEvolutionStrategy(
        [0.0] * _N,
        0.1,
        {"CMA_diagonal": True, "popsize": 5, "seed": np.nan, "verbose": -9},
    )
    expected = np.asarray(reference.ask())

    assert np.allclose(_first_population(seed), expected)


# --------------------------------------------------------------------------- #
# Seed validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-1, -12345, 2**32])
def test_out_of_range_seed_is_rejected(bad):
    with pytest.raises(ValueError, match="seed must be in"):
        SepCMAES(n=_N, seed=bad)


def test_seed_is_recorded_verbatim():
    assert SepCMAES(n=_N, seed=4321).seed == 4321


# --------------------------------------------------------------------------- #
# End-to-end: `run` follows the same trajectory twice
# --------------------------------------------------------------------------- #
def _sphere(theta: np.ndarray) -> float:
    """Deterministic synthetic fitness (maximized), as in smoke test S7."""
    return -float(np.sum(np.square(theta)))


def test_run_is_reproducible_at_the_default_seed():
    best_a, fit_a, hist_a = run(_sphere, n=_N, sigma0=0.1, seed=0, maxiter=3, popsize=5)
    best_b, fit_b, hist_b = run(_sphere, n=_N, sigma0=0.1, seed=0, maxiter=3, popsize=5)

    assert np.allclose(best_a, best_b)
    assert fit_a == fit_b
    assert [h["best_fitness"] for h in hist_a] == [h["best_fitness"] for h in hist_b]


def test_run_diverges_across_seeds():
    _, _, hist_a = run(_sphere, n=_N, sigma0=0.1, seed=0, maxiter=3, popsize=5)
    _, _, hist_b = run(_sphere, n=_N, sigma0=0.1, seed=1, maxiter=3, popsize=5)

    assert [h["best_fitness"] for h in hist_a] != [h["best_fitness"] for h in hist_b]
