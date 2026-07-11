"""Regression: a non-finite entry must not count as a moved (trained) parameter (issue #229).

``n_moved`` was ``size - n_at_init``, but ``n_at_init`` excludes non-finite entries,
so a single NaN in an otherwise-at-init block was counted as "moved" and flipped the
``head_trained``/``svf_trained`` verdict — suppressing the exact "still at init"
warning this module exists to raise. Pure numpy — no network, no GPU, no torch.
"""
from __future__ import annotations

import numpy as np

from trinity.coordinator import params as P
from trinity.theta_inspect import BlockStats, inspect_theta


def test_nan_on_untrained_head_is_not_trained():
    spec = P.make_spec()
    theta = P.initial_theta(spec)  # head=0, svf=1 -> untrained
    theta[0] = np.nan              # one blown-up head coordinate
    r = inspect_theta(theta, spec)
    assert r.head.n_moved == 0
    assert r.head_trained is False
    assert any("uniform policy" in w for w in r.warnings)
    assert any("non-finite" in w for w in r.warnings)


def test_nan_on_untrained_svf_is_not_trained():
    spec = P.make_spec()
    theta = P.initial_theta(spec)
    theta[-1] = np.inf             # one blown-up svf scale
    r = inspect_theta(theta, spec)
    assert r.svf.n_moved == 0
    assert r.svf_trained is False
    assert any("SLM was not adapted" in w for w in r.warnings)


def test_genuinely_trained_block_still_reads_trained():
    spec = P.make_spec()
    theta = P.initial_theta(spec)
    theta[:spec.n_head] = 0.5      # head genuinely moved off 0
    r = inspect_theta(theta, spec)
    assert r.head.n_moved == spec.n_head
    assert r.head_trained is True


def test_n_moved_excludes_only_nonfinite_not_finite_moves():
    # 5 at-init, 3 finite-moved, 2 non-finite -> n_moved == 3.
    bs = BlockStats(
        name="x", size=10, l2_norm=0.0, max_abs=0.0, mean=0.0,
        n_nonfinite=2, n_at_init=5, dist_from_init=0.0,
    )
    assert bs.n_moved == 3
    assert bs.at_init is False


def test_all_nonfinite_block_is_not_trained():
    # A fully non-finite block has zero finite moves -> not trained (at_init True).
    bs = BlockStats(
        name="x", size=6, l2_norm=0.0, max_abs=0.0, mean=0.0,
        n_nonfinite=6, n_at_init=0, dist_from_init=0.0,
    )
    assert bs.n_moved == 0
    assert bs.at_init is True
