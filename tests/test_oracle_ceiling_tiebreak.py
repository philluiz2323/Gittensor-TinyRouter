"""The cross-fit oracle tie-break favours model 0, as documented.

`crossfit_oracle_and_best` breaks argmax ties on the selection half with a
deterministic per-model jitter. The docstring promises "model 0 favoured"; an
increasing jitter ramp would instead hand every tie to the last-indexed model,
biasing the oracle by pool order. Selection-half ties are common at low K
(n_a = K//2 lands sel on a coarse grid), so the direction is load-bearing.
Offline / numpy-only: no GPU, no network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracle_ceiling.py"
_spec = importlib.util.spec_from_file_location("oracle_ceiling", _SCRIPT)
oc = importlib.util.module_from_spec(_spec)
sys.modules["oracle_ceiling"] = oc
_spec.loader.exec_module(oc)


def _tie_tensor(Q: int = 20) -> np.ndarray:
    """Q identical queries where model 0 is best but ties model 1 on half A.

    Per query, K=2 samples:
      model 0 = [1, 1]  (perfect, p=1.0)
      model 1 = [1, 0]  (p=0.5)
      model 2 = [0, 0]  (p=0.0)

    When the selection sample is the shared '1', models 0 and 1 tie on half A;
    the tie-break decides which is scored on half B. Favouring model 0 recovers
    its true accuracy (1.0); favouring the last-tied model (1) scores 0 on B and
    deflates the oracle toward 0.5.
    """
    query = np.array([[[1, 1], [1, 0], [0, 0]]], dtype=float)
    return np.tile(query, (Q, 1, 1))


def test_crossfit_oracle_breaks_ties_toward_model_zero():
    S = _tie_tensor()
    oracle, _best = oc.crossfit_oracle_and_best(S, n_splits=400, seed=0)
    # Model 0 is genuinely the best (p=1.0); a correct tie-break routes ties to it
    # and recovers 1.0. The last-model tie-break would land near 0.5.
    assert oracle == pytest.approx(1.0, abs=1e-9)


def test_oracle_is_stable_across_seeds():
    S = _tie_tensor()
    a, _ = oc.crossfit_oracle_and_best(S, n_splits=300, seed=0)
    b, _ = oc.crossfit_oracle_and_best(S, n_splits=300, seed=12345)
    assert a == pytest.approx(1.0, abs=1e-9)
    assert b == pytest.approx(1.0, abs=1e-9)


def test_oracle_never_below_best_single_after_flooring():
    # compute_stats floors the oracle at the cross-fit best_single; the tie-break
    # fix keeps the raw oracle from dipping below it on tie-heavy inputs.
    S = _tie_tensor()
    stats = oc.compute_stats(S, crossfit_splits=200, seed=0)
    assert stats.routing_oracle >= stats.best_single_crossfit


def test_wrapper_matches_paired_oracle():
    S = _tie_tensor()
    only = oc.routing_oracle_crossfit(S, n_splits=200, seed=7)
    paired, _ = oc.crossfit_oracle_and_best(S, n_splits=200, seed=7)
    assert only == pytest.approx(paired, abs=1e-12)
