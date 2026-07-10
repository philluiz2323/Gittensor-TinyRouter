"""Offline, numpy-only tests for the pool-complementarity audit.

Uses hand-built solve matrices with known answers (per docs/ORACLE_CEILING_DIAGNOSTIC
.md §7.1 self-validation: disjoint specialists, identical models) plus the key
selection property — the audit must flag a *redundant duplicate*, never the specialist
that uniquely lifts the oracle. No torch, no network, no GPU.
"""
import sys

import numpy as np
import pytest

from trinity.analysis import analyze_tensor as analyze_tensor_pkg  # re-export check
from trinity.analysis.complementarity import (
    analyze,
    analyze_tensor,
    solve_matrix_from_matrix,
    solve_matrix_from_records,
)


def test_no_torch_imported():
    assert "torch" not in sys.modules, "complementarity audit must import without torch"


def test_reexported_from_package():
    assert analyze_tensor_pkg is analyze_tensor


# --------------------------------------------------------------------------- #
# Canonical cases from the plan doc
# --------------------------------------------------------------------------- #
def test_three_disjoint_specialists():
    # each model solves a distinct question -> oracle 1.0, everyone unique, low overlap.
    B = np.eye(3, dtype=int)
    s = analyze_tensor(B, ["a", "b", "c"])
    assert s.oracle_any == pytest.approx(1.0)
    for pm in s.per_model:
        assert pm.unique_solves == 1
        assert pm.marginal_oracle_contribution == pytest.approx(1 / 3)
        assert pm.leave_one_out_oracle == pytest.approx(2 / 3)
        assert pm.accuracy == pytest.approx(1 / 3)
    # every pair agrees on exactly the 1 question neither solves -> 1/3.
    assert s.pairwise_agreement["a"]["b"] == pytest.approx(1 / 3)


def test_three_identical_models_are_maximally_redundant():
    # all solve q0,q1, none solve q2 -> identical columns.
    B = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    s = analyze_tensor(B, ["a", "b", "c"])
    assert s.oracle_any == pytest.approx(2 / 3)
    for pm in s.per_model:
        assert pm.unique_solves == 0
        assert pm.marginal_oracle_contribution == pytest.approx(0.0)
    # identical -> agreement and kappa are 1.0 everywhere.
    assert s.pairwise_agreement["a"]["b"] == pytest.approx(1.0)
    assert s.cohen_kappa["a"]["b"] == pytest.approx(1.0)
    assert s.cohen_kappa["a"]["a"] == pytest.approx(1.0)


def test_flags_redundant_duplicate_not_the_specialist():
    # dupA and dupB solve the same {q0,q1}; spec uniquely solves q2; q3 unsolved.
    B = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]])
    s = analyze_tensor(B, ["dupA", "dupB", "spec"])
    # the specialist has the highest marginal contribution and must NOT be flagged.
    marg = {pm.model: pm.marginal_oracle_contribution for pm in s.per_model}
    assert marg["spec"] == pytest.approx(0.25)
    assert marg["dupA"] == pytest.approx(0.0) and marg["dupB"] == pytest.approx(0.0)
    # among the zero-contribution duplicates, the first is the recommended swap.
    assert s.most_redundant_model == "dupA"
    assert s.most_redundant_model != "spec"
    assert "dupA" in s.swap_recommendation
    flagged = [pm.model for pm in s.per_model if pm.is_most_redundant]
    assert flagged == ["dupA"]


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
def test_marginal_equals_unique_rate_and_loo_identity():
    rng = np.random.default_rng(0)
    B = (rng.random((25, 4)) < 0.5).astype(int)
    s = analyze_tensor(B, list("wxyz"))
    Q = s.n_questions
    for pm in s.per_model:
        # marginal contribution == unique-solve rate == oracle_any - leave-one-out
        assert pm.marginal_oracle_contribution == pytest.approx(pm.unique_solves / Q)
        assert pm.marginal_oracle_contribution == pytest.approx(s.oracle_any - pm.leave_one_out_oracle)


def test_pairwise_matrices_are_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(1)
    B = (rng.random((30, 3)) < 0.6).astype(int)
    s = analyze_tensor(B, ["a", "b", "c"])
    for m in ("a", "b", "c"):
        assert s.pairwise_agreement[m][m] == pytest.approx(1.0)
        assert s.cohen_kappa[m][m] == pytest.approx(1.0)
    for i in ("a", "b", "c"):
        for j in ("a", "b", "c"):
            assert s.pairwise_agreement[i][j] == pytest.approx(s.pairwise_agreement[j][i])
            assert s.cohen_kappa[i][j] == pytest.approx(s.cohen_kappa[j][i])
            assert s.pairwise_double_fault[i][j] == pytest.approx(s.pairwise_double_fault[j][i])


def test_tensor_reduces_k_samples_by_majority():
    # (Q=2, M=2, K=4): m0 solves q0 in 3/4 (p=.75 -> solved), q1 in 1/4 (p=.25 -> not).
    S = np.zeros((2, 2, 4))
    S[0, 0] = [1, 1, 1, 0]   # q0,m0 -> solved
    S[1, 0] = [1, 0, 0, 0]   # q1,m0 -> not
    S[0, 1] = [0, 0, 0, 0]   # q0,m1 -> not
    S[1, 1] = [1, 1, 1, 1]   # q1,m1 -> solved
    s = analyze_tensor(S, ["m0", "m1"])
    accs = {pm.model: pm.accuracy for pm in s.per_model}
    assert accs["m0"] == pytest.approx(0.5) and accs["m1"] == pytest.approx(0.5)
    assert s.oracle_any == pytest.approx(1.0)  # each q solved by exactly one


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #
def test_empty_and_single_model():
    empty = analyze_tensor(np.zeros((0, 0)), [])
    assert empty.n_questions == 0 and empty.most_redundant_model is None
    solo = analyze_tensor(np.array([[1], [0]]), ["only"])
    assert solo.most_redundant_model is None  # nothing to swap in a 1-model pool
    assert "single-model" in solo.swap_recommendation


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        analyze_tensor(np.zeros((2, 2, 2, 2)), ["a", "b"])


# --------------------------------------------------------------------------- #
# Adapters: matrix dict and QuestionAgreement-like records
# --------------------------------------------------------------------------- #
def test_from_oracle_matrix_dict():
    matrix = {
        "benchmark": "math500",
        "n_samples": 1,
        "tasks": [
            {"id": "q0", "per_model": {"a": [1], "b": [0]}},
            {"id": "q1", "per_model": {"a": [0], "b": [1]}},
        ],
    }
    B, models = solve_matrix_from_matrix(matrix)
    assert models == ["a", "b"]
    assert B.tolist() == [[1, 0], [0, 1]]
    s = analyze(matrix)
    assert s.oracle_any == pytest.approx(1.0)


def test_ragged_matrix_rejected():
    matrix = {"tasks": [
        {"id": "q0", "per_model": {"a": [1], "b": [0]}},
        {"id": "q1", "per_model": {"a": [1], "c": [0]}},  # different model set
    ]}
    with pytest.raises(ValueError):
        solve_matrix_from_matrix(matrix)


class _Rec:
    def __init__(self, per_model_correct):
        self.per_model_correct = per_model_correct

    @property
    def models(self):
        return sorted(self.per_model_correct)


def test_from_records_duck_typed():
    recs = [
        _Rec({"a": 1, "b": 0}),
        _Rec({"a": 0, "b": 1}),
        _Rec({"a": 1, "b": 1}),
    ]
    B, models = solve_matrix_from_records(recs)
    assert models == ["a", "b"]
    assert B.tolist() == [[1, 0], [0, 1], [1, 1]]
    s = analyze_tensor(B, models)
    assert s.oracle_any == pytest.approx(1.0)
