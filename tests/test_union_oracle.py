"""Offline tests for the cross-benchmark union oracle-headroom analysis.

Synthetic solve matrices (the oracle_matrix schema), numpy-only, no torch/network. The
headline case: each model wins a different benchmark, so per-benchmark headroom is 0 yet
the equally-weighted UNION has real headroom — the "win is cross-task" thesis.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis import union_oracle as union_oracle_pkg  # re-export check
from trinity.analysis.union_oracle import (
    oracle_from_matrix,
    relative_error_reduction,
    render,
    union_oracle,
)

_REPO = Path(__file__).resolve().parents[1]


def _matrix(benchmark, per_query):
    """per_query: list of {model: 0/1} dicts -> an oracle_matrix dict (K=1)."""
    tasks = [{"id": f"q{i}", "per_model": {m: [v] for m, v in pm.items()}}
             for i, pm in enumerate(per_query)]
    return {"benchmark": benchmark, "tasks": tasks}


def test_module_imports_without_torch():
    # A global sys.modules check is unreliable where torch IS installed (CI): another
    # test imports it into the shared process. Verify in a CLEAN subprocess that
    # importing this module alone never pulls in torch.
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.analysis.union_oracle; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


def test_reexported_from_package():
    assert union_oracle_pkg is union_oracle


# --------------------------------------------------------------------------- #
# relative_error_reduction (SPEC §6.3)
# --------------------------------------------------------------------------- #
def test_rer_formula():
    assert relative_error_reduction(0.9, 0.8) == pytest.approx(0.5)   # (.9-.8)/(1-.8)
    assert relative_error_reduction(0.8, 0.8) == pytest.approx(0.0)
    assert relative_error_reduction(1.0, 0.5) == pytest.approx(1.0)
    assert relative_error_reduction(0.5, 1.0) is None                 # no residual error


# --------------------------------------------------------------------------- #
# oracle_from_matrix
# --------------------------------------------------------------------------- #
def test_disjoint_specialists_full_headroom():
    o = oracle_from_matrix(_matrix("math", [{"a": 1, "b": 0, "c": 0},
                                            {"a": 0, "b": 1, "c": 0},
                                            {"a": 0, "b": 0, "c": 1}]))
    assert o.best_single == pytest.approx(1 / 3)
    assert o.routing_oracle == pytest.approx(1.0)
    assert o.headroom == pytest.approx(2 / 3)
    assert o.disagreement_rate == pytest.approx(1.0)


def test_identical_models_zero_headroom():
    o = oracle_from_matrix(_matrix("mmlu", [{"a": 1, "b": 1}, {"a": 0, "b": 0}]))
    assert o.headroom == pytest.approx(0.0) and o.disagreement_rate == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# union_oracle — the cross-task thesis
# --------------------------------------------------------------------------- #
def test_union_headroom_is_cross_task():
    # x wins benchmark A, y wins benchmark B: each benchmark alone has 0 headroom
    # (one model dominates), but no fixed single model is good on BOTH.
    a = _matrix("math", [{"x": 1, "y": 0}, {"x": 1, "y": 0}])
    b = _matrix("mmlu", [{"x": 0, "y": 1}, {"x": 0, "y": 1}])
    s = union_oracle([a, b])
    assert [bo.headroom for bo in s.benchmarks] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert s.best_single == pytest.approx(0.5)        # best FIXED single over the union
    assert s.routing_oracle == pytest.approx(1.0)     # per-query router picks the specialist
    assert s.union_headroom == pytest.approx(0.5)     # the cross-task gain
    assert s.oracle_rer == pytest.approx(1.0)         # closes all residual error


def test_union_uses_equal_benchmark_weighting_not_query_count():
    # A: 4 queries x-solves-all ; B: 1 query y-solves. Query-weighted best single would be
    # x at 0.8; equal-weight is 0.5 (each benchmark counts once).
    a = _matrix("math", [{"x": 1, "y": 0}] * 4)
    b = _matrix("mmlu", [{"x": 0, "y": 1}])
    s = union_oracle([a, b])
    assert s.best_single == pytest.approx(0.5)
    assert s.equal_weight_per_model_accuracy["x"] == pytest.approx(0.5)


def test_union_ragged_model_set_rejected():
    a = _matrix("math", [{"x": 1, "y": 0}])
    b = _matrix("mmlu", [{"x": 0, "z": 1}])   # different model set
    with pytest.raises(ValueError):
        union_oracle([a, b])


def test_union_empty():
    s = union_oracle([])
    assert s.n_benchmarks == 0 and s.best_single_model is None


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_report():
    a = _matrix("math", [{"x": 1, "y": 0}, {"x": 1, "y": 0}])
    b = _matrix("mmlu", [{"x": 0, "y": 1}, {"x": 0, "y": 1}])
    md = render(union_oracle([a, b]))
    assert "union oracle headroom" in md.lower()
    assert "UNION (equal-weight, n=2)" in md and "oracle RER" in md
    assert "cross-task" in md
    assert render(union_oracle([])).strip().endswith("(no benchmark matrices)_")
