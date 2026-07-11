"""Offline tests for the per-model sampling-stability diagnostic (pass@1/pass@K/maj@K).

Synthetic K-sample oracle matrices + a real-data check on the committed math500 matrix.
numpy-only, no torch/network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis import analyze_sampling  # re-export check
from trinity.analysis.sampling import analyze, render, solve_counts

_REPO = Path(__file__).resolve().parents[1]


def _matrix(per_model_by_q, benchmark="math500"):
    """per_model_by_q: list of {model: [0/1,...K]} dicts (one per question)."""
    return {"benchmark": benchmark,
            "tasks": [{"id": f"q{i}", "per_model": pm} for i, pm in enumerate(per_model_by_q)]}


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.analysis.sampling; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


def test_reexported_from_package():
    assert analyze_sampling is analyze


# --------------------------------------------------------------------------- #
# solve_counts
# --------------------------------------------------------------------------- #
def test_solve_counts_decodes():
    solves, k, models = solve_counts(_matrix([{"a": [1, 1, 0], "b": [0, 0, 0]}]))
    assert k == 3 and models == ["a", "b"]
    assert solves.tolist() == [[2, 0]]


def test_solve_counts_ragged_k_and_models_raise():
    with pytest.raises(ValueError):
        solve_counts(_matrix([{"a": [1, 1, 0]}, {"a": [1, 0]}]))          # ragged K
    with pytest.raises(ValueError):
        solve_counts(_matrix([{"a": [1, 0]}, {"b": [1, 0]}]))             # ragged models


# --------------------------------------------------------------------------- #
# analyze — self-consistency gain + rivals-oracle verdict
# --------------------------------------------------------------------------- #
def test_pass_and_majority_metrics():
    # a: 3/5 correct each question -> pass@1 0.6, pass@5 1.0, majority@5 (>=3) 1.0.
    # b: 1/5 correct -> pass@1 0.2, pass@5 1.0, majority@5 0.0.
    s = analyze(_matrix([{"a": [1, 1, 1, 0, 0], "b": [1, 0, 0, 0, 0]},
                         {"a": [1, 1, 1, 0, 0], "b": [1, 0, 0, 0, 0]}]))
    a = next(p for p in s.per_model if p.model == "a")
    b = next(p for p in s.per_model if p.model == "b")
    assert a.pass_at_1 == pytest.approx(0.6) and a.pass_at_k == pytest.approx(1.0)
    assert a.majority_at_k == pytest.approx(1.0) and a.self_consistency_gain == pytest.approx(0.4)
    assert b.pass_at_1 == pytest.approx(0.2) and b.majority_at_k == pytest.approx(0.0)
    # best single's majority (a, 1.0) beats the routing oracle (0.6) -> rivals it.
    assert s.best_majority_model == "a" and s.best_majority == pytest.approx(1.0)
    assert s.routing_oracle == pytest.approx(0.6) and s.majority_rivals_oracle is True


def test_majority_does_not_rival_when_routing_wins():
    # each model solves a different question 5/5; no single model's majority covers both,
    # but the per-query routing oracle does -> routing beats best-single majority.
    s = analyze(_matrix([{"a": [1, 1, 1, 1, 1], "b": [0, 0, 0, 0, 0]},
                         {"a": [0, 0, 0, 0, 0], "b": [1, 1, 1, 1, 1]}]))
    assert s.best_majority == pytest.approx(0.5)      # each model majority-correct on 1/2
    assert s.routing_oracle == pytest.approx(1.0) and s.majority_rivals_oracle is False


def test_k1_has_zero_gain():
    s = analyze(_matrix([{"a": [1], "b": [0]}, {"a": [0], "b": [1]}]))
    for p in s.per_model:
        assert p.pass_at_1 == p.pass_at_k == p.majority_at_k
        assert p.self_consistency_gain == pytest.approx(0.0)


def test_empty_matrix():
    s = analyze({"benchmark": "math500", "tasks": []})
    assert s.n_questions == 0 and s.best_pass1_model is None


# --------------------------------------------------------------------------- #
# real committed data: self-consistency genuinely helps the best model
# --------------------------------------------------------------------------- #
def test_real_math500_matrix_if_present():
    p = _REPO / "experiments" / "final" / "oracle_matrix_math500.json"
    if not p.exists():
        pytest.skip("real oracle_matrix_math500.json not present")
    s = analyze(json.loads(p.read_text()))
    assert s.n_questions == 120 and s.k == 5 and len(s.models) == 3
    assert s.best_majority > s.best_pass1          # majority voting beats a single sample


def test_render_report():
    s = analyze(_matrix([{"a": [1, 1, 1, 0, 0], "b": [1, 0, 0, 0, 0]}]))
    md = render(s)
    assert "sampling stability" in md.lower() and "majority@K" in md
    assert "self-consistency" in md.lower() and "routing oracle" in md
    assert render(analyze({"benchmark": "x", "tasks": []})).strip().endswith("(no matrix data)_")
