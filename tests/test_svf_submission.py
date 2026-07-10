"""Tests for SVF scale preservation in submission pack/eval paths."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from trinity.coordinator import params as P


def test_unpack_preserves_non_identity_svf():
    """SVF scales must round-trip through θ unpack (not fall back to ones)."""
    spec = P.make_spec()
    rng = np.random.default_rng(0)
    head_W = rng.standard_normal((6, 1024))
    svf = rng.uniform(0.85, 1.15, spec.n_svf)
    theta = P.pack(head_W, svf)
    head2, svf2 = P.unpack(theta, spec)
    assert np.allclose(head2, head_W)
    assert np.allclose(svf2, svf)
    assert not np.allclose(svf2, 1.0)


def test_extract_head_and_svf_from_theta(tmp_path):
    """pack_submission must extract trained SVF scales from best_theta.npy."""
    from pack_submission import extract_head_and_svf

    spec = P.make_spec()
    rng = np.random.default_rng(1)
    head_W = rng.standard_normal((6, 1024))
    svf = rng.uniform(0.9, 1.1, spec.n_svf)
    theta = P.pack(head_W, svf)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(str(run_dir / "best_theta.npy"), theta)

    got_head, got_svf = extract_head_and_svf(run_dir)
    assert got_head.shape == (6, 1024)
    assert got_svf.shape == (spec.n_svf,)
    assert np.allclose(got_head, head_W.astype(np.float32))
    assert np.allclose(got_svf, svf.astype(np.float32))
    assert not np.allclose(got_svf, 1.0)


def test_evaluate_cached_uses_policy_svf():
    """Cached pr_eval must route via policy.decide (encoder + SVF + head)."""
    from pr_eval import _evaluate_cached

    class _FakePolicy:
        def __init__(self):
            self.calls = 0

        def decide(self, prompt, *, sample=False):
            self.calls += 1
            assert sample is False
            return (1 if len(prompt) % 2 == 0 else 0, None)

    policy = _FakePolicy()
    items = [
        {
            "question_text": "ab",
            "benchmark": "math500",
            "correct_answer": "42",
            "model_answers": {"deepseek-v4-pro": "wrong", "glm-5p2": "\\boxed{42}"},
        },
        {
            "question_text": "abc",
            "benchmark": "math500",
            "correct_answer": "7",
            "model_answers": {"deepseek-v4-pro": "\\boxed{7}", "glm-5p2": "wrong"},
        },
    ]
    acc = _evaluate_cached(policy, items, ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"])
    assert acc == 1.0
    assert policy.calls == 2
