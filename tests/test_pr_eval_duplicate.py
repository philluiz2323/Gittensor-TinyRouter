"""Offline unit tests for the pr_eval duplicate-detection gate (anti-cheat Gate 3).

No GPU/API/network (torch is imported lazily inside pr_eval, never at module load).

Regression target: Gate 3 must compare the trained *routing head* — the artifact
"original work" refers to. The SVF singular-value scales start at the identity
(all 1.0) and barely move, so every submission's SVF block is near-identical to
every other's. The previous gate folded head + SVF into one cosine, letting the
near-constant (and larger) SVF block dominate — so a copied head could slip under
the 0.999 threshold just by re-rolling its SVF scales.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pr_eval.py"
_spec = importlib.util.spec_from_file_location("pr_eval", _SCRIPT)
pr_eval = importlib.util.module_from_spec(_spec)
sys.modules["pr_eval"] = pr_eval
_spec.loader.exec_module(pr_eval)

_HEAD_SHAPE = (6, 1024)
_N_SVF = 7168


def _write_submission(root: Path, miner: str, gen: int,
                      head: np.ndarray, svf: np.ndarray) -> None:
    d = root / miner / str(gen)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "head_weights.npy", head.astype(np.float32))
    np.save(d / "svf_scales.npy", svf.astype(np.float32))


def _rand_head(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.05, _HEAD_SHAPE)


def _near_identity_svf(seed: int, std: float = 0.02) -> np.ndarray:
    return 1.0 + np.random.default_rng(seed).normal(0, std, _N_SVF)


@pytest.fixture(autouse=True)
def _isolate_leaderboard(monkeypatch):
    """Gate 3 also consults the leaderboard 'king'; keep tests hermetic."""
    monkeypatch.setattr(pr_eval, "_load_leaderboard", lambda: {"benchmarks": {}})


def test_copied_head_with_rerolled_svf_is_caught(tmp_path):
    """The core evasion: copy a rival's head verbatim, re-roll only the SVF block.

    Head cosine is 1.0 (a literal copy), but the concatenated cosine the old gate
    used falls to ~0.9986 (< 0.999) because the differing SVF block dominates. The
    head-based gate must still reject it.
    """
    head = _rand_head(1)
    _write_submission(tmp_path, "alice", 1, head, _near_identity_svf(10))

    # Attacker: identical head, freshly re-rolled SVF scales.
    err = pr_eval._check_duplicate(head, _near_identity_svf(999, std=0.05),
                                   tmp_path, "bob", 1)
    assert err is not None and err.startswith("duplicate_of_alice_gen_1")


def test_exact_full_copy_is_caught(tmp_path):
    head, svf = _rand_head(1), _near_identity_svf(10)
    _write_submission(tmp_path, "alice", 1, head, svf)
    err = pr_eval._check_duplicate(head, svf, tmp_path, "bob", 1)
    assert err is not None and err.startswith("duplicate_of_alice_gen_1")


def test_distinct_heads_are_not_flagged(tmp_path):
    """Two independently trained heads (both near-identity SVF) must pass."""
    _write_submission(tmp_path, "alice", 1, _rand_head(1), _near_identity_svf(10))
    err = pr_eval._check_duplicate(_rand_head(2), _near_identity_svf(11),
                                   tmp_path, "bob", 1)
    assert err is None


def test_self_same_gen_is_skipped(tmp_path):
    """A submission is never flagged as a duplicate of itself."""
    head, svf = _rand_head(1), _near_identity_svf(10)
    _write_submission(tmp_path, "alice", 1, head, svf)
    assert pr_eval._check_duplicate(head, svf, tmp_path, "alice", 1) is None


def test_mismatched_head_shape_is_skipped(tmp_path):
    """A prior submission with a different head geometry is not comparable."""
    _write_submission(tmp_path, "alice", 1, _rand_head(1), _near_identity_svf(10))
    wrong_shape_head = np.random.default_rng(3).normal(0, 0.05, (6, 512))
    err = pr_eval._check_duplicate(wrong_shape_head, _near_identity_svf(11),
                                   tmp_path, "bob", 1)
    assert err is None


def test_king_copy_is_caught(tmp_path, monkeypatch):
    """A head copied from the leaderboard king is rejected even with new SVF.

    (When the king's files live under ``submissions/`` the general directory scan
    catches the copy first; the dedicated king path is the fallback. Either way a
    copy of the king's head must be flagged.)
    """
    head = _rand_head(1)
    _write_submission(tmp_path, "alice", 2, head, _near_identity_svf(10))
    monkeypatch.setattr(pr_eval, "_load_leaderboard", lambda: {
        "benchmarks": {"math500": {"best_miner": "alice", "best_generation": 2}}
    })
    err = pr_eval._check_duplicate(head, _near_identity_svf(999, std=0.05),
                                   tmp_path, "bob", 1)
    assert err is not None and "alice_gen_2" in err
