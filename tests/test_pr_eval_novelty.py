"""Tests for pr_eval novelty king lookup and policy-based routing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from pr_eval import _king_submission_dir, _routing_decisions


def test_king_submission_dir_scoped_to_benchmark(tmp_path):
    submissions = tmp_path / "submissions"
    math_dir = submissions / "alice" / "1"
    mmlu_dir = submissions / "bob" / "2"
    math_dir.mkdir(parents=True)
    mmlu_dir.mkdir(parents=True)
    np.save(str(math_dir / "head_weights.npy"), np.zeros((6, 1024), dtype=np.float32))
    np.save(str(math_dir / "svf_scales.npy"), np.ones(7168, dtype=np.float32))
    np.save(str(mmlu_dir / "head_weights.npy"), np.zeros((6, 1024), dtype=np.float32))
    np.save(str(mmlu_dir / "svf_scales.npy"), np.ones(7168, dtype=np.float32))

    lb = {
        "benchmarks": {
            "math500": {"best_miner": "alice", "best_generation": 1},
            "mmlu": {"best_miner": "bob", "best_generation": 2},
        }
    }
    assert _king_submission_dir("mmlu", lb, submissions) == mmlu_dir
    assert _king_submission_dir("math500", lb, submissions) == math_dir
    assert _king_submission_dir("gpqa", lb, submissions) is None


def test_routing_decisions_use_query_transcript():
    recorded: list[str] = []

    class _FakePolicy:
        def decide(self, transcript, *, sample=False):
            recorded.append(transcript)
            return 0, "worker"

    items = [{"question_text": "What is 2+2?", "benchmark": "math500", "correct_answer": "4"}]
    _routing_decisions(_FakePolicy(), items, ref_count=1)
    assert recorded[0].startswith("QUERY:\nWhat is 2+2?")
