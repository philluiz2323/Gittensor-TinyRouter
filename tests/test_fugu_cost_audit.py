"""Offline tests for the Fugu Conductor cost & worker-utilization audit
(trinity.fugu.cost_audit).

Synthetic baseline cost blocks + a real-data check on the committed
fugu_baseline_math500.json. No torch, no network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.fugu import analyze_cost  # re-export check
from trinity.fugu.cost_audit import analyze, render

_REPO = Path(__file__).resolve().parents[1]


def _worker(pt, ct, usd):
    return {"prompt_tokens": pt, "completion_tokens": ct, "usd": usd}


def _baseline(per_model, *, n_tasks=100, accuracy=0.9, spend=1.0, llm_calls=300,
              parse_rate=0.97, prompt=0, completion=0):
    return {
        "benchmark": "math500", "conductor_model": "cond", "n_tasks": n_tasks,
        "accuracy": accuracy, "parse_rate": parse_rate,
        "cost": {"spend_usd": spend, "llm_calls": llm_calls, "prompt_tokens": prompt,
                 "completion_tokens": completion, "per_model": per_model},
    }


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.fugu.cost_audit; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


def test_reexported_from_package():
    assert analyze_cost is analyze


# --------------------------------------------------------------------------- #
# concentration -> test-time compute, not routing
# --------------------------------------------------------------------------- #
def test_concentrated_pool_is_test_time_compute():
    b = _baseline({
        "<conductor>": _worker(100, 100, 0.10),
        "a": _worker(1000, 9000, 0.90),
        "b": _worker(10, 10, 0.001),
        "c": _worker(10, 10, 0.001),
    }, prompt=1120, completion=9120)
    s = analyze(b)
    assert s.most_used_worker == "a" and s.workers[0].model == "a"
    assert s.workers[0].completion_share == pytest.approx(9000 / 9020)
    assert s.effective_workers < 1.5 and s.lift_is_test_time_compute is True
    assert s.calls_per_task == pytest.approx(3.0)
    assert s.usd_per_correct == pytest.approx(1.0 / (0.9 * 100))
    assert 0.0 < s.conductor_token_share < 1.0


def test_balanced_pool_is_routing():
    b = _baseline({"a": _worker(500, 1000, 0.3), "b": _worker(500, 1000, 0.3),
                   "c": _worker(500, 1000, 0.3)})
    s = analyze(b)
    assert s.effective_workers == pytest.approx(3.0)
    assert s.lift_is_test_time_compute is False
    for w in s.workers:
        assert w.completion_share == pytest.approx(1 / 3)


def test_empty_cost_degrades():
    s = analyze({"benchmark": "math500"})
    assert s.workers == [] and s.effective_workers == 0.0 and s.usd_per_correct is None


# --------------------------------------------------------------------------- #
# real committed artifact
# --------------------------------------------------------------------------- #
def test_real_fugu_baseline_if_present():
    p = _REPO / "experiments" / "final" / "fugu_baseline_math500.json"
    if not p.exists():
        pytest.skip("real fugu_baseline_math500.json not present")
    s = analyze(json.loads(p.read_text()))
    assert s.most_used_worker == "deepseek-v4-pro"       # the Conductor sent ~all work there
    assert s.effective_workers < 1.5 and s.lift_is_test_time_compute is True
    assert s.spend_usd == pytest.approx(1.0975, abs=1e-3)


def test_render_report():
    s = analyze(_baseline({"<conductor>": _worker(100, 100, 0.1), "a": _worker(1000, 9000, 0.9),
                           "b": _worker(10, 10, 0.001)}, prompt=1110, completion=9110))
    md = render(s)
    assert "cost & worker-utilization" in md.lower() and "effective workers" in md
    assert "test-time compute, not routing" in md
    assert render(analyze({"benchmark": "x"})).strip().endswith("(no cost data)_")
