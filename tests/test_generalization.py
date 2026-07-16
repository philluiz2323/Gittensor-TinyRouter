"""Offline tests for the eval->audit generalization (overfit) gap report.

Synthetic eval/audit dicts, no torch/network. Includes a no-drift test that loads
scripts/pr_eval.py and pins this report's thresholds equal to GATE 5's.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from trinity.analysis import analyze_pair as analyze_pair_pkg  # re-export check
from trinity.analysis import generalization as gen
from trinity.analysis.generalization import analyze_pair, overfit_verdict, render

_REPO = Path(__file__).resolve().parents[1]


def test_no_torch_imported():
    assert "torch" not in sys.modules


def test_reexported_from_package():
    assert analyze_pair_pkg is analyze_pair


def _result(trinity, singles=None, bench="math500"):
    r = {"TRINITY": trinity}
    if singles:
        r.update({f"single::{m}": v for m, v in singles.items()})
    return {"benchmark": bench, "results": r}


# --------------------------------------------------------------------------- #
# no-drift: thresholds must equal pr_eval GATE 5
# --------------------------------------------------------------------------- #
def test_thresholds_match_pr_eval_gate5():
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("pr_eval", _REPO / "scripts" / "pr_eval.py")
    pe = importlib.util.module_from_spec(spec)
    sys.modules["pr_eval"] = pe
    spec.loader.exec_module(pe)
    assert gen.OVERFIT_HARD_REJECT == pe._OVERFIT_HARD_REJECT
    assert gen.OVERFIT_PENALTY == pe._OVERFIT_PENALTY


# --------------------------------------------------------------------------- #
# overfit_verdict
# --------------------------------------------------------------------------- #
def test_overfit_verdict_thresholds():
    assert overfit_verdict(0.02) == ("ok", 1.0)
    assert overfit_verdict(-0.05) == ("ok", 1.0)         # audit >= eval, no overfit
    assert overfit_verdict(0.10) == ("penalty", 0.85)    # 0.08 < gap <= 0.15
    assert overfit_verdict(0.20) == ("reject", 0.0)      # gap > 0.15
    assert overfit_verdict(0.08) == ("ok", 1.0)          # boundary: not > 0.08


# --------------------------------------------------------------------------- #
# analyze_pair
# --------------------------------------------------------------------------- #
def test_analyze_pair_ok():
    g = analyze_pair(_result(0.85, {"deepseek": 0.80}), _result(0.82, {"deepseek": 0.79}))
    assert g.benchmark == "math500" and g.gap == pytest.approx(0.03) and g.verdict == "ok"
    assert g.eval_best_single == pytest.approx(0.80) and g.audit_best_single == pytest.approx(0.79)


def test_analyze_pair_penalty_and_reject():
    assert analyze_pair(_result(0.85), _result(0.75)).verdict == "penalty"   # gap 0.10
    r = analyze_pair(_result(0.90), _result(0.72))
    assert r.verdict == "reject" and r.penalty_factor == 0.0                 # gap 0.18


def test_analyze_pair_missing_trinity_is_na():
    g = analyze_pair({"benchmark": "mmlu", "results": {}}, _result(0.8))
    assert g.verdict == "n/a" and g.gap is None


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_flags_reject_and_penalty():
    reject = analyze_pair(_result(0.9, bench="math500"), _result(0.72, bench="math500"))
    ok = analyze_pair(_result(0.8, bench="mmlu"), _result(0.79, bench="mmlu"))
    md = render([reject, ok])
    assert "generalization" in md.lower() and "math500" in md and "mmlu" in md
    assert "would be REJECTED by GATE 5" in md
    assert render([]).strip().endswith("(no paired eval/audit runs found)_")


def test_render_all_ok():
    md = render([analyze_pair(_result(0.8), _result(0.79))])
    assert "within the overfit tolerance" in md
