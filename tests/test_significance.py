"""Offline, numpy+stdlib tests for the paired significance layer (no scipy/torch).

Covers the stats core (McNemar exact binomial, paired bootstrap CI, CI-gated verdict),
the invariant assessment over a per_item block, the eval `_binary_scores` helper, and
the report renderer. All synthetic — no API calls, no GPU.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity.analysis import assess_invariants  # re-export check
from trinity.analysis.significance import (
    mcnemar,
    paired_bootstrap_ci,
    paired_diff_test,
)

_REPO = Path(__file__).resolve().parents[1]


def test_no_torch_imported():
    # Global check, like tests/test_complementarity.py: torch must never be pulled in.
    assert "torch" not in sys.modules


def test_module_imports_without_scipy_or_torch():
    # A global sys.modules scipy check is unreliable (other tests import scipy into the
    # shared process), so verify in a CLEAN subprocess that importing the module alone
    # pulls in neither scipy nor torch.
    code = (
        "import sys; sys.path.insert(0, 'src'); import trinity.analysis.significance; "
        "assert 'scipy' not in sys.modules and 'torch' not in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #
def test_mcnemar_no_discordant_pairs_is_p1():
    r = mcnemar([1, 1, 0, 0], [1, 1, 0, 0])
    assert r["n_discordant"] == 0 and r["p_value"] == 1.0


def test_mcnemar_exact_binomial_small_n():
    # 10 discordant pairs, all favouring A -> two-sided p = 2 * C(10,0)/2^10.
    r = mcnemar([1] * 10, [0] * 10)
    assert r["a_right_b_wrong"] == 10 and r["b_right_a_wrong"] == 0
    assert r["p_value"] == pytest.approx(2.0 / 1024)


def test_mcnemar_even_split_is_not_significant():
    r = mcnemar([1, 1, 0, 0], [0, 0, 1, 1])  # 2 vs 2 discordant
    assert r["p_value"] == pytest.approx(1.0)


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar([1, 0, 1], [1, 0])


# --------------------------------------------------------------------------- #
# Paired bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_identical_is_zero():
    ci = paired_bootstrap_ci([1, 0, 1, 0], [1, 0, 1, 0])
    assert ci["point"] == 0.0 and ci["ci_lo"] == 0.0 and ci["ci_hi"] == 0.0


def test_bootstrap_clear_gap_excludes_zero():
    a = np.array([1] * 40 + [0] * 10)
    b = np.array([0] * 40 + [1] * 10)  # a much better, paired
    ci = paired_bootstrap_ci(a, b, seed=0)
    assert ci["point"] == pytest.approx(0.6)
    assert ci["ci_lo"] > 0.0


def test_bootstrap_is_seed_reproducible():
    a, b = [1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0]
    assert paired_bootstrap_ci(a, b, seed=3) == paired_bootstrap_ci(a, b, seed=3)


def test_bootstrap_empty():
    assert paired_bootstrap_ci([], []) == {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}


# --------------------------------------------------------------------------- #
# CI-gated verdict
# --------------------------------------------------------------------------- #
def _clear_win():
    a = [1] * 30 + [0] * 5 + [1] * 15   # mean 0.9
    b = [0] * 30 + [1] * 5 + [1] * 15   # mean 0.4
    return a, b


def test_paired_diff_significant_win():
    a, b = _clear_win()
    c = paired_diff_test(a, b, name_a="TRINITY", name_b="best single (x)")
    assert c.significant is True and c.diff == pytest.approx(0.5)
    assert c.ci_lo > 0.0 and c.verdict.startswith("SIGNIFICANT: TRINITY")


def test_paired_diff_inside_the_noise():
    # 48 concordant, 1 a-win, 1 b-win -> point 0, CI includes 0.
    a = [1] * 24 + [0] * 24 + [1, 0]
    b = [1] * 24 + [0] * 24 + [0, 1]
    c = paired_diff_test(a, b, name_a="TRINITY", name_b="best single (x)")
    assert c.significant is False and "NOT SIGNIFICANT" in c.verdict


def test_paired_diff_significant_miss():
    b, a = _clear_win()  # now A is the weaker one
    c = paired_diff_test(a, b, name_a="TRINITY", name_b="best single (x)")
    assert c.significant is True and c.ci_hi < 0.0
    assert "SIGNIFICANT MISS" in c.verdict


# --------------------------------------------------------------------------- #
# assess_invariants over a per_item block
# --------------------------------------------------------------------------- #
def test_assess_invariants_picks_best_single_and_random():
    a, weak = _clear_win()
    per_item = {
        "task_ids": [f"q{i}" for i in range(50)],
        "TRINITY": a,
        "single::weak": weak,
        "single::strong": [1] * 25 + [0] * 25,  # mean 0.5, the best single
        "random_routing": [0] * 50,
    }
    sig = assess_invariants(per_item, benchmark="math500", n_boot=500)
    assert sig.n_questions == 50
    names = [(c.name_a, c.name_b) for c in sig.comparisons]
    assert ("TRINITY", "best single (strong)") in names   # highest-mean single chosen
    assert ("TRINITY", "random routing") in names
    d = sig.to_dict()
    assert d["benchmark"] == "math500" and len(d["comparisons"]) == 2


def test_assess_invariants_skips_missing_systems():
    sig = assess_invariants({"task_ids": ["q0"], "TRINITY": [1]})  # no singles, no random
    assert sig.comparisons == []


# --------------------------------------------------------------------------- #
# eval helper + report renderer (loaded like a script)
# --------------------------------------------------------------------------- #
def test_eval_binary_scores_helper():
    from trinity.eval import _binary_scores
    scores = [1.0, 0.0, 0.3, 0.7, ValueError("boom")]
    assert _binary_scores(scores) == [1, 0, 0, 1, 0]  # >=0.5 correct; exception -> 0


def _load_report_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "significance_report.py"
    spec = importlib.util.spec_from_file_location("significance_report", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["significance_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_report_render_and_missing_per_item():
    sr = _load_report_script()
    a, weak = _clear_win()
    good = {"file": "e1.json", "benchmark": "math500",
            "significance": __import__("trinity.analysis.significance", fromlist=["assess_invariants"])
            .assess_invariants({"task_ids": [f"q{i}" for i in range(50)], "TRINITY": a,
                                "single::x": weak, "random_routing": [0] * 50}, n_boot=300).to_dict()}
    missing = {"file": "old.json", "benchmark": "mmlu", "missing": True}
    md = sr.render([good, missing])
    assert "paired significance" in md
    assert "SIGNIFICANT: TRINITY" in md
    assert "re-run" in md.lower()  # the missing-per_item note
