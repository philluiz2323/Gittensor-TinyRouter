"""Offline tests for the pool price/membership consistency check.

Synthetic ``PoolSources`` for the drift cases + a real-data check that the committed repo
is consistent today (green). stdlib+PyYAML only, no torch/network.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.llm.pool_consistency import (
    PoolConsistencyReport,
    PoolSources,
    PriceTable,
    check_pool_consistency,
    gather_sources,
    render,
)

_REPO = Path(__file__).resolve().parents[1]

_POOL = ["qwen3.5-35b-a3b", "minimax-m3", "deepseek-v4-flash"]
_PRICES = {"qwen3.5-35b-a3b": (0.14, 1.00), "minimax-m3": (0.30, 1.20),
           "deepseek-v4-flash": (0.09, 0.18)}


def _sources(*, yaml_pool=None, dpm=None, canonical=None, duplicates=None):
    """A fully-consistent PoolSources, overridable per field to inject one drift."""
    return PoolSources(
        yaml_pool=list(yaml_pool if yaml_pool is not None else _POOL),
        default_pool_models=list(dpm if dpm is not None else _POOL),
        canonical=PriceTable("openrouter_pricing.OPENROUTER_POOL_PRICES",
                             dict(canonical if canonical is not None else _PRICES)),
        duplicates=[PriceTable(n, dict(p)) for n, p in (duplicates if duplicates is not None else [
            ("oracle_ceiling._DEFAULT_PRICES", _PRICES),
            ("fugu.cost.PRICES", _PRICES),
        ])],
    )


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.llm.pool_consistency; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_consistent_sources_are_ok():
    report = check_pool_consistency(_sources())
    assert report.ok and report.problems == []
    assert report.pool == _POOL
    assert report.tables_checked == ["openrouter_pricing.OPENROUTER_POOL_PRICES",
                                     "oracle_ceiling._DEFAULT_PRICES", "fugu.cost.PRICES"]


# --------------------------------------------------------------------------- #
# membership drift
# --------------------------------------------------------------------------- #
def test_default_pool_models_missing_a_model_is_flagged():
    # gate 6 would false-reject an honest submission.
    report = check_pool_consistency(_sources(dpm=["qwen3.5-35b-a3b", "minimax-m3"]))
    assert not report.ok
    assert any("missing pool model" in p and "gate 6" in p for p in report.problems)


def test_default_pool_models_extra_model_is_flagged():
    report = check_pool_consistency(_sources(dpm=[*_POOL, "ghost-model"]))
    assert any("non-pool model" in p and "ghost-model" in p for p in report.problems)


def test_duplicate_pool_name_is_flagged():
    report = check_pool_consistency(_sources(yaml_pool=[*_POOL, "minimax-m3"]))
    assert any("duplicate pool name" in p and "minimax-m3" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# price-table drift
# --------------------------------------------------------------------------- #
def test_price_table_missing_a_model_is_flagged():
    stale = {"qwen3.5-35b-a3b": (0.14, 1.00), "minimax-m3": (0.30, 1.20)}  # no deepseek
    report = check_pool_consistency(_sources(duplicates=[("fugu.cost.PRICES", stale)]))
    assert any("missing price" in p and "deepseek-v4-flash" in p for p in report.problems)


def test_price_table_extra_entry_is_flagged():
    extra = {**_PRICES, "old-model": (1.0, 2.0)}
    report = check_pool_consistency(_sources(duplicates=[("fugu.cost.PRICES", extra)]))
    assert any("stale/extra" in p and "old-model" in p for p in report.problems)


def test_price_disagreement_is_flagged():
    drifted = {**_PRICES, "minimax-m3": (0.99, 1.20)}   # canonical says 0.30
    report = check_pool_consistency(_sources(duplicates=[("fugu.cost.PRICES", drifted)]))
    assert any("price disagreement" in p and "minimax-m3" in p for p in report.problems)


@pytest.mark.parametrize("bad", [(0.0, 1.0), (-0.1, 1.0), (0.14, float("nan")),
                                 (0.14, float("inf")), (0.14,)])
def test_invalid_price_is_flagged(bad):
    report = check_pool_consistency(
        _sources(canonical={**_PRICES, "qwen3.5-35b-a3b": bad}))
    assert not report.ok
    assert any("qwen3.5-35b-a3b" in p for p in report.problems)


def test_empty_pool_is_flagged_not_crash():
    report = check_pool_consistency(_sources(yaml_pool=[], dpm=[]))
    assert not report.ok and any("no pool membership" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# real committed repo: every source agrees today (green)
# --------------------------------------------------------------------------- #
def test_real_repo_is_consistent():
    report = check_pool_consistency(gather_sources(_REPO))
    assert report.ok, f"pool drift on committed repo: {report.problems}"
    assert set(report.pool) == set(_POOL)
    assert len(report.tables_checked) == 3


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_ok_and_drift():
    ok_md = render(check_pool_consistency(_sources()))
    assert "consistency" in ok_md.lower() and "OK" in ok_md
    drift_md = render(check_pool_consistency(_sources(dpm=["qwen3.5-35b-a3b"])))
    assert "DRIFT" in drift_md and "gate 6" in drift_md


def test_report_dataclass_to_dict():
    d = PoolConsistencyReport(problems=["x"], pool=["a"], tables_checked=["t"]).to_dict()
    assert d == {"ok": False, "n_problems": 1, "problems": ["x"],
                 "pool": ["a"], "tables_checked": ["t"]}
