"""Offline tests for the results-table oracle columns (ORACLE_CEILING_DIAGNOSTIC §7 item 3).

Two things must hold for this to be safe to land: the parse must prefer the **bootstrap CI**
block (§4: "the verdict is read off the CIs, never the point estimates") while still
rendering an older point-only report, and the wiring must be strictly **additive** — with no
oracle reports the results table has to stay byte-identical to what it always emitted.
Synthetic reports, stdlib only — no torch/network.
"""
import importlib.util
import json
import sys
from pathlib import Path

from trinity.analysis import oracle_column as oracle_column_pkg  # re-export check
from trinity.analysis.oracle_column import (
    Estimate,
    from_oracle_report,
    index_by_benchmark,
)

_REPO = Path(__file__).resolve().parents[1]

# Load results_table.py as a module (it lives under scripts/, not the importable package).
_SCRIPT = _REPO / "scripts" / "results_table.py"
_spec = importlib.util.spec_from_file_location("results_table_oracle", _SCRIPT)
rt = importlib.util.module_from_spec(_spec)
sys.modules["results_table_oracle"] = rt
_spec.loader.exec_module(rt)


def _report(**over):
    d = {
        "benchmark": "math500",
        "point_estimates": {"routing_oracle": 0.80, "routing_headroom": 0.04},
        "bootstrap_ci_95": {
            "routing_oracle": {"point": 0.855, "ci_lo": 0.801, "ci_hi": 0.903},
            "routing_headroom": {"point": 0.050, "ci_lo": 0.005, "ci_hi": 0.085},
        },
        "verdict": {"label": "NEAR_CEILING", "router_gap_closed": 2.2995},
    }
    d.update(over)
    return d


def _rows(*benches):
    return [{"file": f"f{i}", "benchmark": b, "coordinator": "c", "variant": "eval",
             "trinity": 0.9, "random": 0.5, "best_single": 0.8, "best_model": "glm",
             "singles": {"glm": 0.8, "ds": 0.7}} for i, b in enumerate(benches)]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_prefers_bootstrap_ci_over_point_estimates():
    c = from_oracle_report(_report())
    assert c is not None
    assert c.oracle.point == 0.855 and c.oracle.has_ci      # CI block wins over 0.80
    assert c.oracle.render() == "0.855 [0.801, 0.903]"
    assert c.headroom.point == 0.050
    assert c.verdict_label == "NEAR_CEILING"


def test_falls_back_to_point_estimates_without_a_ci_block():
    c = from_oracle_report(_report(bootstrap_ci_95={}))
    assert c.oracle.point == 0.80 and c.oracle.has_ci is False
    assert c.oracle.render() == "0.800"                     # renders, no interval


def test_gap_closed_falls_back_to_the_trinity_block():
    c = from_oracle_report(_report(verdict={"label": "X"}, trinity={"router_gap_closed": 0.5}))
    assert c.gap_closed == 0.5 and c.render_gap_closed() == "50%"


def test_gap_closed_above_100_percent_is_reported_not_clamped():
    # TRINITY can beat the cross-fit oracle (it debiases the winner's curse); hiding that
    # would conceal the disagreement worth looking at.
    assert from_oracle_report(_report()).render_gap_closed() == "230%"


def test_missing_gap_closed_renders_as_dash():
    c = from_oracle_report(_report(verdict={"label": "INCONCLUSIVE"}))
    assert c.gap_closed is None and c.render_gap_closed() == "—"


def test_report_without_a_benchmark_is_rejected():
    # A stray JSON must never be joined onto some other benchmark's row.
    assert from_oracle_report({"point_estimates": {"routing_oracle": 0.9}}) is None
    assert from_oracle_report({"benchmark": ""}) is None
    assert from_oracle_report({}) is None


def test_non_numeric_values_are_ignored():
    c = from_oracle_report(_report(bootstrap_ci_95={}, point_estimates={
        "routing_oracle": "nope", "routing_headroom": True}))
    assert c is None or (c.oracle is None and c.headroom is None)


def test_index_skips_empty_and_keeps_first_per_benchmark():
    empty = {"benchmark": "gpqa"}                            # no columns at all
    first, second = _report(), _report(verdict={"label": "OTHER", "router_gap_closed": 9.0})
    idx = index_by_benchmark([first, second, empty])
    assert set(idx) == {"math500"}                           # empty dropped
    assert idx["math500"].verdict_label == "NEAR_CEILING"    # first wins


# --------------------------------------------------------------------------- #
# results_table wiring — additive, no regression
# --------------------------------------------------------------------------- #
def test_table_is_unchanged_without_oracle_data():
    rows = _rows("math500", "mmlu")
    assert rt.render(rows) == rt.render(rows, {})            # no columns, no legend
    assert "oracle (95% CI)" not in rt.render(rows)
    assert "gap closed" not in rt.render(rows)


def test_columns_and_legend_appear_with_oracle_data():
    rows = _rows("math500")
    md = rt.render(rows, index_by_benchmark([_report()]))
    assert "oracle (95% CI)" in md and "headroom (95% CI)" in md and "gap closed" in md
    assert "0.855 [0.801, 0.903]" in md and "230%" in md
    assert "NEAR_CEILING" in md


def test_benchmark_without_a_report_gets_dashes_not_another_benchmarks_numbers():
    rows = _rows("math500", "gpqa")
    md = rt.render(rows, index_by_benchmark([_report()]))
    gpqa_line = next(ln for ln in md.splitlines() if ln.startswith("| gpqa "))
    assert "0.855" not in gpqa_line and gpqa_line.rstrip().endswith("| — | — | — |")


def test_load_oracle_columns_reads_reports_from_disk(tmp_path):
    (tmp_path / "oracle_report_math500.json").write_text(json.dumps(_report()))
    (tmp_path / "oracle_report_broken.json").write_text("{not json")
    idx = rt.load_oracle_columns(str(tmp_path))
    assert set(idx) == {"math500"}                           # unreadable file skipped
    assert idx["math500"].oracle.render() == "0.855 [0.801, 0.903]"


def test_load_oracle_columns_empty_tree_is_empty(tmp_path):
    assert rt.load_oracle_columns(str(tmp_path)) == {}


def test_estimate_render_without_ci():
    assert Estimate(0.5).render() == "0.500"
    assert Estimate(0.5, 0.4, 0.6).render(digits=2) == "0.50 [0.40, 0.60]"


def test_module_is_reachable_through_the_analysis_package():
    assert hasattr(oracle_column_pkg, "from_oracle_report")
    assert hasattr(oracle_column_pkg, "index_by_benchmark")
