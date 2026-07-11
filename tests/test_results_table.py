"""Offline tests for the multi-task summary in scripts/results_table.py.

The summary reduces each benchmark's eval rows to one number per system, then averages
across benchmarks. TRINITY, the fixed single models, and random routing must all use the
SAME reduction — reducing TRINITY with max() while averaging the baselines compares a
cherry-picked best against a mean and biases the R1/R2 and R4 verdicts toward TRINITY.

No API calls, no GPU, no filesystem: `render` takes rows directly.
"""
import importlib.util
import sys
from pathlib import Path

# Load the script as a module (it lives under scripts/, not the importable package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "results_table.py"
_spec = importlib.util.spec_from_file_location("results_table", _SCRIPT)
rt = importlib.util.module_from_spec(_spec)
sys.modules["results_table"] = rt
_spec.loader.exec_module(rt)


def _row(bench: str, coord: str, trinity: float, single: float, random_: float = 0.10) -> dict:
    """One eval*.json row, in the shape `load_rows` produces."""
    return {
        "file": f"{bench}/{coord}/eval.json",
        "benchmark": bench,
        "coordinator": coord,
        "variant": "eval",
        "trinity": trinity,
        "random": random_,
        "best_single": single,
        "best_model": "deepseek",
        "singles": {"deepseek": single},
    }


def _summary(rows: list[dict]) -> str:
    return rt.render(rows).split("## Multi-task summary")[1]


# math500 carries two eval rows (two coordinators); mmlu one. TRINITY's best math500 row
# is 0.60, but deepseek's best is 0.72 — averaging deepseek's rows hides that.
_ROWS = [
    _row("math500", "coordA", trinity=0.60, single=0.72),
    _row("math500", "coordB", trinity=0.50, single=0.44),
    _row("mmlu", "coordA", trinity=0.70, single=0.60),
]


def test_baselines_reduce_per_benchmark_the_same_way_trinity_does():
    """deepseek must reduce to max(0.72, 0.44)=0.72 on math500, not mean=0.58."""
    summary = _summary(_ROWS)
    # (0.72 + 0.60) / 2 = 0.660, not (0.58 + 0.60) / 2 = 0.590
    assert "| single: deepseek (fixed) | 0.660 |" in summary
    assert "0.590" not in summary


def test_r1_r2_verdict_is_not_biased_toward_trinity():
    """TRINITY 0.650 < best fixed single 0.660 -> the claim must NOT read as holding."""
    summary = _summary(_ROWS)
    assert "**TRINITY (per-task best coordinator)** | **0.650**" in summary
    assert "**R1/R2** (TRINITY avg > best fixed single avg): ❌ (0.650 vs 0.660)" in summary


def test_random_routing_also_reduces_per_benchmark_best():
    """R4 compares like with like: random reduces with max, as TRINITY does."""
    rows = [
        _row("math500", "coordA", trinity=0.60, single=0.10, random_=0.90),
        _row("math500", "coordB", trinity=0.50, single=0.10, random_=0.10),
        _row("mmlu", "coordA", trinity=0.70, single=0.10, random_=0.50),
    ]
    summary = _summary(rows)
    # random: max(0.90, 0.10)=0.90 on math500, 0.50 on mmlu -> 0.700, not (0.50+0.50)/2=0.500
    assert "| random routing | 0.700 |" in summary
    assert "**R4** (TRINITY avg > random avg): ❌ (0.650 vs 0.700)" in summary


def test_summary_tolerates_a_null_system_score():
    """A partial/older eval*.json may carry a null TRINITY or lack random_routing.

    `load_rows` only guarantees the keys exist, not that the values are non-null, so
    `max(...)`/`sum(...)` over a `None` used to crash the multi-task summary. The
    per-bench reduction now skips `None` (treating an all-null benchmark as 0.0).
    """
    rows = [
        _row("math500", "coordA", trinity=0.60, single=0.50, random_=0.20),
        _row("mmlu", "coordA", trinity=0.70, single=0.55, random_=0.30),
    ]
    rows[1]["trinity"] = None      # mmlu TRINITY missing
    rows[1]["random"] = None       # mmlu random_routing missing

    summary = _summary(rows)       # must not raise
    # TRINITY: (0.60 + 0.0) / 2 = 0.300 ; random: (0.20 + 0.0) / 2 = 0.100
    assert "**TRINITY (per-task best coordinator)** | **0.300**" in summary
    assert "| random routing | 0.100 |" in summary


def test_single_eval_row_per_benchmark_is_unchanged():
    """With one row per benchmark max == mean, so the summary is untouched by this fix."""
    rows = [
        _row("math500", "coordA", trinity=0.60, single=0.50, random_=0.20),
        _row("mmlu", "coordA", trinity=0.70, single=0.55, random_=0.30),
    ]
    summary = _summary(rows)
    assert "**TRINITY (per-task best coordinator)** | **0.650**" in summary
    assert "| single: deepseek (fixed) | 0.525 |" in summary
    assert "| random routing | 0.250 |" in summary
    assert "**R1/R2** (TRINITY avg > best fixed single avg): ✅ HOLDS (0.650 vs 0.525)" in summary


def test_trinity_still_wins_when_it_genuinely_beats_the_best_row():
    """The fix must not invert honest wins — only remove the thumb on the scale."""
    rows = [
        _row("math500", "coordA", trinity=0.80, single=0.72),
        _row("math500", "coordB", trinity=0.50, single=0.44),
        _row("mmlu", "coordA", trinity=0.70, single=0.60),
    ]
    summary = _summary(rows)
    # TRINITY (0.80 + 0.70)/2 = 0.750 vs deepseek (0.72 + 0.60)/2 = 0.660
    assert "**R1/R2** (TRINITY avg > best fixed single avg): ✅ HOLDS (0.750 vs 0.660)" in summary


# --------------------------------------------------------------------------- #
# Partial coverage: a single model present on only SOME benchmarks is averaged
# over its own (favorable) subset and must NOT be the "best fixed single".
# --------------------------------------------------------------------------- #
def _rowx(bench: str, coord: str, trinity: float, singles: dict, random_: float = 0.10) -> dict:
    """An eval*.json row with an explicit multi-model ``singles`` map."""
    best_model = max(singles, key=singles.get)
    return {
        "file": f"{bench}/{coord}/eval.json",
        "benchmark": bench,
        "coordinator": coord,
        "variant": "eval",
        "trinity": trinity,
        "random": random_,
        "best_single": singles[best_model],
        "best_model": best_model,
        "singles": dict(singles),
    }


def test_partial_coverage_single_is_excluded_from_best_fixed():
    """A 1-of-2-bench model with a high score must not flip R1/R2.

    gpt5 scores 0.90 but only on math500. Averaged over its own single benchmark it
    looks like the best fixed single (0.900) and used to flip R1/R2 to ❌, even though
    it was never run on mmlu. Only deepseek (full coverage, 0.525) is a valid
    comparator, so the claim holds.
    """
    rows = [
        _rowx("math500", "coordA", 0.60, {"deepseek": 0.50, "gpt5": 0.90}),
        _rowx("mmlu", "coordA", 0.70, {"deepseek": 0.55}),
    ]
    summary = _summary(rows)
    assert "| single: gpt5 (partial: 1/2 benches) | 0.900 |" in summary
    assert "| single: deepseek (fixed) | 0.525 |" in summary
    assert "**R1/R2** (TRINITY avg > best fixed single avg): ✅ HOLDS (0.650 vs 0.525)" in summary
    # the buggy comparator (gpt5's subset 0.900) must not appear in the verdict
    assert "0.900)" not in summary.split("**R1/R2**")[1].splitlines()[0]


def test_r1r2_is_na_when_no_single_model_covers_all_benches():
    """With no full-coverage single, there is no valid fixed baseline — don't fake one."""
    rows = [
        _rowx("math500", "coordA", 0.60, {"gpt5": 0.90}),
        _rowx("mmlu", "coordA", 0.70, {"deepseek": 0.55}),
    ]
    summary = _summary(rows)
    assert "| single: gpt5 (partial: 1/2 benches) | 0.900 |" in summary
    assert "| single: deepseek (partial: 1/2 benches) | 0.550 |" in summary
    # scope the check to the R1/R2 line (R4 legitimately holds at 0.650 vs random)
    r1r2_line = summary.split("**R1/R2**")[1].splitlines()[0]
    assert "N/A — no single model covers all 2 benchmarks" in r1r2_line
    assert "HOLDS" not in r1r2_line and "❌" not in r1r2_line


def test_full_coverage_multi_model_compares_only_full_models():
    """Clean multi-model runs are unchanged: every model is full-coverage and compared."""
    rows = [
        _rowx("math500", "coordA", 0.60, {"deepseek": 0.50, "gpt5": 0.55}),
        _rowx("mmlu", "coordA", 0.70, {"deepseek": 0.62, "gpt5": 0.40}),
    ]
    summary = _summary(rows)
    # deepseek (0.50+0.62)/2 = 0.560 ; gpt5 (0.55+0.40)/2 = 0.475 -> best_fixed = 0.560
    assert "| single: deepseek (fixed) | 0.560 |" in summary
    assert "| single: gpt5 (fixed) | 0.475 |" in summary
    assert "**R1/R2** (TRINITY avg > best fixed single avg): ✅ HOLDS (0.650 vs 0.560)" in summary
    assert "partial" not in summary
