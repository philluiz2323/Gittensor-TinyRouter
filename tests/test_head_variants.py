"""Offline tests for the R10 (linear head >= all other head variants) verifier.

No network, no GPU.
"""
from __future__ import annotations

import pytest

from trinity.analysis.head_variants import analyze_heads, render

# docs/SPEC.md Table 3 anchors (linear per-benchmark; sparse edges wins MMLU only).
_SPEC = {
    "linear":         {"lcb": 0.615, "math500": 0.880, "mmlu": 0.916, "rlpr": 0.401},
    "sparse":         {"lcb": 0.600, "math500": 0.870, "mmlu": 0.917, "rlpr": 0.395},
    "block_diag_10":  {"lcb": 0.590, "math500": 0.865, "mmlu": 0.905, "rlpr": 0.390},
}
_PARAMS = {"linear": 40960, "sparse": 8192, "block_diag_10": 1024}


def test_spec_table3_linear_is_best_overall():
    s = analyze_heads(_SPEC, params=_PARAMS)
    assert s.linear_is_best                       # R10 holds
    assert s.best_variant == "linear"
    # linear overall is the equal-weight mean of its four benchmarks.
    assert s.linear_overall == pytest.approx((0.615 + 0.880 + 0.916 + 0.401) / 4)
    assert s.margin > 0                           # beats the strongest challenger overall


def test_per_benchmark_exception_is_reported_without_breaking_r10():
    # sparse edges out linear on MMLU (0.917 > 0.916) but loses overall -> R10 still holds.
    s = analyze_heads(_SPEC)
    assert "mmlu" in s.linear_exceptions
    assert s.per_benchmark_winner["mmlu"] == "sparse"
    assert s.per_benchmark_winner["lcb"] == "linear"
    assert s.linear_is_best


def test_a_variant_that_wins_overall_violates_r10():
    scores = {
        "linear": {"a": 0.80, "b": 0.80},
        "fancy":  {"a": 0.90, "b": 0.85},        # beats linear on both -> wins overall
    }
    s = analyze_heads(scores)
    assert not s.linear_is_best
    assert s.best_variant == "fancy"
    assert s.margin < 0
    assert s.linear_exceptions == ["a", "b"]


def test_a_tie_overall_still_counts_as_linear_ge_all():
    # R10 is ">=", so an exact overall tie keeps it holding.
    scores = {"linear": {"a": 0.8, "b": 0.6}, "other": {"a": 0.6, "b": 0.8}}
    s = analyze_heads(scores)
    assert s.linear_overall == pytest.approx(0.7)
    assert s.linear_is_best                       # 0.7 >= 0.7


def test_overall_uses_only_the_shared_benchmark_set():
    # 'other' lacks 'rlpr'; the overall must be averaged over the shared {a} only, so the
    # non-shared columns can't skew the comparison.
    scores = {
        "linear": {"a": 0.9, "rlpr": 0.1},
        "other":  {"a": 0.8},
    }
    s = analyze_heads(scores)
    assert s.benchmarks == ["a"]
    assert s.linear_overall == pytest.approx(0.9)
    assert s.linear_is_best


def test_missing_linear_key_does_not_hold():
    s = analyze_heads({"sparse": {"a": 0.9}, "block": {"a": 0.8}})
    assert not s.linear_is_best and s.best_variant is None


def test_single_variant_has_nothing_to_compare():
    s = analyze_heads({"linear": {"a": 0.9, "b": 0.8}})
    assert not s.linear_is_best                    # no other variant
    assert s.linear_overall == pytest.approx(0.85)


def test_custom_linear_key():
    scores = {"lin_head": {"a": 0.9}, "sparse": {"a": 0.8}}
    s = analyze_heads(scores, linear_key="lin_head")
    assert s.linear_is_best and s.linear_key == "lin_head"


def test_non_numeric_scores_are_cleaned():
    scores = {"linear": {"a": 0.9, "b": "oops"}, "sparse": {"a": 0.8, "b": 0.7}}
    s = analyze_heads(scores)
    # 'b' is not shared (linear dropped it) -> overall over {a} only
    assert s.benchmarks == ["a"] and s.linear_is_best


def test_to_dict_and_render():
    d = analyze_heads(_SPEC, params=_PARAMS).to_dict()
    assert d["linear_is_best"] is True and d["best_variant"] == "linear"
    assert d["variants"][0]["name"] == "linear"
    out = render(_SPEC, params=_PARAMS)
    assert "R10" in out and "HOLDS" in out
    out2 = render({"linear": {"a": 0.5}, "fancy": {"a": 0.9}})
    assert "VIOLATED" in out2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
