"""Offline tests for the L0/L1/L2 reachability layer (ORACLE §2.2, §6, §9).

Pure numpy/stdlib over synthetic ``oracle_matrix`` dicts — no network, no GPU, no torch.
"""
from __future__ import annotations

import pytest

from trinity.analysis.reachability import (
    LEVEL_DESCRIPTIONS,
    LEVEL_ORDER,
    THIN_HEADROOM,
    analyze,
    render,
)


def _matrix(per_model_bits: dict[str, list[int]], benchmark: str = "math500") -> dict:
    """Build a canonical ``oracle_matrix`` from ``{model: [0/1 per task]}``."""
    models = list(per_model_bits)
    n_tasks = len(next(iter(per_model_bits.values())))
    tasks = [
        {"id": f"q{i}", "per_model": {m: [per_model_bits[m][i]] for m in models}}
        for i in range(n_tasks)
    ]
    return {"benchmark": benchmark, "tasks": tasks}


# A pool with NO complementarity: 'a' solves everything 'b' does. oracle == best_single,
# so headroom is exactly 0 (thin).
FLAT = {"a": [1, 1, 0, 0], "b": [1, 0, 0, 0]}
# A complementary pool: each model solves a different query. oracle 0.5 vs best 0.25.
COMPLEMENTARY = {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]}
# Wider still: three solved queries between them.
WIDER = {"a": [1, 0, 1, 0], "b": [0, 1, 0, 0]}
# Strictly narrower than the others: only one query is solvable at all (oracle 0.25).
# Also thin (headroom 0), so using it as a WIDER level is both impossible and tempting.
NARROW = {"a": [1, 0, 0, 0], "b": [0, 0, 0, 0]}


# --------------------------------------------------------------------------------------
# Level bookkeeping
# --------------------------------------------------------------------------------------


def test_levels_are_ordered_narrowest_first():
    assert LEVEL_ORDER == ("L0", "L1", "L2")
    assert set(LEVEL_DESCRIPTIONS) == set(LEVEL_ORDER)


def test_levels_are_sorted_regardless_of_input_order():
    s = analyze({"L2": _matrix(WIDER), "L0": _matrix(FLAT), "L1": _matrix(COMPLEMENTARY)})
    assert s.levels_measured == ["L0", "L1", "L2"]
    assert s.widest_level == "L2"


def test_unknown_level_is_an_error():
    with pytest.raises(ValueError, match="unknown reachability level"):
        analyze({"L3": _matrix(FLAT)})
    with pytest.raises(ValueError, match="unknown reachability level"):
        analyze({"L0": _matrix(FLAT)}, cis={"L9": (0.0, 0.01)})


def test_no_data_is_reported_not_crashed():
    s = analyze({})
    assert s.verdict == "NO_DATA"
    assert s.widest_level is None
    assert s.can_rule_out_routing is False


def test_reuses_the_canonical_decoder_so_l1_needs_no_new_schema():
    """An L1 matrix is just composite 'model/role' keys in the SAME oracle_matrix schema."""
    l1 = _matrix({"a/worker": [1, 0, 0, 0], "a/thinker": [0, 1, 0, 0]})
    s = analyze({"L1": l1})
    assert s.levels[0].oracle.models == ["a/worker", "a/thinker"]
    assert s.levels[0].routing_oracle == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# The monotonicity guard
# --------------------------------------------------------------------------------------


def test_oracle_dropping_as_the_level_widens_is_flagged_as_impossible():
    """L1's option set contains L0's, so its oracle cannot be lower. A drop = bad data."""
    s = analyze({"L0": _matrix(COMPLEMENTARY), "L1": _matrix(NARROW)})
    assert s.monotonicity_violations
    assert "impossible" in s.monotonicity_violations[0]
    assert s.verdict == "INCONSISTENT"
    assert s.can_rule_out_routing is False


def test_a_non_decreasing_oracle_raises_no_violation():
    s = analyze({"L0": _matrix(COMPLEMENTARY), "L1": _matrix(WIDER)})
    assert s.monotonicity_violations == []


def test_equal_oracles_across_levels_are_not_a_violation():
    """Widening may add no new solves; that is legal, only a DROP is impossible."""
    s = analyze({"L0": _matrix(COMPLEMENTARY), "L1": _matrix(COMPLEMENTARY)})
    assert s.monotonicity_violations == []


def test_headroom_may_fall_as_the_level_widens_without_being_a_violation():
    """Headroom is a difference of two monotone quantities, so it is NOT constrained.

    Here L1 lifts the best single model more than it lifts the ceiling: the oracle rises
    (0.50 -> 0.75) while the headroom falls (0.25 -> 0.25-> lower). That must not be
    reported as a data bug.
    """
    l0 = _matrix({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})       # oracle .50, best .25
    l1 = _matrix({"a": [1, 1, 1, 0], "b": [0, 1, 0, 0]})       # oracle .75, best .75
    s = analyze({"L0": l0, "L1": l1})

    assert s.levels[1].routing_oracle > s.levels[0].routing_oracle   # oracle rose
    assert s.levels[1].headroom < s.levels[0].headroom               # headroom fell
    assert s.monotonicity_violations == []                           # still legal


def test_inconsistency_outranks_every_other_verdict():
    """A collection bug must not be reported as a routing conclusion."""
    s = analyze(
        {"L0": _matrix(COMPLEMENTARY), "L1": _matrix(NARROW)},
        cis={"L1": (0.0, 0.0)},  # would otherwise look thin and conclusive
    )
    assert s.verdict == "INCONSISTENT"


# --------------------------------------------------------------------------------------
# The §6 widest-level rule
# --------------------------------------------------------------------------------------


def test_thin_l0_alone_cannot_rule_routing_out():
    """The headline false-negative fix: L0 is a LOWER BOUND (§2.2)."""
    s = analyze({"L0": _matrix(FLAT)})
    assert s.levels[0].is_thin
    assert s.verdict == "LOWER_BOUND_ONLY"
    assert s.can_rule_out_routing is False
    assert "LOWER BOUND" in s.message
    assert "L1" in s.message  # names the next level to measure


def test_thin_l1_still_cannot_rule_routing_out_because_l2_is_unmeasured():
    s = analyze({"L0": _matrix(FLAT), "L1": _matrix(FLAT)})
    assert s.verdict == "LOWER_BOUND_ONLY"
    assert s.can_rule_out_routing is False
    assert "L2" in s.message


def test_thin_at_the_widest_defined_level_rules_routing_out():
    s = analyze(
        {"L0": _matrix(FLAT), "L1": _matrix(FLAT), "L2": _matrix(FLAT)},
        cis={"L2": (0.0, 0.01)},
    )
    assert s.verdict == "POOL_BOUND"
    assert s.can_rule_out_routing is True
    assert "the lever is the pool" in s.message


def test_real_headroom_at_the_widest_level_is_not_ruled_out():
    s = analyze({"L0": _matrix(FLAT), "L1": _matrix(COMPLEMENTARY)})
    assert s.widest_headroom == pytest.approx(0.25)
    assert s.verdict == "HEADROOM_REMAINS"
    assert s.can_rule_out_routing is False


def test_the_verdict_reads_the_widest_level_not_the_narrowest():
    """L0 thin but L1 wide → routing is NOT ruled out. This is the whole point of §2.2."""
    s = analyze({"L0": _matrix(FLAT), "L1": _matrix(COMPLEMENTARY)})
    assert s.levels[0].is_thin          # narrowest says "hopeless"
    assert not s.levels[1].is_thin      # widest disagrees
    assert s.can_rule_out_routing is False


# --------------------------------------------------------------------------------------
# Sampled levels need a CI
# --------------------------------------------------------------------------------------


def test_sampled_l2_without_a_ci_cannot_carry_a_verdict():
    s = analyze({"L0": _matrix(FLAT), "L2": _matrix(FLAT)})
    assert s.levels[-1].is_sampled is True
    assert s.levels[-1].verdict_supported is False
    assert s.verdict == "NEEDS_CI"
    assert s.can_rule_out_routing is False


def test_unsampled_levels_need_no_ci():
    for level in ("L0", "L1"):
        s = analyze({level: _matrix(FLAT)})
        assert s.levels[0].is_sampled is False
        assert s.levels[0].verdict_supported is True


def test_thinness_uses_the_ci_upper_bound_not_the_point_estimate():
    """A point estimate of 0 with a wide CI is not thin — §6 gates on the upper bound."""
    s = analyze({"L2": _matrix(FLAT)}, cis={"L2": (-0.01, 0.30)})
    assert s.levels[0].headroom == pytest.approx(0.0)  # point estimate is thin...
    assert s.levels[0].is_thin is False                # ...but the CI says otherwise
    assert s.verdict == "HEADROOM_REMAINS"


def test_ci_accepts_both_a_pair_and_a_mapping():
    a = analyze({"L2": _matrix(FLAT)}, cis={"L2": (0.0, 0.01)})
    b = analyze({"L2": _matrix(FLAT)}, cis={"L2": {"ci_lo": 0.0, "ci_hi": 0.01}})
    assert a.widest_ci == b.widest_ci == (0.0, 0.01)


def test_inverted_ci_bounds_are_an_error():
    with pytest.raises(ValueError, match="inverted"):
        analyze({"L0": _matrix(FLAT)}, cis={"L0": (0.4, 0.1)})


def test_malformed_ci_is_an_error():
    with pytest.raises(ValueError, match="must be a"):
        analyze({"L0": _matrix(FLAT)}, cis={"L0": "thin"})


def test_ci_mapping_missing_a_bound_is_a_clean_error():
    """A half-filled mapping must raise ValueError, not a TypeError from float(None)."""
    with pytest.raises(ValueError, match="both 'ci_lo' and 'ci_hi'"):
        analyze({"L0": _matrix(FLAT)}, cis={"L0": {"ci_lo": 0.0}})


def test_non_numeric_ci_bounds_are_a_clean_error():
    with pytest.raises(ValueError, match="must be numbers"):
        analyze({"L0": _matrix(FLAT)}, cis={"L0": ("low", "high")})


def test_thin_bar_matches_the_spec_threshold():
    assert THIN_HEADROOM == 0.02


# --------------------------------------------------------------------------------------
# Serialization + rendering
# --------------------------------------------------------------------------------------


def test_to_dict_is_json_serializable_and_complete():
    import json

    s = analyze({"L0": _matrix(FLAT), "L2": _matrix(COMPLEMENTARY)}, cis={"L2": (0.1, 0.4)})
    d = s.to_dict()
    json.dumps(d)  # must not raise

    assert d["levels_measured"] == ["L0", "L2"]
    assert d["widest_level"] == "L2"
    assert d["widest_ci_95"] == [0.1, 0.4]
    assert d["levels"][0]["ci_95"] is None
    assert d["levels"][1]["is_sampled"] is True
    assert "oracle" in d["levels"][0]


def test_render_lists_every_level_and_the_verdict():
    s = analyze({"L0": _matrix(FLAT), "L1": _matrix(COMPLEMENTARY)})
    text = render(s)
    assert "L0" in text and "L1" in text
    assert s.verdict in text
    assert "Widest level measured" in text


def test_render_surfaces_integrity_violations():
    s = analyze({"L0": _matrix(COMPLEMENTARY), "L1": _matrix(NARROW)})
    text = render(s)
    assert "Integrity violations" in text


def test_render_of_nothing_is_not_a_crash():
    assert "No reachability levels" in render(analyze({}))
