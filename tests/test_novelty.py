"""Offline tests for novelty / routing-diversity analysis. No network, no GPU."""
from __future__ import annotations

import math

import pytest

from trinity.novelty import (
    NEUTRAL_NOVELTY,
    agreement_rate,
    normalize_decision,
    novelty_report,
    novelty_score,
    selection_diversity,
)


class _Role:
    """Stand-in for a Role enum, to exercise normalize_decision."""

    def __init__(self, name: str) -> None:
        self.name = name


# ---------------------------------------------------------------------------
# normalize_decision
# ---------------------------------------------------------------------------
def test_enum_like_decisions_compare_by_name():
    a = (0, _Role("WORKER"))
    b = (0, _Role("WORKER"))
    assert normalize_decision(a) == normalize_decision(b) == (0, "WORKER")


def test_scalar_and_pair_decisions_both_normalize():
    assert normalize_decision(2) == 2
    assert normalize_decision((1, "THINKER")) == (1, "THINKER")


# ---------------------------------------------------------------------------
# agreement / novelty
# ---------------------------------------------------------------------------
def test_identical_decisions_have_zero_novelty():
    a = [(0, "WORKER"), (1, "WORKER"), (2, "THINKER")]
    assert agreement_rate(a, a) == 1.0
    assert novelty_score(a, a) == 0.0


def test_completely_different_decisions_have_full_novelty():
    a = [(0, "WORKER"), (1, "WORKER")]
    b = [(1, "WORKER"), (2, "WORKER")]
    assert agreement_rate(a, b) == 0.0
    assert novelty_score(a, b) == 1.0


def test_half_agreement_is_half_novelty():
    a = [(0, "W"), (1, "W"), (2, "W"), (0, "T")]
    b = [(0, "W"), (9, "W"), (2, "W"), (9, "T")]
    assert agreement_rate(a, b) == 0.5
    assert novelty_score(a, b) == 0.5


def test_novelty_matches_the_pr_eval_definition():
    # pr_eval: novelty = 1 - agreement over aligned (agent, role) decisions.
    head = [(0, "W"), (1, "W"), (2, "W")]
    king = [(0, "W"), (0, "W"), (0, "W")]
    agree = 1 / 3
    assert agreement_rate(head, king) == pytest.approx(agree)
    assert novelty_score(head, king) == pytest.approx(1 - agree)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="aligned"):
        agreement_rate([(0, "W")], [(0, "W"), (1, "W")])


def test_empty_inputs_are_full_agreement_zero_novelty():
    assert agreement_rate([], []) == 1.0
    assert novelty_score([], []) == 0.0


# ---------------------------------------------------------------------------
# novelty_report
# ---------------------------------------------------------------------------
def test_report_lists_differing_questions_and_switches():
    head = [(0, "W"), (1, "W"), (2, "W")]
    king = [(0, "W"), (0, "W"), (0, "W")]
    r = novelty_report(head, king)
    assert r.n_questions == 3 and r.n_agree == 1
    assert r.novelty == pytest.approx(2 / 3)
    assert r.differing_indices == [1, 2]
    assert r.switched_from_to == {"(0, 'W') -> (1, 'W')": 1, "(0, 'W') -> (2, 'W')": 1}


def test_no_reference_gives_neutral_novelty():
    r = novelty_report([(0, "W"), (1, "W")], None)
    assert r.novelty == NEUTRAL_NOVELTY
    assert r.differing_indices == [] and r.switched_from_to == {}
    assert r.n_questions == 2


def test_report_roundtrips_to_dict():
    r = novelty_report([(0, "W")], [(1, "W")])
    d = r.to_dict()
    assert d["novelty"] == 1.0 and d["differing_indices"] == [0]


# ---------------------------------------------------------------------------
# selection_diversity
# ---------------------------------------------------------------------------
def test_uniform_choices_have_max_entropy():
    d = selection_diversity([0, 1, 2, 0, 1, 2])
    assert d.n_distinct == 3
    assert d.normalized_entropy == pytest.approx(1.0)
    assert d.top_share == pytest.approx(1 / 3)


def test_a_constant_head_has_zero_entropy():
    d = selection_diversity([("m", "W")] * 5)
    assert d.n_distinct == 1
    assert d.normalized_entropy == 0.0
    assert d.top_share == 1.0
    assert d.top_choice == "('m', 'W')"


def test_skewed_distribution_entropy_is_between_zero_and_one():
    d = selection_diversity([0, 0, 0, 1])
    assert 0.0 < d.normalized_entropy < 1.0
    assert d.top_choice == "0" and d.top_share == 0.75
    # Sanity vs the closed form for a 3:1 split.
    p = [0.75, 0.25]
    expected = -sum(x * math.log(x) for x in p) / math.log(2)
    assert d.normalized_entropy == pytest.approx(expected)


def test_diversity_of_nothing_is_all_zero():
    d = selection_diversity([])
    assert d.n_questions == 0 and d.n_distinct == 0
    assert d.top_choice is None and d.normalized_entropy == 0.0
    assert d.to_dict()["counts"] == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
