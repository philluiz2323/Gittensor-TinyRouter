"""Offline tests for gate 8 + the report-only advisories (issue #208).

All three checks are ADVISORIES: they must NEVER reject. ``4bb03a7`` deliberately relaxed
the gate chain "to attract miners, not repel them" (8 -> 5 gates, dropping the receipt's
fitness-shape checks), and the review on the closed #210 ruled that the ledger is not
run-scoped and that a collapsed router is still a valid submission under the score-based
contract. These tests pin that contract: ``OFFLINE_GATES`` gains nothing.

No torch / GPU / network.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity.submission.constants import N_HEAD_MODELS
from trinity.submission.gates import (
    OFFLINE_ADVISORIES,
    OFFLINE_GATES,
    audit_head_routing_diversity,
    audit_ledger_call_volume,
    validate_fitness_history_sequence,
)


def _rows(n: int = 5) -> list[dict]:
    """A well-formed fitness history: consecutive gens, mean <= max, best non-decreasing."""
    return [{"generation": g, "mean_fitness": 0.30 + 0.01 * g,
             "max_fitness": 0.40 + 0.01 * g, "best_fitness": 0.40 + 0.01 * g}
            for g in range(1, n + 1)]


def _receipt(rows: list[dict], **extra) -> dict:
    return {"fitness_history": rows, "generations": len(rows), "popsize": 33, **extra}


# --------------------------------------------------------------------------- #
# Gate 8 — wiring
# --------------------------------------------------------------------------- #
def test_all_three_checks_are_advisories_not_hard_gates():
    # 4bb03a7 relaxed the chain "to attract miners, not repel them", so #208's checks
    # inform rather than block: NONE of them may appear in OFFLINE_GATES.
    gate_names = {g.name for g in OFFLINE_GATES}
    advisory_names = {a.name for a in OFFLINE_ADVISORIES}
    assert advisory_names == {"fitness_history_sequence", "ledger_call_volume",
                              "head_routing_diversity"}
    assert not (advisory_names & gate_names)      # this PR adds ZERO new rejections


# --------------------------------------------------------------------------- #
# Gate 8 — the invariants
# --------------------------------------------------------------------------- #
def test_clean_history_passes():
    assert validate_fitness_history_sequence(_receipt(_rows())) is None


def test_duplicate_generation_is_rejected():
    rows = _rows()
    rows[3]["generation"] = 2                      # gen 2 logged twice
    assert "duplicate_generation" in validate_fitness_history_sequence(_receipt(rows))


def test_out_of_order_generations_are_rejected():
    rows = _rows()
    rows[1], rows[2] = rows[2], rows[1]
    assert "out_of_order" in validate_fitness_history_sequence(_receipt(rows))


def test_gappy_generations_are_rejected():
    rows = _rows()
    rows[-1]["generation"] = 99                    # 5 rows spanning 1..99
    assert "not_consecutive" in validate_fitness_history_sequence(_receipt(rows))


def test_row_mean_above_its_own_max_is_rejected():
    rows = _rows()
    rows[2]["mean_fitness"] = 0.99                 # a population mean cannot beat its max
    assert "mean_exceeds_max" in validate_fitness_history_sequence(_receipt(rows))


def test_running_best_that_decreases_is_rejected():
    rows = _rows()
    rows[3]["best_fitness"] = 0.01                 # best-so-far can only improve
    assert "best_decreased" in validate_fitness_history_sequence(_receipt(rows))


def test_generation_start_index_is_free():
    # a run may log gens 0..N or 1..N; only gaps/dupes/order matter.
    rows = [{"generation": g, "mean_fitness": 0.3, "max_fitness": 0.4, "best_fitness": 0.4}
            for g in range(0, 4)]
    assert validate_fitness_history_sequence(_receipt(rows)) is None


# --------------------------------------------------------------------------- #
# Gate 8 — never rejects an honest-but-legacy pack
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("history", [
    [],                                            # gate 4 owns empty/short histories
    [0.1, 0.2, 0.3],                               # legacy bare-number rows
    [{"mean_fitness": 0.3, "max_fitness": 0.4}],   # rows predating `generation`
])
def test_legacy_or_absent_history_is_left_to_other_gates(history):
    assert validate_fitness_history_sequence({"fitness_history": history}) is None


def test_missing_history_key_is_not_gate8s_problem():
    assert validate_fitness_history_sequence({}) is None


# --------------------------------------------------------------------------- #
# Advisory: head_routing_diversity  (must never reject — score-based contract)
# --------------------------------------------------------------------------- #
def _head(rows: np.ndarray) -> np.ndarray:
    return np.vstack([rows, np.random.default_rng(1).normal(size=(3, rows.shape[1]))])


def test_advisory_flags_a_collapsed_head():
    base = np.random.default_rng(0).normal(size=(1, 64))
    warn = audit_head_routing_diversity(_head(np.tile(base, (N_HEAD_MODELS, 1))))
    assert warn and "collapsed" in warn and "informational only" in warn


def test_advisory_flags_a_near_collapsed_head():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 64))
    agent = np.tile(base, (N_HEAD_MODELS, 1)) + rng.normal(scale=1e-7, size=(N_HEAD_MODELS, 64))
    assert audit_head_routing_diversity(_head(agent)) is not None


def test_advisory_silent_on_a_diverse_head():
    assert audit_head_routing_diversity(np.random.default_rng(2).normal(size=(6, 64))) is None


def test_advisory_silent_on_parallel_rows_of_different_magnitude():
    # parallel but unequal rows give DIFFERENT logits, so the head still routes: not a
    # collapse. (A pairwise-cosine test would wrongly flag this.)
    base = np.random.default_rng(3).normal(size=(1, 64))
    agent = np.vstack([base, base * 2.0, base * 3.0])
    assert audit_head_routing_diversity(_head(agent)) is None


def test_advisory_ignores_an_all_zero_head():
    assert audit_head_routing_diversity(np.zeros((6, 64))) is None   # gate 2 rejects that


# --------------------------------------------------------------------------- #
# Advisory: ledger_call_volume  (skipped without a ledger, like gate 5)
# --------------------------------------------------------------------------- #
def test_ledger_advisory_skipped_without_a_ledger_path():
    assert audit_ledger_call_volume(_receipt(_rows()), None) is None


def test_ledger_advisory_reports_thin_traffic(tmp_path):
    from trinity.llm.cost_ledger import format_ledger_line, tip_hash_from_text
    # a 2-row ledger cannot back a claimed 60 x 33 = 1980-evaluation run.
    text, tip = "", ""
    for model in ("qwen3.5-35b-a3b", "deepseek-v4-flash"):
        line = format_ledger_line(model, 100, 50, prev_hash=tip)
        text += line + "\n"
        tip = tip_hash_from_text(text)
    p = tmp_path / "ledger.jsonl"
    p.write_text(text)
    warn = audit_ledger_call_volume(_receipt(_rows(), generations=60, popsize=33), str(p))
    assert warn and "ledger_call_volume_low" in warn
    assert "informational only" in warn          # must read as a signal, never a rejection
