"""Offline, numpy-only tests for the HERO dense reward (improvement #3, Stage A).

Covers the two pure cores added to ``trinity.optim.fitness``:
  * ``hero_quality`` — per-trajectory self-consistency vote fraction in [0, 1]
  * ``hero_bucket_bonus`` — min-max normalization WITHIN the correct/incorrect
    buckets, and the invariant that adding it keeps correct > wrong.

No torch, no network: trajectories are hand-built and the extractors are the same
pure ``reward`` helpers the eval scorer uses.
"""
import sys

import numpy as np
import pytest

from trinity.optim.fitness import (
    FitnessConfig,
    hero_bucket_bonus,
    hero_quality,
    shaped_reward,
)
from trinity.types import Role, Task, Trajectory, TurnRecord


def test_no_torch_imported():
    assert "torch" not in sys.modules, "hero-dense tests must not import torch"


def _traj(benchmark, turn_texts, final):
    task = Task(task_id="t", benchmark=benchmark, prompt="q", answer="x")
    turns = [
        TurnRecord(turn=i + 1, agent_name="m", role=Role.WORKER, raw_output=x, processed_output=x)
        for i, x in enumerate(turn_texts)
    ]
    return Trajectory(task=task, turns=turns, final_answer=final)


# --------------------------------------------------------------------------- #
# hero_quality
# --------------------------------------------------------------------------- #
def test_hero_quality_all_agree_is_one():
    assert hero_quality(_traj("mmlu", ["The answer is B.", "I think B.", "Final: B"], "answer B")) == 1.0
    assert hero_quality(_traj("math500", [r"\boxed{42}", "so 42", r"= \boxed{42}"], r"\boxed{42}")) == 1.0


def test_hero_quality_split_vote_fraction():
    # 2 of 3 answer-bearing turns say B (the committed answer).
    q = hero_quality(_traj("mmlu", ["Answer: B", "Answer: A", "Answer: B"], "Answer: B"))
    assert q == pytest.approx(2 / 3)


def test_hero_quality_ignores_non_answer_turns():
    # Only the 2 answer-bearing turns count; the musing turn is skipped.
    q = hero_quality(_traj("mmlu", ["let me think...", "Answer: C", "so C"], "Answer: C"))
    assert q == 1.0


def test_hero_quality_zero_when_no_parseable_answer():
    assert hero_quality(_traj("mmlu", ["not sure", "hard to say"], "")) == 0.0
    assert hero_quality(_traj("math500", ["no number here", "hmm"], "")) == 0.0


def test_hero_quality_code_self_consistency():
    same = "```python\ndef f():\n    return 1\n```"
    other = "```python\ndef f():\n    return 2\n```"
    assert hero_quality(_traj("livecodebench", [same, same], same)) == 1.0
    assert hero_quality(_traj("livecodebench", [same, other], same)) == pytest.approx(0.5)


def test_hero_quality_in_unit_interval():
    for tt in (["Answer: A", "Answer: B", "Answer: C"], ["Answer: D"], [r"\boxed{1}", r"\boxed{2}"]):
        q = hero_quality(_traj("mmlu", tt, tt[0]))
        assert 0.0 <= q <= 1.0


def test_hero_quality_excludes_verifier_turn():
    # The bug that closed the first cut of this reward: a terminal Verifier turn
    # whose critique NAMES a different letter (B) than the solver committed (C)
    # must never enter the self-consistency vote. With the Verifier excluded, the
    # only solver turn agrees with the committed answer, so quality is 1.0.
    # (Counting the Verifier would have diluted the vote to 1/2 = 0.5.)
    task = Task(task_id="t", benchmark="mmlu", prompt="q", answer="C")
    turns = [
        TurnRecord(turn=1, agent_name="w", role=Role.WORKER,
                   raw_output="Answer: C", processed_output="Answer: C"),
        TurnRecord(turn=2, agent_name="v", role=Role.VERIFIER,
                   raw_output="Answer: B. VERDICT: ACCEPT",
                   processed_output="Answer: B. VERDICT: ACCEPT"),
    ]
    assert hero_quality(Trajectory(task=task, turns=turns, final_answer="Answer: C")) == 1.0

    # And when the final answer is unparseable, the committed reference is recovered
    # from the Worker turn (never the Verifier), so quality is still 1.0 — not
    # driven by, nor diluted by, the checker's words.
    assert hero_quality(Trajectory(task=task, turns=turns, final_answer="looks good")) == 1.0


# --------------------------------------------------------------------------- #
# hero_bucket_bonus
# --------------------------------------------------------------------------- #
def test_bucket_bonus_minmax_within_buckets():
    cfg = FitnessConfig(hero_dense=True, hero_bonus=0.05)
    # correct bucket qualities .2/.8/.5 ; incorrect .9/.1
    b = hero_bucket_bonus([0.2, 0.8, 0.5, 0.9, 0.1], [1, 1, 1, 0, 0], cfg)
    # within the CORRECT bucket: min(.2)->0, max(.8)->hero_bonus, .5 in between
    assert b[0] == pytest.approx(0.0)
    assert b[1] == pytest.approx(0.05)
    assert 0.0 < b[2] < 0.05
    # within the INCORRECT bucket: max(.9)->hero_bonus, min(.1)->0 (normalized separately)
    assert b[3] == pytest.approx(0.05)
    assert b[4] == pytest.approx(0.0)


def test_bucket_bonus_degenerate_bucket_is_neutral():
    cfg = FitnessConfig(hero_dense=True, hero_bonus=0.05)
    # all-equal quality (or single member) -> neutral 0.5 * hero_bonus
    assert np.allclose(hero_bucket_bonus([0.7, 0.7], [1, 1], cfg), 0.025)
    assert np.allclose(hero_bucket_bonus([0.3], [0], cfg), 0.025)


def test_bucket_bonus_zero_and_empty():
    off = FitnessConfig(hero_dense=True, hero_bonus=0.0)
    assert np.allclose(hero_bucket_bonus([0.2, 0.8], [1, 0], off), 0.0)
    assert hero_bucket_bonus([], [], FitnessConfig(hero_dense=True)).shape == (0,)


def test_bucket_bonus_bounded():
    cfg = FitnessConfig(hero_dense=True, hero_bonus=0.05)
    b = hero_bucket_bonus(np.linspace(0, 1, 9), [1, 1, 1, 1, 0, 0, 0, 0, 1], cfg)
    assert np.all(b >= 0.0) and np.all(b <= 0.05)


# --------------------------------------------------------------------------- #
# THE invariant: adding HERO never makes a wrong trajectory outrank a correct one
# --------------------------------------------------------------------------- #
def test_correct_always_outranks_wrong_with_hero():
    cfg = FitnessConfig(format_bonus=0.05, turn_penalty=0.05, hero_dense=True, hero_bonus=0.05)
    # Build the worst plausible CORRECT reward and the best plausible WRONG reward,
    # combining shaped_reward (format/turn) with the full hero bonus.
    # correct: no format bonus, max turn penalty, and worst-in-bucket hero (0).
    worst_correct = shaped_reward(1, False, 5, 5, cfg) + 0.0
    # wrong: format bonus, no turn penalty, best-in-bucket hero (+hero_bonus).
    best_wrong = shaped_reward(0, True, 1, 5, cfg) + cfg.hero_bonus
    assert worst_correct > best_wrong

    # Exhaustive: any correct beats any wrong across turn counts and hero bonuses.
    for nt in range(1, 6):
        for hero_c in (0.0, cfg.hero_bonus):
            for hero_w in (0.0, cfg.hero_bonus):
                for ha_c in (True, False):
                    for ha_w in (True, False):
                        c = shaped_reward(1, ha_c, nt, 5, cfg) + hero_c
                        w = shaped_reward(0, ha_w, 1, 5, cfg) + hero_w
                        assert c > w


# --------------------------------------------------------------------------- #
# config wiring
# --------------------------------------------------------------------------- #
def test_config_hero_fields():
    d = FitnessConfig()
    assert d.hero_dense is False and d.hero_bonus == 0.05
    assert d.shaping_active is True  # nonzero format/turn by default
    # hero_dense alone activates shaping even with zero format/turn.
    only_hero = FitnessConfig(format_bonus=0.0, turn_penalty=0.0, hero_dense=True)
    assert only_hero.shaping_active is True
    # from_dict round-trip.
    c = FitnessConfig.from_dict({"hero_dense": True, "hero_bonus": 0.1})
    assert c.hero_dense is True and c.hero_bonus == 0.1
    # default-off: no hero, no other shaping -> inactive.
    off = FitnessConfig(format_bonus=0.0, turn_penalty=0.0, hero_dense=False)
    assert off.shaping_active is False
