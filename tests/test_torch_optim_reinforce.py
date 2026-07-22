"""Offline tests for the REINFORCE R8 baseline trainer (docs/SPEC.md L420/L462). CPU-only.

Named ``test_torch_*`` so pytest's alphabetical collection runs this AFTER the modules that
assert ``"torch" not in sys.modules``; importing torch earlier would break them.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from trinity.coordinator.params import make_spec
from trinity.optim.reinforce import (
    REINFORCE_BASELINE_DECAY,
    REINFORCE_BATCH_SIZE,
    REINFORCE_ITERATIONS,
    REINFORCE_LR,
    MovingBaseline,
    REINFORCETrainer,
    budget_matched_batch,
    run_reinforce,
    sample_actions,
)


# ---------------------------------------------------------------------------
# module import contract + SPEC defaults
# ---------------------------------------------------------------------------
def test_importing_the_reinforce_module_does_not_pull_torch_at_module_scope():
    import trinity.optim.reinforce as rf_mod

    assert "torch" not in vars(rf_mod)


def test_spec_defaults_match_the_spec_recipe():
    # docs/SPEC.md L420/L462 and configs/trinity.yaml: batch = m_CMA*lambda = 528, 60 iters.
    assert REINFORCE_BATCH_SIZE == 528
    assert REINFORCE_ITERATIONS == 60
    assert REINFORCE_LR > 0
    assert 0.0 <= REINFORCE_BASELINE_DECAY < 1.0


# ---------------------------------------------------------------------------
# budget_matched_batch
# ---------------------------------------------------------------------------
def test_budget_match_reproduces_the_configured_batch_size():
    # lambda=33, m_cma=16 -> 528, exactly the trinity.yaml reinforce.batch_size.
    assert budget_matched_batch(33, 16) == 528
    assert budget_matched_batch(33, 16) == REINFORCE_BATCH_SIZE


@pytest.mark.parametrize("popsize, m_cma", [(0, 16), (33, 0), (-1, 16)])
def test_budget_match_rejects_non_positive_counts(popsize, m_cma):
    with pytest.raises(ValueError):
        budget_matched_batch(popsize, m_cma)


# ---------------------------------------------------------------------------
# MovingBaseline
# ---------------------------------------------------------------------------
def test_baseline_initializes_on_the_first_observation():
    # A cold start at 0.0 would make every early advantage look large and positive on a
    # non-negative reward, biasing the first updates.
    b = MovingBaseline(decay=0.9)
    assert b.value == 0.0 and b.count == 0
    assert b.update(0.7) == pytest.approx(0.7)
    assert b.count == 1


def test_baseline_tracks_an_exponential_moving_average():
    b = MovingBaseline(decay=0.5)
    b.update(1.0)
    assert b.update(0.0) == pytest.approx(0.5)
    assert b.update(0.0) == pytest.approx(0.25)


def test_baseline_converges_toward_a_constant_reward():
    b = MovingBaseline(decay=0.9)
    for _ in range(200):
        b.update(0.42)
    assert b.value == pytest.approx(0.42, abs=1e-6)


def test_zero_decay_baseline_tracks_the_latest_observation():
    b = MovingBaseline(decay=0.0)
    b.update(1.0)
    assert b.update(0.3) == pytest.approx(0.3)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_baseline_rejects_out_of_range_decay(decay):
    with pytest.raises(ValueError):
        MovingBaseline(decay=decay)


# ---------------------------------------------------------------------------
# sample_actions
# ---------------------------------------------------------------------------
def test_degenerate_distributions_sample_their_only_supported_action():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    actions = sample_actions(probs, np.random.default_rng(0))
    np.testing.assert_array_equal(actions, [0, 2, 1])


def test_sampled_action_frequencies_track_the_distribution():
    probs = np.tile(np.array([0.7, 0.2, 0.1]), (4000, 1))
    actions = sample_actions(probs, np.random.default_rng(0))
    freq = np.bincount(actions, minlength=3) / len(actions)
    np.testing.assert_allclose(freq, [0.7, 0.2, 0.1], atol=0.03)


def test_unnormalized_rows_are_renormalized_before_sampling():
    probs = np.tile(np.array([2.0, 2.0]), (2000, 1))
    freq = np.bincount(sample_actions(probs, np.random.default_rng(0)), minlength=2) / 2000
    np.testing.assert_allclose(freq, [0.5, 0.5], atol=0.05)


def test_sampling_is_reproducible_per_generator_seed():
    probs = np.tile(np.array([0.5, 0.3, 0.2]), (50, 1))
    a = sample_actions(probs, np.random.default_rng(3))
    b = sample_actions(probs, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_sampled_actions_are_always_valid_indices():
    probs = np.tile(np.array([0.999999, 1e-6]), (500, 1))
    actions = sample_actions(probs, np.random.default_rng(0))
    assert actions.min() >= 0 and actions.max() <= 1


@pytest.mark.parametrize(
    "probs",
    [np.zeros(3), np.zeros((2, 3))],  # not 2-D; all-zero rows
)
def test_sample_actions_rejects_malformed_input(probs):
    with pytest.raises(ValueError):
        sample_actions(probs, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# run_reinforce  (torch, CPU)
# ---------------------------------------------------------------------------
def _bandit_problem(n_per_class: int = 40, d_h: int = 8, n_models: int = 3, seed: int = 0):
    """Contextual bandit where each feature cluster is solved only by its own model."""
    rng = np.random.default_rng(seed)
    feats, solve = [], []
    for m in range(n_models):
        centre = np.zeros(d_h)
        centre[m] = 1.0
        feats.append(centre + 0.01 * rng.standard_normal((n_per_class, d_h)))
        row = np.zeros((n_per_class, n_models))
        row[:, m] = 1.0
        solve.append(row)
    return np.vstack(feats), np.vstack(solve)


def test_reinforce_learns_the_correct_routing_on_a_contextual_bandit():
    feats, solve = _bandit_problem()
    W = run_reinforce(
        feats, solve, batch_size=64, iterations=400, lr=0.05, seed=0, return_history=False
    )
    predicted = np.argmax(feats @ W.T, axis=1)
    expected = np.argmax(solve, axis=1)
    accuracy = float(np.mean(predicted == expected))
    assert accuracy > 0.9, f"REINFORCE should solve this bandit, got {accuracy:.2f}"


def test_mean_reward_improves_over_iterations():
    feats, solve = _bandit_problem()
    _W, history = run_reinforce(feats, solve, batch_size=64, iterations=400, lr=0.05, seed=0)
    early = float(np.mean([h["mean_reward"] for h in history[:20]]))
    late = float(np.mean([h["mean_reward"] for h in history[-20:]]))
    assert late > early


def test_history_records_one_entry_per_iteration_with_the_expected_keys():
    feats, solve = _bandit_problem(n_per_class=5, d_h=4)
    _W, history = run_reinforce(feats, solve, batch_size=8, iterations=7, lr=0.01, seed=0)
    assert len(history) == 7
    assert [h["iteration"] for h in history] == list(range(7))
    for rec in history:
        assert set(rec) == {"iteration", "mean_reward", "baseline", "mean_advantage", "loss"}
        assert np.isfinite(rec["loss"])


def test_advantage_baseline_is_carried_in_from_previous_iterations():
    """The baseline must depend only on PAST batches, or the gradient picks up a bias.

    Iteration 0 has no history and falls back to its own batch mean (advantage is then
    exactly zero-mean). From iteration 1 on, ``mean_advantage`` must equal
    ``mean_reward - baseline`` for a baseline that is strictly the carried-in EMA.
    """
    feats, solve = _bandit_problem(n_per_class=20, d_h=4)
    _W, history = run_reinforce(feats, solve, batch_size=32, iterations=25, lr=0.02, seed=0)

    assert history[0]["mean_advantage"] == pytest.approx(0.0, abs=1e-12)

    decay = REINFORCE_BASELINE_DECAY
    carried = history[0]["mean_reward"]  # EMA initializes on the first observation
    for rec in history[1:]:
        assert rec["mean_advantage"] == pytest.approx(rec["mean_reward"] - carried)
        carried = decay * carried + (1.0 - decay) * rec["mean_reward"]


def test_reinforce_is_reproducible_per_seed_and_varies_across_seeds():
    feats, solve = _bandit_problem(n_per_class=10, d_h=4)
    kw = dict(batch_size=16, iterations=20, lr=0.01, return_history=False)
    a = run_reinforce(feats, solve, seed=1, **kw)
    b = run_reinforce(feats, solve, seed=1, **kw)
    c = run_reinforce(feats, solve, seed=2, **kw)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_batch_size_is_independent_of_the_labelled_set_size():
    # Sampling WITH replacement keeps per-iteration env spend exactly batch_size, which is
    # what makes the budget match against sep-CMA-ES exact even on a small label set.
    feats, solve = _bandit_problem(n_per_class=2, d_h=4)  # only 6 rows
    _W, history = run_reinforce(feats, solve, batch_size=100, iterations=3, lr=0.01, seed=0)
    assert len(history) == 3


def test_fitted_head_has_one_row_per_pool_model():
    feats, solve = _bandit_problem(n_per_class=5, d_h=6)
    W = run_reinforce(
        feats, solve, batch_size=8, iterations=3, lr=0.01, seed=0, return_history=False
    )
    assert W.shape == (3, 6)
    assert W.dtype == np.float64


@pytest.mark.parametrize(
    "feats, solve, kwargs",
    [
        (np.zeros((4,)), np.zeros((4, 3)), {}),          # features not 2-D
        (np.zeros((4, 6)), np.zeros((4,)), {}),          # rewards not 2-D
        (np.zeros((4, 6)), np.zeros((5, 3)), {}),        # row-count mismatch
        (np.zeros((0, 6)), np.zeros((0, 3)), {}),        # empty
        (np.zeros((4, 6)), np.zeros((4, 3)), {"batch_size": 0}),
        (np.zeros((4, 6)), np.zeros((4, 3)), {"iterations": 0}),
        (np.zeros((4, 6)), np.zeros((4, 3)), {"lr": 0.0}),
    ],
)
def test_run_reinforce_rejects_malformed_input(feats, solve, kwargs):
    with pytest.raises(ValueError):
        run_reinforce(feats, solve, **kwargs)


# ---------------------------------------------------------------------------
# REINFORCETrainer  (BaseTrainer contract)
# ---------------------------------------------------------------------------
def _run_trainer(tmp_path, **kwargs):
    spec = make_spec(n_a=6, d_h=8, n_svf=16)
    feats, solve = _bandit_problem(n_per_class=10, d_h=8)
    trainer = REINFORCETrainer(batch_size=32, iterations=20, lr=0.05, seed=0)
    summary = asyncio.run(
        trainer.train(
            None, None, [],
            spec=spec, features=feats, solve_prob=solve,
            run_dir=tmp_path, benchmark="mmlu", **kwargs,
        )
    )
    return spec, summary


def test_trainer_returns_the_base_trainer_summary_keys(tmp_path):
    _spec, summary = _run_trainer(tmp_path)
    for key in ("benchmark", "best_fitness", "best_theta_path", "run_dir", "total_cost_usd"):
        assert key in summary
    assert summary["trainer"] == "reinforce"
    assert summary["benchmark"] == "mmlu"


def test_trainer_writes_a_theta_of_exactly_spec_n_total(tmp_path):
    spec, summary = _run_trainer(tmp_path)
    theta = np.load(summary["best_theta_path"])
    assert theta.shape == (spec.n_total,)


def test_trainer_leaves_role_rows_uniform_and_svf_identity(tmp_path):
    spec, summary = _run_trainer(tmp_path)
    theta = np.load(summary["best_theta_path"])
    head = theta[: spec.n_head].reshape(spec.head_shape)
    np.testing.assert_allclose(head[3:], 0.0)
    np.testing.assert_allclose(theta[spec.n_head :], 1.0)
    assert np.any(head[:3] != 0.0), "agent rows should have been trained"


def test_trainer_reports_the_budget_it_actually_spent(tmp_path):
    _spec, summary = _run_trainer(tmp_path)
    assert summary["env_interactions"] == 32 * 20


def test_best_fitness_is_the_final_mean_reward_not_the_best_ever(tmp_path):
    # Reporting the best-ever batch would let a lucky early draw flatter the baseline.
    _spec, summary = _run_trainer(tmp_path)
    assert summary["best_fitness"] == pytest.approx(summary["final_mean_reward"])
    assert summary["best_fitness"] == pytest.approx(summary["history"][-1]["mean_reward"])


def test_trainer_reports_zero_api_cost(tmp_path):
    _spec, summary = _run_trainer(tmp_path)
    assert summary["total_cost_usd"] == 0.0


def test_trainer_summary_is_json_serializable_and_persisted(tmp_path):
    _spec, summary = _run_trainer(tmp_path)
    json.dumps(summary)
    assert json.loads((tmp_path / "summary.json").read_text())["trainer"] == "reinforce"
    assert json.loads((tmp_path / "history.json").read_text())


def test_trainer_reads_rewards_from_an_oracle_matrix(tmp_path):
    matrix = {
        "benchmark": "mmlu",
        "tasks": [
            {"id": "q1", "per_model": {"a": [1, 1], "b": [0, 0], "c": [0, 1]}},
            {"id": "q2", "per_model": {"a": [0, 0], "b": [1, 1], "c": [0, 0]}},
        ],
    }
    path = tmp_path / "oracle_matrix_mmlu.json"
    path.write_text(json.dumps(matrix))
    trainer = REINFORCETrainer(batch_size=4, iterations=2, lr=0.01, seed=0)
    summary = asyncio.run(
        trainer.train(
            None, None, [],
            spec=make_spec(n_a=6, d_h=4, n_svf=8), features=np.eye(2, 4),
            matrix_path=str(path), run_dir=tmp_path / "run",
        )
    )
    assert summary["pool"] == ["a", "b", "c"]
    assert summary["n_labelled"] == 2


def test_trainer_rejects_features_misaligned_with_labels(tmp_path):
    feats, solve = _bandit_problem(n_per_class=10, d_h=8)
    trainer = REINFORCETrainer(batch_size=8, iterations=2, lr=0.01, seed=0)
    with pytest.raises(ValueError, match="aligned row-wise"):
        asyncio.run(
            trainer.train(
                None, None, [], spec=make_spec(n_a=6, d_h=8, n_svf=16),
                features=feats[:-1], solve_prob=solve, run_dir=tmp_path,
            )
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"spec": None}, "spec"),
        ({"run_dir": None}, "run_dir"),
        ({"features": None}, "features"),
    ],
)
def test_trainer_requires_its_mandatory_arguments(tmp_path, kwargs, match):
    base = {
        "spec": make_spec(n_a=6, d_h=4, n_svf=8),
        "features": np.eye(2, 4),
        "solve_prob": np.eye(2, 3),
        "run_dir": tmp_path,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        asyncio.run(REINFORCETrainer(iterations=1, batch_size=2).train(None, None, [], **base))


def test_trainer_requires_labels_from_somewhere(tmp_path):
    with pytest.raises(ValueError, match="solve_prob"):
        asyncio.run(
            REINFORCETrainer(iterations=1, batch_size=2).train(
                None, None, [],
                spec=make_spec(n_a=6, d_h=4, n_svf=8),
                features=np.eye(2, 4), run_dir=tmp_path,
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"batch_size": 0}, {"iterations": 0}, {"lr": 0.0}, {"baseline_decay": 1.0}],
)
def test_trainer_rejects_out_of_range_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        REINFORCETrainer(**kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
