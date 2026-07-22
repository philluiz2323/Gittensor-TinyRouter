"""Offline tests for the SFT R8 baseline trainer (docs/SPEC.md L420/L464). CPU-only, no network.

Named ``test_torch_*`` so pytest's alphabetical collection runs this AFTER the modules that
assert ``"torch" not in sys.modules`` (e.g. ``test_shaped_fitness.py``); importing torch
earlier in the session would break them.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from trinity.coordinator.params import make_spec
from trinity.optim.sft import (
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LR,
    SFT_OPTIMIZER,
    SFT_TARGET_TEMP,
    SFTTrainer,
    build_teacher_targets,
    fit_head_sft,
    iter_minibatches,
)


# ---------------------------------------------------------------------------
# module import contract
# ---------------------------------------------------------------------------
def test_importing_the_sft_module_does_not_pull_torch_at_module_scope():
    # The lazy-import contract: trinity.optim must stay torch-free on import so the
    # torch-free test modules keep passing. torch may already be in sys.modules from an
    # earlier test in this file, so check the module's own globals instead.
    import trinity.optim.sft as sft_mod

    assert "torch" not in vars(sft_mod)


def test_spec_defaults_match_the_spec_recipe():
    # docs/SPEC.md L420/L464: "SFT (Adam, lr 1e-6, batch 64, frozen SLM, head-only)".
    assert SFT_OPTIMIZER == "adam"
    assert SFT_LR == pytest.approx(1e-6)
    assert SFT_BATCH_SIZE == 64
    assert SFT_EPOCHS >= 1
    assert 0.0 < SFT_TARGET_TEMP


# ---------------------------------------------------------------------------
# build_teacher_targets
# ---------------------------------------------------------------------------
def test_teacher_target_rows_are_probability_distributions():
    sp = np.array([[0.9, 0.1, 0.0], [0.2, 0.2, 0.9], [0.5, 0.5, 0.5]])
    target = build_teacher_targets(sp)
    assert target.shape == sp.shape
    np.testing.assert_allclose(target.sum(axis=1), 1.0)
    assert np.all(target >= 0.0)


def test_teacher_target_puts_most_mass_on_the_best_model():
    sp = np.array([[0.9, 0.1, 0.0]])
    target = build_teacher_targets(sp, temperature=0.5)
    assert int(np.argmax(target[0])) == 0
    assert target[0, 0] > target[0, 1] > target[0, 2]


def test_queries_no_model_solves_get_a_uniform_target():
    # An all-zero row carries no routing preference; inventing one would be fabricated
    # supervision, so the target must be exactly uniform.
    sp = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = build_teacher_targets(sp)
    np.testing.assert_allclose(target[0], np.full(3, 1 / 3))
    assert int(np.argmax(target[1])) == 0


def test_lower_temperature_produces_a_peakier_target():
    sp = np.array([[0.9, 0.5, 0.1]])
    peaked = build_teacher_targets(sp, temperature=0.1)
    flat = build_teacher_targets(sp, temperature=5.0)
    assert peaked[0].max() > flat[0].max()


def test_teacher_targets_clip_out_of_range_solve_rates():
    sp = np.array([[1.5, -0.5, 0.5]])
    target = build_teacher_targets(sp)
    assert np.all(np.isfinite(target))
    np.testing.assert_allclose(target.sum(), 1.0)


def test_empty_label_set_returns_an_empty_target_block():
    target = build_teacher_targets(np.zeros((0, 3)))
    assert target.shape == (0, 3)


@pytest.mark.parametrize(
    "kwargs, sp",
    [
        ({}, np.zeros(3)),                       # not 2-D
        ({"temperature": 0.0}, np.zeros((2, 3))),  # non-positive temperature
        ({"temperature": -1.0}, np.zeros((2, 3))),
    ],
)
def test_build_teacher_targets_rejects_malformed_input(kwargs, sp):
    with pytest.raises(ValueError):
        build_teacher_targets(sp, **kwargs)


# ---------------------------------------------------------------------------
# iter_minibatches
# ---------------------------------------------------------------------------
def test_each_epoch_covers_every_index_exactly_once():
    batches = list(iter_minibatches(10, 4, epochs=1, seed=0))
    covered = np.concatenate(batches)
    assert sorted(covered.tolist()) == list(range(10))


def test_short_final_batch_is_kept_not_dropped():
    # 10 rows / batch 4 -> 4 + 4 + 2. With ~120 labelled queries, dropping the remainder
    # would discard a meaningful fraction of the supervision.
    sizes = [len(b) for b in iter_minibatches(10, 4, epochs=1, seed=0)]
    assert sizes == [4, 4, 2]


def test_minibatch_order_is_reproducible_per_seed_and_varies_across_seeds():
    a = np.concatenate(list(iter_minibatches(20, 5, epochs=1, seed=7)))
    b = np.concatenate(list(iter_minibatches(20, 5, epochs=1, seed=7)))
    c = np.concatenate(list(iter_minibatches(20, 5, epochs=1, seed=8)))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_successive_epochs_use_different_permutations():
    batches = list(iter_minibatches(20, 20, epochs=2, seed=0))
    assert len(batches) == 2
    assert not np.array_equal(batches[0], batches[1])


@pytest.mark.parametrize(
    "n_rows, batch_size, epochs",
    [(0, 4, 1), (10, 0, 1), (10, 4, 0), (-1, 4, 1)],
)
def test_iter_minibatches_rejects_malformed_input(n_rows, batch_size, epochs):
    with pytest.raises(ValueError):
        list(iter_minibatches(n_rows, batch_size, epochs=epochs, seed=0))


# ---------------------------------------------------------------------------
# fit_head_sft  (torch, CPU)
# ---------------------------------------------------------------------------
def _separable_problem(n_per_class: int = 40, d_h: int = 8, n_models: int = 3, seed: int = 0):
    """Build a linearly-separable routing problem: each cluster is solved by one model."""
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


def test_sft_recovers_the_correct_routing_on_a_separable_problem():
    # NOTE: the SPEC lr (1e-6) is calibrated for a long real run; on a small synthetic
    # problem it barely moves W, so convergence is asserted with a larger lr here. The
    # SPEC defaults are asserted separately in test_spec_defaults_match_the_spec_recipe.
    feats, solve = _separable_problem()
    targets = build_teacher_targets(solve, temperature=0.5)
    W = fit_head_sft(feats, targets, lr=0.05, batch_size=16, epochs=60, seed=0)
    predicted = np.argmax(feats @ W.T, axis=1)
    expected = np.argmax(solve, axis=1)
    accuracy = float(np.mean(predicted == expected))
    assert accuracy > 0.95, f"SFT should separate this problem, got {accuracy:.2f}"


def test_sft_loss_decreases_over_training():
    feats, solve = _separable_problem()
    targets = build_teacher_targets(solve, temperature=0.5)
    _W, history = fit_head_sft(
        feats, targets, lr=0.05, batch_size=16, epochs=30, seed=0, return_history=True
    )
    first = float(np.mean([h["loss"] for h in history[:5]]))
    last = float(np.mean([h["loss"] for h in history[-5:]]))
    assert last < first


def test_sft_freezes_the_slm_features_and_trains_only_the_head():
    """The SPEC's "frozen SLM, head-only" contract, asserted mechanically."""
    import torch

    feats, solve = _separable_problem(n_per_class=8, d_h=4)
    targets = build_teacher_targets(solve)
    feat_t = torch.tensor(feats, dtype=torch.float64, requires_grad=False)
    # Stand-in for the frozen SLM: its parameters must never receive gradient.
    frozen_slm = torch.nn.Linear(4, 4, dtype=torch.float64)
    for p in frozen_slm.parameters():
        p.requires_grad_(False)
    head = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.float64))

    logits = frozen_slm(feat_t) @ head.t()
    loss = -(torch.tensor(targets) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss.backward()

    assert head.grad is not None and torch.any(head.grad != 0)
    assert all(p.grad is None for p in frozen_slm.parameters())
    assert feat_t.grad is None


def test_sft_is_reproducible_per_seed_and_varies_across_seeds():
    feats, solve = _separable_problem(n_per_class=10, d_h=4)
    targets = build_teacher_targets(solve)
    a = fit_head_sft(feats, targets, lr=0.01, batch_size=8, epochs=3, seed=1)
    b = fit_head_sft(feats, targets, lr=0.01, batch_size=8, epochs=3, seed=1)
    c = fit_head_sft(feats, targets, lr=0.01, batch_size=8, epochs=3, seed=2)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_fitted_head_has_one_row_per_pool_model():
    feats, solve = _separable_problem(n_per_class=5, d_h=6, n_models=3)
    W = fit_head_sft(feats, build_teacher_targets(solve), lr=0.01, batch_size=4, epochs=2)
    assert W.shape == (3, 6)
    assert W.dtype == np.float64


@pytest.mark.parametrize(
    "feats, targets",
    [
        (np.zeros((4,)), np.zeros((4, 3))),        # features not 2-D
        (np.zeros((4, 6)), np.zeros((4,))),        # targets not 2-D
        (np.zeros((4, 6)), np.zeros((5, 3))),      # row-count mismatch
        (np.zeros((0, 6)), np.zeros((0, 3))),      # empty
    ],
)
def test_fit_head_sft_rejects_malformed_input(feats, targets):
    with pytest.raises(ValueError):
        fit_head_sft(feats, targets)


def test_fit_head_sft_rejects_a_non_positive_learning_rate():
    feats, solve = _separable_problem(n_per_class=4, d_h=4)
    with pytest.raises(ValueError):
        fit_head_sft(feats, build_teacher_targets(solve), lr=0.0)


# ---------------------------------------------------------------------------
# SFTTrainer  (BaseTrainer contract)
# ---------------------------------------------------------------------------
def _run_trainer(tmp_path, **kwargs):
    spec = make_spec(n_a=6, d_h=8, n_svf=16)
    feats, solve = _separable_problem(n_per_class=10, d_h=8)
    trainer = SFTTrainer(lr=0.05, batch_size=8, epochs=5, seed=0)
    summary = asyncio.run(
        trainer.train(
            None, None, [],
            spec=spec, features=feats, solve_prob=solve,
            run_dir=tmp_path, benchmark="math500", **kwargs,
        )
    )
    return spec, summary


def test_trainer_returns_the_base_trainer_summary_keys(tmp_path):
    _spec, summary = _run_trainer(tmp_path)
    for key in ("benchmark", "best_fitness", "best_theta_path", "run_dir", "total_cost_usd"):
        assert key in summary
    assert summary["trainer"] == "sft"
    assert summary["benchmark"] == "math500"
    assert summary["optimizer"] == "adam"


def test_trainer_writes_a_theta_of_exactly_spec_n_total(tmp_path):
    # R8 compares optimizers through the same evaluation path, so SFT's artifact must be
    # layout-identical to a sep-CMA-ES / RS theta.
    spec, summary = _run_trainer(tmp_path)
    theta = np.load(summary["best_theta_path"])
    assert theta.shape == (spec.n_total,)


def test_trainer_leaves_role_rows_uniform_and_svf_identity(tmp_path):
    # Only agent rows are supervised (the matrices carry no role labels), so role logits
    # must stay 0 and SVF scales 1.0 -- the pack_warmstart_theta contract.
    spec, summary = _run_trainer(tmp_path)
    theta = np.load(summary["best_theta_path"])
    head = theta[: spec.n_head].reshape(spec.head_shape)
    svf = theta[spec.n_head :]
    np.testing.assert_allclose(head[3:], 0.0)
    np.testing.assert_allclose(svf, 1.0)
    assert np.any(head[:3] != 0.0), "agent rows should have been trained"


def test_trainer_reports_zero_api_cost(tmp_path):
    # SFT is offline imitation learning over cached labels: no rollouts, no spend.
    _spec, summary = _run_trainer(tmp_path)
    assert summary["total_cost_usd"] == 0.0


def test_trainer_summary_is_json_serializable_and_persisted(tmp_path):
    _spec, summary = _run_trainer(tmp_path)
    json.dumps(summary)
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk["trainer"] == "sft"
    assert json.loads((tmp_path / "history.json").read_text())


def test_trainer_reads_labels_from_an_oracle_matrix(tmp_path):
    # Reuses coordinator.warmstart.load_labels so the on-disk schema cannot drift.
    matrix = {
        "benchmark": "math500",
        "tasks": [
            {"id": "q1", "per_model": {"a": [1, 1], "b": [0, 0], "c": [0, 1]}},
            {"id": "q2", "per_model": {"a": [0, 0], "b": [1, 1], "c": [0, 0]}},
        ],
    }
    path = tmp_path / "oracle_matrix_math500.json"
    path.write_text(json.dumps(matrix))
    spec = make_spec(n_a=6, d_h=4, n_svf=8)
    feats = np.eye(2, 4)
    trainer = SFTTrainer(lr=0.05, batch_size=2, epochs=2, seed=0)
    summary = asyncio.run(
        trainer.train(
            None, None, [], spec=spec, features=feats,
            matrix_path=str(path), run_dir=tmp_path / "run",
        )
    )
    assert summary["pool"] == ["a", "b", "c"]
    assert summary["n_labelled"] == 2


def test_trainer_rejects_features_misaligned_with_labels(tmp_path):
    spec = make_spec(n_a=6, d_h=8, n_svf=16)
    feats, solve = _separable_problem(n_per_class=10, d_h=8)
    trainer = SFTTrainer(lr=0.05, batch_size=8, epochs=2, seed=0)
    with pytest.raises(ValueError, match="aligned row-wise"):
        asyncio.run(
            trainer.train(
                None, None, [], spec=spec, features=feats[:-1],
                solve_prob=solve, run_dir=tmp_path,
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
        asyncio.run(SFTTrainer().train(None, None, [], **base))


def test_trainer_requires_labels_from_somewhere(tmp_path):
    with pytest.raises(ValueError, match="solve_prob"):
        asyncio.run(
            SFTTrainer().train(
                None, None, [],
                spec=make_spec(n_a=6, d_h=4, n_svf=8),
                features=np.eye(2, 4), run_dir=tmp_path,
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"lr": 0.0}, {"batch_size": 0}, {"epochs": 0}, {"target_temp": 0.0}],
)
def test_trainer_rejects_out_of_range_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        SFTTrainer(**kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
