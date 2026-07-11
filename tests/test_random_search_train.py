"""Offline coverage for ``RandomSearchTrainer.train()`` — the R8 RS baseline run.

``test_random_search_baseline.py`` covers the pure-numpy RS core (sampler,
budget-match, keep-best, the synthetic-objective driver) and the trainer's
*validation guards*, but the async ``train()`` **happy path** — the actual
budget-matched run that samples candidates, scores them through
``evaluate_population``, streams ``best_theta.npy`` / ``history.json``, keeps the
best, and writes ``summary.json`` — was untested (it sits past the early
``spec``/``run_dir`` guards the existing tests stop at).

These tests drive ``train()`` end-to-end with a fake pool and a stubbed
``evaluate_population`` (no GPU, no network, no torch). The stub honours the real
contract — it calls ``minibatch_fn(i)`` and ``on_candidate(i, fit, elapsed)`` for
each candidate and returns one fitness per candidate — so the trainer's own
orchestration (budget-match default, per-candidate minibatch seeding, streaming
artifact writes, keep-best selection, summary contract) is what runs.
"""
import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np

from trinity.optim.baselines import (
    RandomSearchTrainer,
    budget_matched_candidates,
    sample_candidates,
)


def test_no_torch_imported():
    assert "torch" not in sys.modules, "RS trainer must run without torch"


class _FakePool:
    """Minimal pool: slot-ordered ``models`` and a running ``total_cost_usd``."""

    def __init__(self, models, cost=0.0):
        self.models = list(models)
        self.total_cost_usd = float(cost)


def _spec(n_total=8):
    return SimpleNamespace(n_total=n_total)


def _tasks(n=20, benchmark="math500"):
    return [SimpleNamespace(task_id=f"t{i}", benchmark=benchmark) for i in range(n)]


def _install_fake_eval(monkeypatch, fits, capture):
    """Replace ``evaluate_population`` with a contract-faithful offline stub."""

    async def fake_evaluate_population(
        thetas, spec, policy, pool, pool_models, minibatch_fn, *,
        sample=True, on_candidate=None, fitness_cfg=None, max_turns=5, **run_kwargs,
    ):
        capture["n_thetas"] = len(thetas)
        capture["sample"] = sample
        capture["fitness_cfg"] = fitness_cfg
        capture["max_turns"] = max_turns
        capture["pool_models"] = list(pool_models)
        capture["run_kwargs"] = dict(run_kwargs)
        capture["minibatch_ids"] = []
        out = []
        for i in range(len(thetas)):
            mb = minibatch_fn(i)  # exercises the per-candidate seeding closure
            capture["minibatch_ids"].append([t.task_id for t in mb])
            f = float(fits[i])
            out.append(f)
            if on_candidate is not None:
                on_candidate(i, f, 0.1 * (i + 1))
        return out

    monkeypatch.setattr(
        "trinity.optim.baselines.evaluate_population", fake_evaluate_population
    )


def _run_train(trainer, *, tasks, pool, run_dir, capture, monkeypatch, fits, **kw):
    _install_fake_eval(monkeypatch, fits, capture)
    return asyncio.run(
        trainer.train(
            policy=object(), pool=pool, tasks=tasks, spec=_spec(),
            run_dir=run_dir, **kw,
        )
    )


# --------------------------------------------------------------------------- #
# happy path: run, keep-best, artifacts
# --------------------------------------------------------------------------- #
def test_train_runs_scores_and_writes_all_artifacts(tmp_path, monkeypatch):
    fits = [0.2, 0.9, 0.5]  # best is candidate index 1
    cap: dict = {}
    pool = _FakePool(["m0", "m1"], cost=1.25)
    run_dir = tmp_path / "run"

    summary = _run_train(
        RandomSearchTrainer(trials_per_candidate=3, seed=7),
        tasks=_tasks(), pool=pool, run_dir=run_dir, capture=cap,
        monkeypatch=monkeypatch, fits=fits, num_candidates=3, benchmark="math500",
    )

    # --- returned summary contract ---
    assert summary["trainer"] == "random_search"
    assert summary["benchmark"] == "math500"
    assert summary["num_candidates"] == 3
    assert summary["trials_per_candidate"] == 3
    assert summary["best_fitness"] == 0.9
    assert summary["best_trial"] == 1
    assert summary["pool"] == ["m0", "m1"]
    assert summary["sample_range"] == [-0.5, 0.5]
    assert summary["total_cost_usd"] == 1.25
    assert summary["seed"] == 7

    # --- best_theta.npy holds the winning vector (index 1 of the seeded draw) ---
    expected = sample_candidates(8, 3, seed=7, low=-0.5, high=0.5)
    saved = np.load(run_dir / "best_theta.npy")
    assert np.allclose(saved, expected[1])

    # --- history.json: one row per candidate, best-so-far monotone, ends at max ---
    history = json.loads((run_dir / "history.json").read_text())
    assert [h["trial"] for h in history] == [0, 1, 2]
    best_curve = [h["best_fitness"] for h in history]
    assert best_curve == sorted(best_curve)  # non-decreasing
    assert best_curve[-1] == 0.9
    assert history[0]["fitness"] == 0.2

    # --- summary.json on disk equals the returned summary ---
    assert json.loads((run_dir / "summary.json").read_text())["best_fitness"] == 0.9


def test_train_keeps_best_when_winner_is_first_candidate(tmp_path, monkeypatch):
    # Winner at index 0: the streaming best-so-far must never regress afterwards.
    fits = [0.8, 0.1, 0.4]
    cap: dict = {}
    run_dir = tmp_path / "run"
    summary = _run_train(
        RandomSearchTrainer(trials_per_candidate=2, seed=0),
        tasks=_tasks(), pool=_FakePool(["a", "b"]), run_dir=run_dir,
        capture=cap, monkeypatch=monkeypatch, fits=fits, num_candidates=3,
    )
    assert summary["best_trial"] == 0
    assert summary["best_fitness"] == 0.8
    saved = np.load(run_dir / "best_theta.npy")
    assert np.allclose(saved, sample_candidates(8, 3, seed=0)[0])


# --------------------------------------------------------------------------- #
# budget-matched default
# --------------------------------------------------------------------------- #
def test_train_defaults_num_candidates_to_budget_match(tmp_path, monkeypatch):
    popsize, m_cma, generations, trials = 2, 3, 2, 1
    expected = budget_matched_candidates(popsize, m_cma, generations, trials)
    assert expected == 12
    cap: dict = {}
    summary = _run_train(
        RandomSearchTrainer(trials_per_candidate=trials, seed=1),
        tasks=_tasks(), pool=_FakePool(["a"]), run_dir=tmp_path / "run",
        capture=cap, monkeypatch=monkeypatch, fits=[i / 100 for i in range(expected)],
        popsize=popsize, m_cma=m_cma, generations=generations,
    )
    assert summary["num_candidates"] == expected
    assert cap["n_thetas"] == expected


# --------------------------------------------------------------------------- #
# benchmark inference, defaults, forwarding
# --------------------------------------------------------------------------- #
def test_train_infers_benchmark_from_first_task(tmp_path, monkeypatch):
    cap: dict = {}
    summary = _run_train(
        RandomSearchTrainer(trials_per_candidate=1, seed=0),
        tasks=_tasks(benchmark="gpqa"), pool=_FakePool(["a"]),
        run_dir=tmp_path / "run", capture=cap, monkeypatch=monkeypatch,
        fits=[0.3, 0.6], num_candidates=2,  # benchmark not passed -> inferred
    )
    assert summary["benchmark"] == "gpqa"


def test_train_defaults_pool_models_to_pool_dot_models(tmp_path, monkeypatch):
    cap: dict = {}
    _run_train(
        RandomSearchTrainer(trials_per_candidate=1, seed=0),
        tasks=_tasks(), pool=_FakePool(["deepseek", "glm", "kimi"]),
        run_dir=tmp_path / "run", capture=cap, monkeypatch=monkeypatch,
        fits=[0.1, 0.2], num_candidates=2,  # pool_models not passed
    )
    assert cap["pool_models"] == ["deepseek", "glm", "kimi"]


def test_train_uses_plain_binary_reward_and_forwards_run_kwargs(tmp_path, monkeypatch):
    cap: dict = {}
    _run_train(
        RandomSearchTrainer(trials_per_candidate=1, seed=0),
        tasks=_tasks(), pool=_FakePool(["a"]), run_dir=tmp_path / "run",
        capture=cap, monkeypatch=monkeypatch, fits=[0.5, 0.5], num_candidates=2,
        max_turns=9, sample=False, max_tokens=1234, reasoning="high",
    )
    # RS default: no fitness shaping (fitness_cfg stays None -> plain binary accuracy).
    assert cap["fitness_cfg"] is None
    assert cap["sample"] is False
    assert cap["max_turns"] == 9
    # Extra run_kwargs reach the trajectory runner untouched.
    assert cap["run_kwargs"]["max_tokens"] == 1234
    assert cap["run_kwargs"]["reasoning"] == "high"


# --------------------------------------------------------------------------- #
# guards and progress callback
# --------------------------------------------------------------------------- #
def test_train_rejects_budget_matched_zero_candidates(tmp_path, monkeypatch):
    # popsize=0 -> budget-matched count floors to 0 -> train must refuse to run.
    import pytest

    _install_fake_eval(monkeypatch, [], {})
    with pytest.raises(ValueError, match="num_candidates must be >= 1"):
        asyncio.run(
            RandomSearchTrainer(trials_per_candidate=1).train(
                policy=object(), pool=_FakePool(["a"]), tasks=_tasks(),
                spec=_spec(), run_dir=tmp_path / "run", popsize=0,
            )
        )


def test_train_invokes_user_on_candidate_callback(tmp_path, monkeypatch):
    seen: list = []
    cap: dict = {}
    _run_train(
        RandomSearchTrainer(trials_per_candidate=1, seed=0),
        tasks=_tasks(), pool=_FakePool(["a"]), run_dir=tmp_path / "run",
        capture=cap, monkeypatch=monkeypatch, fits=[0.2, 0.7, 0.4],
        num_candidates=3, on_candidate=lambda i, f, s: seen.append((i, f)),
    )
    assert seen == [(0, 0.2), (1, 0.7), (2, 0.4)]


# --------------------------------------------------------------------------- #
# per-candidate minibatch seeding
# --------------------------------------------------------------------------- #
def test_train_minibatch_is_sized_and_reproducibly_seeded(tmp_path, monkeypatch):
    # Each candidate scores its own minibatch of `trials_per_candidate` tasks, seeded
    # from `seed * 100000 + i`, so the whole run reproduces across processes.
    def once():
        cap: dict = {}
        _run_train(
            RandomSearchTrainer(trials_per_candidate=4, seed=3),
            tasks=_tasks(30), pool=_FakePool(["a"]), run_dir=tmp_path / "r",
            capture=cap, monkeypatch=monkeypatch, fits=[0.1, 0.2, 0.3],
            num_candidates=3,
        )
        return cap["minibatch_ids"]

    first = once()
    assert all(len(mb) == 4 for mb in first)          # correct minibatch size
    assert first == once()                            # identical under the same seed
