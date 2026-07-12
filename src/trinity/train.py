"""Entrypoint: evolve the coordinator (linear head + SVF scales) with sep-CMA-ES.

One coordinator is trained per benchmark (SPEC §6.1). Each candidate θ is scored
by the mean binary reward over a freshly-sampled minibatch of `m_cma` train tasks.

Usage (on GPU 5 via scripts/run_remote.sh, or directly):
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python -m trinity.train --benchmark math500 \
        --config configs/trinity.yaml --models configs/models.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .coordinator import params as P
from .coordinator.policy import CoordinatorPolicy
from .llm.openrouter_client import OpenRouterPool
from .optim.budget import AtomicEvalBudget
from .optim.fitness import FitnessConfig, evaluate_candidate, evaluate_population
from .optim.selection import ValidationSelector
from .optim.sep_cmaes import SepCMAES, default_popsize
from .orchestration.dataset import load_tasks, sample_minibatch, split_train_val

_REPO = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _resolve_x0(args, spec) -> np.ndarray:
    """CMA-ES initial mean: a supervised warm-start theta if given, else the zero init.

    ``--warmstart-theta`` (IMPROVEMENTS.md #2) loads a pre-fit head produced by
    ``scripts/warmstart_head.py``. Its length must match ``spec.n_total`` exactly,
    otherwise it is a layout mismatch and we refuse to start (a silent reshape would
    corrupt the head/SVF split).
    """
    from .coordinator import warmstart as WS

    warm = getattr(args, "warmstart_theta", "") or ""
    if not warm:
        return P.initial_theta(spec)
    theta = WS.load_warmstart_theta(warm, spec)  # validates length == spec.n_total
    print(f"[train] warm-start x0 from {warm} (||head||={np.linalg.norm(theta[:spec.n_head]):.3f}, "
          f"deviates from zero-init by {float(np.linalg.norm(theta - P.initial_theta(spec))):.3f})")
    return theta


def build_summary(
    *,
    benchmark: str,
    pool_models: list[str],
    n_total: int,
    popsize: int,
    m_cma: int,
    generations: int,
    best_fitness: float,
    seed: int,
    run_dir: Path | str,
) -> dict:
    """Assemble the ``summary.json`` payload for a finished training run.

    ``seed`` is recorded because it drives every source of randomness in the run
    — task sampling, the sep-CMA-ES trajectory, and the per-generation RNG — and
    a submission receipt cannot name the seed behind its fitness curve without it
    (issue #109).

    Args:
        benchmark: Benchmark the head was trained on.
        pool_models: Coordinated LLM pool, in slot order.
        n_total: Search dimension of the parameter vector.
        popsize: sep-CMA-ES population size λ.
        m_cma: Replications per candidate.
        generations: Generations actually completed.
        best_fitness: Fitness of the best candidate seen.
        seed: The run's ``--seed``.
        run_dir: Directory holding the run artifacts.

    Returns:
        The summary mapping, JSON-serialisable as written to ``summary.json``.
    """
    return {
        "benchmark": benchmark,
        "pool": pool_models,
        "n_total": int(n_total),
        "popsize": int(popsize),
        "m_cma": int(m_cma),
        "generations": int(generations),
        "best_fitness": float(best_fitness),
        "seed": int(seed),
        "run_dir": str(run_dir),
    }


async def train(args) -> dict:
    cfg = _load_yaml(args.config)
    cc = cfg["coordinator"]
    sc = cfg["sep_cmaes"]
    sess = cfg.get("session", {})
    # Training-only fitness shaping (improvement #3). Defaults preserve the
    # original mean-binary fitness exactly. The eval path stays pure binary.
    fitness_cfg = FitnessConfig.from_dict(cfg.get("fitness"))
    if getattr(args, "enable_reweight", False) and not fitness_cfg.enable_reweight:
        import dataclasses
        fitness_cfg = dataclasses.replace(fitness_cfg, enable_reweight=True)
    if fitness_cfg.enable_reweight or fitness_cfg.shaping_active:
        print(f"[train] fitness shaping ACTIVE: {fitness_cfg}")

    pool = OpenRouterPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    print(f"[train] benchmark={args.benchmark}  pool={pool_models}")
    print("[train] building coordinator on GPU (this loads Qwen3-0.6B)...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cc["encoder_model"],
        device=cc.get("device", "cuda:0"),
        dtype=cc.get("dtype", "bfloat16"),
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=n_models,
        n_roles=cc["head"].get("n_roles", 3),
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )
    assert spec.n_svf == int(policy.svf.num_scales), (
        f"spec.n_svf={spec.n_svf} != svf.num_scales={policy.svf.num_scales}"
    )
    print(f"[train] θ dimension n = {spec.n_total} (head {spec.n_head} + SVF {spec.n_svf})")

    popsize = args.popsize or sc.get("population_size") or default_popsize(spec.n_total)
    m_cma = args.m_cma or sc.get("m_cma", 16)
    generations = args.generations or sc.get("generations", 60)
    sigma0 = sc.get("sigma0", 0.1)

    # Atomic-eval budget cap (SPEC §5.2: T = ⌊B_env / (m_cma·λ)⌋). Opt-in via --budget:
    # when set, cap T to what the budget affords and track consumption. Default off
    # (--budget 0) -> generations governed by config, no cap (behaviour unchanged).
    budget_b_env = getattr(args, "budget", 0) or 0
    budget: AtomicEvalBudget | None = None
    if budget_b_env > 0:
        budget = AtomicEvalBudget(b_env=budget_b_env, m_cma=m_cma, popsize=popsize)
        if budget.max_generations < generations:
            print(f"[train] atomic-eval budget {budget_b_env}: capping T {generations} -> "
                  f"{budget.max_generations} (cost/gen = m_cma·λ = {budget.cost_per_generation})")
        generations = min(generations, budget.max_generations)

    # Validation-based model selection (issue #172). Default off (val_fraction=0.0)
    # -> save es.best() and run all generations, exactly as before.
    _val_arg = getattr(args, "val_fraction", None)
    val_fraction = _val_arg if _val_arg is not None else float(sc.get("val_fraction", 0.0))
    _pat_arg = getattr(args, "patience", None)
    patience = _pat_arg if _pat_arg is not None else int(sc.get("patience", 0))

    tasks = load_tasks(args.benchmark, "train", max_items=args.max_items, seed=args.seed)
    print(f"[train] loaded {len(tasks)} train tasks")

    selector: ValidationSelector | None = None
    val_tasks: list = []
    if val_fraction > 0.0:
        # A distinct RNG stream from the per-generation minibatch RNG so the split
        # is reproducible from --seed without correlating with task sampling.
        tasks, val_tasks = split_train_val(tasks, val_fraction, random.Random(args.seed * 100003 + 7))
        selector = ValidationSelector()
        print(f"[train] validation holdout: {len(val_tasks)} val / {len(tasks)} train "
              f"(val_fraction={val_fraction}, patience={patience})")

    x0 = _resolve_x0(args, spec)

    es = SepCMAES(
        n=spec.n_total,
        sigma0=sigma0,
        x0=x0,
        popsize=popsize,
        seed=args.seed,
        maxiter=generations,
    )
    print(f"[train] sep-CMA-ES: λ={es.popsize}, σ0={sigma0}, m_cma={m_cma}, T={generations}, "
          f"budget≈{es.popsize * m_cma * generations}")

    run_dir = _REPO / "experiments" / args.benchmark / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    run_kwargs = dict(
        max_turns=args.max_turns or sess.get("max_turns", 5),
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        verifier_requires_prior_worker=sess.get("verifier_requires_prior_worker", True),
    )

    gen = 0
    while not es.stop() and gen < generations:
        t0 = time.time()
        thetas = es.ask()

        # Common random numbers: ALL candidates in a generation are scored on the
        # SAME minibatch (re-sampled across generations). This removes task-luck from
        # intra-generation ranking — the variance-reduction the noisy binary reward
        # needs for sep-CMA-ES to rank candidates by policy quality, not by which
        # tasks they happened to draw. (Pilot without this showed a flat/bouncing J.)
        gen_rng = random.Random(args.seed * 100000 + gen)
        gen_minibatch = sample_minibatch(tasks, m_cma, gen_rng)

        def minibatch_fn(i, _mb=gen_minibatch):
            return _mb

        def _on_cand(i, fit, elapsed, _g=gen):
            print(f"    [gen {_g} cand {i + 1}/{len(thetas)}] fit={fit:.3f} ({elapsed:.0f}s)",
                  flush=True)

        fits = await evaluate_population(
            thetas, spec, policy, pool, pool_models, minibatch_fn,
            sample=True, on_candidate=_on_cand, fitness_cfg=fitness_cfg,
            run_seed=args.seed, generation=gen, **run_kwargs
        )
        es.tell(thetas, fits)

        best_x, best_f = es.best()

        val_fit: float | None = None
        if selector is not None:
            # Score THIS generation's best candidate on the fixed val set with the
            # eval decision rule (argmax, sample=False) and pure-binary reward
            # (fitness_cfg=None) so val fitness proxies the hidden-eval accuracy.
            gen_best_x = thetas[int(np.argmax(fits))]
            vf, _ = await evaluate_candidate(
                gen_best_x, spec, policy, pool, pool_models, val_tasks,
                sample=False, fitness_cfg=None, **run_kwargs,
            )
            val_fit = float(vf)
            selector.update(gen, gen_best_x, val_fit)

        rec = {
            "generation": gen,
            "gen_mean_fitness": float(np.mean(fits)),
            "gen_max_fitness": float(np.max(fits)),
            "best_fitness": float(best_f),
            "seconds": round(time.time() - t0, 1),
        }
        if val_fit is not None:
            rec["val_fitness"] = val_fit
        history.append(rec)
        line = (f"[gen {gen:3d}] mean={rec['gen_mean_fitness']:.3f} "
                f"max={rec['gen_max_fitness']:.3f} best={rec['best_fitness']:.3f}")
        if val_fit is not None:
            line += f" val={val_fit:.3f}"
        print(line + f" ({rec['seconds']}s)")

        # Save the validation-selected theta when a holdout is active, else es.best().
        save_x = selector.best_theta if (selector is not None and selector.best_theta is not None) else best_x
        np.save(run_dir / "best_theta.npy", save_x)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        gen += 1

        if budget is not None:
            budget.record_generation()
            if budget.exhausted:
                print(f"[train] atomic-eval budget spent: {budget.consumed}/{budget.b_env} "
                      f"({budget.fraction_used:.1%}) after {gen} generation(s)")
                break

        if selector is not None and selector.should_stop(patience):
            print(f"[train] early stop: no val improvement for {patience} generation(s) "
                  f"(best val={selector.best_val_fitness:.3f} @ gen {selector.best_gen})")
            break

    train_best_x, best_f = es.best()
    best_x = (selector.best_theta if (selector is not None and selector.best_theta is not None)
              else train_best_x)
    np.save(run_dir / "best_theta.npy", best_x)
    summary = build_summary(
        benchmark=args.benchmark,
        pool_models=pool_models,
        n_total=spec.n_total,
        popsize=es.popsize,
        m_cma=m_cma,
        generations=gen,
        best_fitness=best_f,
        seed=args.seed,
        run_dir=run_dir,
    )
    if selector is not None:
        # ``best_fitness`` stays the TRAIN best (the fitness-curve max the receipt
        # gate cross-checks); these record which generation the saved theta came
        # from and its held-out score.
        summary["selected_generation"] = selector.best_gen
        summary["val_fitness"] = float(selector.best_val_fitness)
    if budget is not None:
        # Atomic-eval budget accounting for the receipt/audit trail (SPEC §5.2).
        summary["atomic_eval_budget"] = budget.report()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] DONE. best_fitness={best_f:.4f}  -> {run_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve the TRINITY coordinator with sep-CMA-ES")
    ap.add_argument("--benchmark", required=True, help="math500 | mmlu | gpqa | livecodebench")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=256, dest="max_items")
    ap.add_argument("--max-turns", type=int, default=0, dest="max_turns", help="override K")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--generations", type=int, default=0, help="override config T")
    ap.add_argument("--popsize", type=int, default=0, help="override λ")
    ap.add_argument("--m-cma", type=int, default=0, dest="m_cma", help="override replications")
    ap.add_argument("--budget", type=int, default=0, dest="budget",
                    help="atomic-eval (B_env) cap: run T=floor(B_env/(m_cma·λ)) generations "
                         "and stop when spent (SPEC §5.2); 0 (default) disables the cap")
    ap.add_argument("--val-fraction", type=float, default=None, dest="val_fraction",
                    help="held-out validation share for model selection (overrides config; "
                         "0.0 disables, the default)")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop after this many generations with no val improvement "
                         "(overrides config; 0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="run", dest="run_name")
    ap.add_argument("--warmstart-theta", default="", dest="warmstart_theta",
                    help="path to a warm-start theta .npy (scripts/warmstart_head.py); "
                         "used as the sep-CMA-ES initial mean instead of the zero init")
    ap.add_argument("--enable-reweight", action="store_true", dest="enable_reweight",
                    help="turn on variance-aware task reweighting (#3) regardless of config")
    args = ap.parse_args()
    # argparse stores 0 for "not set" on the int overrides; normalize to None-ish.
    args.generations = args.generations or None
    args.popsize = args.popsize or None
    args.m_cma = args.m_cma or None
    args.max_turns = args.max_turns or None
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
