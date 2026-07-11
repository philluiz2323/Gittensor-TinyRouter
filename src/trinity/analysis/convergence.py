"""Offline training-convergence + optimizer-comparison report over run artifacts.

Every sep-CMA-ES / baseline training run writes ``history.json`` (per-iteration
``J(θ)``) + ``summary.json``, but nothing consumes them to check the SPEC
definition-of-done — *"the optimizer drives ``J(θ)`` upward over iterations"* — or to
rank optimizers for **R8** (``sep-CMA-ES > SFT > RS > REINFORCE``). ``sep_cmaes.run``'s
own docstring says its history is *"suitable for logging J(θ) over training"* — produced,
never analyzed.

This reads those on-disk artifacts and reports, per run, the convergence diagnostics
(net gain, monotonicity of the best-so-far, generations-to-best, trend slope, tail
plateau, and a **degenerate-run flag** for the noisy-reward failure RESULTS.md records —
"sep-CMA-ES occasionally converged to a bad policy"), then across runs a per-benchmark
optimizer ranking + the observed R8 order + the DoD verdict.

Pure numpy over JSON — no torch, no network, no GPU. Handles both the sep-CMA-ES history
schema (``generation`` / ``gen_mean_fitness`` / ``best_fitness``) and the Random-Search
baseline schema (``trial`` / ``fitness`` / ``best_fitness``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeGuard

import numpy as np

__all__ = ["RunConvergence", "analyze_run", "analyze_runs", "render"]

_TOL = 1e-6


def _is_num(x: Any) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _curves(history: Sequence[Any]) -> tuple[list[float], list[float]]:
    """Extract (best-so-far curve, per-iteration signal curve) from a history list.

    ``best`` is the ``best_fitness`` series (present in both schemas). ``signal`` is the
    per-iteration mean the population actually explored — ``gen_mean_fitness`` for
    sep-CMA-ES, ``fitness`` for the RS baseline.
    """
    best: list[float] = []
    signal: list[float] = []
    for h in history:
        if not isinstance(h, Mapping):
            continue
        if _is_num(h.get("best_fitness")):
            best.append(float(h["best_fitness"]))
        s = h.get("gen_mean_fitness")
        if s is None:
            s = h.get("fitness")
        if _is_num(s):
            signal.append(float(s))
    return best, signal


def _overfit_gap(summary: Mapping[str, Any]) -> float | None:
    """Train-vs-validation gap, when a validation holdout was used (#173)."""
    v, b = summary.get("val_fitness"), summary.get("best_fitness")
    return float(b) - float(v) if _is_num(v) and _is_num(b) else None


@dataclass(frozen=True)
class RunConvergence:
    """Convergence diagnostics for one training run's fitness curve."""

    run_id: str
    benchmark: str
    trainer: str
    n_iters: int
    initial: float
    final: float
    peak: float
    net_gain: float
    improved: bool
    best_monotone: bool
    iters_to_best: int
    trend_slope: float
    tail_plateau: int
    signal_drop: float
    degenerate: bool
    overfit_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "trainer": self.trainer,
            "n_iters": self.n_iters,
            "initial": self.initial,
            "final": self.final,
            "peak": self.peak,
            "net_gain": self.net_gain,
            "improved": self.improved,
            "best_monotone": self.best_monotone,
            "iters_to_best": self.iters_to_best,
            "trend_slope": self.trend_slope,
            "tail_plateau": self.tail_plateau,
            "signal_drop": self.signal_drop,
            "degenerate": self.degenerate,
            "overfit_gap": self.overfit_gap,
        }


def analyze_run(
    summary: Mapping[str, Any],
    history: Sequence[Any],
    *,
    run_id: str | None = None,
    tol: float = _TOL,
) -> RunConvergence:
    """Compute convergence diagnostics from one run's ``summary`` + ``history``.

    ``trainer`` is read from ``summary["trainer"]`` (RS sets ``"random_search"``);
    sep-CMA-ES runs omit it, defaulting to ``"sep_cmaes"``. An empty/curve-less history
    yields a degenerate zero record rather than raising.
    """
    trainer = str(summary.get("trainer", "sep_cmaes"))
    benchmark = str(summary.get("benchmark", "?"))
    rid = run_id or str(summary.get("run_dir", "")).rstrip("/").rsplit("/", 1)[-1] or benchmark

    best, signal = _curves(history)
    n = len(best)
    if n == 0:
        return RunConvergence(rid, benchmark, trainer, 0, 0.0, 0.0, 0.0, 0.0,
                              False, True, 0, 0.0, 0, 0.0, True, _overfit_gap(summary))

    initial, final, peak = best[0], best[-1], max(best)
    net_gain = final - initial
    best_monotone = all(best[i] <= best[i + 1] + tol for i in range(n - 1))
    iters_to_best = int(np.argmax(best))
    trend_slope = float(np.polyfit(np.arange(n), best, 1)[0]) if n >= 2 else 0.0

    tail_plateau = 0
    for v in reversed(best):
        if abs(v - final) <= tol:
            tail_plateau += 1
        else:
            break

    signal_drop = (max(signal) - signal[-1]) if signal else 0.0
    improved = net_gain > tol
    return RunConvergence(
        run_id=rid, benchmark=benchmark, trainer=trainer, n_iters=n,
        initial=initial, final=final, peak=peak, net_gain=net_gain,
        improved=improved, best_monotone=best_monotone, iters_to_best=iters_to_best,
        trend_slope=trend_slope, tail_plateau=tail_plateau, signal_drop=signal_drop,
        degenerate=not improved, overfit_gap=_overfit_gap(summary),
    )


#: The SPEC R8 optimizer ordering (docs/SPEC.md §1.3), strongest first.
R8_EXPECTED_ORDER = ("sep_cmaes", "sft", "random_search", "reinforce")


def analyze_runs(runs: Sequence[RunConvergence]) -> dict[str, Any]:
    """Cross-run report: per-benchmark optimizer ranking + observed R8 order + DoD.

    Returns a dict with ``rankings`` (per benchmark, optimizers by final fitness),
    ``observed_optimizer_order`` (trainers by mean final fitness across benchmarks),
    ``r8_expected_order``, ``dod_drives_J_upward`` (do the sep-CMA-ES runs all improve
    over their start), and ``degenerate_runs``.
    """
    by_bench: dict[str, list[RunConvergence]] = {}
    by_trainer: dict[str, list[float]] = {}
    for r in runs:
        by_bench.setdefault(r.benchmark, []).append(r)
        by_trainer.setdefault(r.trainer, []).append(r.final)

    rankings = {
        bench: [
            {"trainer": r.trainer, "final": r.final, "run_id": r.run_id,
             "improved": r.improved, "degenerate": r.degenerate}
            for r in sorted(rs, key=lambda x: x.final, reverse=True)
        ]
        for bench, rs in sorted(by_bench.items())
    }
    observed_order = sorted(
        by_trainer, key=lambda t: sum(by_trainer[t]) / len(by_trainer[t]), reverse=True
    )
    dod_runs = [r for r in runs if r.trainer == "sep_cmaes"]
    return {
        "rankings": rankings,
        "observed_optimizer_order": observed_order,
        "r8_expected_order": list(R8_EXPECTED_ORDER),
        "dod_drives_J_upward": bool(dod_runs) and all(r.improved for r in dod_runs),
        "degenerate_runs": [r.run_id for r in runs if r.degenerate],
    }


def render(runs: Sequence[RunConvergence]) -> str:
    """Markdown report: a per-run diagnostics table + the cross-run verdicts."""
    out = ["# Training convergence & optimizer comparison\n"]
    if not runs:
        return "".join(out) + "\n_(no training runs found)_\n"

    out.append("| run | benchmark | trainer | iters | initial→final | net gain | slope | "
               "converged@ | flag |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda x: (x.benchmark, -x.final)):
        flag = "⚠ degenerate" if r.degenerate else ("✅" if r.improved else "≈")
        conv = f"{r.iters_to_best + 1}/{r.n_iters}"
        out.append(
            f"| {r.run_id} | {r.benchmark} | {r.trainer} | {r.n_iters} | "
            f"{r.initial:.3f}→{r.final:.3f} | {r.net_gain:+.3f} | {r.trend_slope:+.4f} | "
            f"{conv} | {flag} |"
        )

    cross = analyze_runs(runs)
    out.append("\n## R8 optimizer comparison\n")
    for bench, ranked in cross["rankings"].items():
        order = " > ".join(f"{e['trainer']}({e['final']:.3f})" for e in ranked)
        out.append(f"- **{bench}**: {order}")
    out.append(f"\n- observed optimizer order (mean final): "
               f"{' > '.join(cross['observed_optimizer_order'])}")
    out.append(f"- R8 expected order: {' > '.join(cross['r8_expected_order'])}")
    dod = cross["dod_drives_J_upward"]
    out.append(f"\n**Definition of done** (sep-CMA-ES drives J(θ) upward): "
               f"{'✅ HOLDS' if dod else '❌ / N/A'}")
    if cross["degenerate_runs"]:
        out.append(f"**⚠ degenerate runs** (no improvement over start): "
                   f"{', '.join(cross['degenerate_runs'])}")
    return "\n".join(out) + "\n"
