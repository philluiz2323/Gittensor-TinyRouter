"""Offline multi-agent ensemble (plurality / self-consistency) baseline — SPEC R3.

SPEC §1.3 **R3** requires TRINITY to beat the best MULTI-AGENT baseline, but the eval only
compares against single models, random routing, and the (unrealizable) oracle upper
bound. R1/R2/R4 have ``significance.py``, R8 has ``convergence.py``, R12 has
``efficiency.py`` — **R3 alone has no offline tool**, and RESULTS.md's system tables never
include an ensemble row.

This adds the realizable pool-ensemble baseline: a per-question **plurality vote** over the
pool's cached answers, clustered by answer-equivalence via the FIXED grader and scored by
the same scorer. It is the offline-computable member of the multi-agent-baseline family
(not a literal MoA / Smoothie / RouterDC / MasRouter — those need an aggregator LLM or a
trained router, i.e. online) — giving **R3 its first offline verdict**.

Reuses ``item["model_answers"]`` (cached per-model answers) + reward's extractors and
``score_text`` — the exact equivalence/grading the eval uses. Pure: no torch, no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = ["EnsembleSummary", "answers_agree", "plurality_answer", "analyze", "render"]

#: A grader: ``(benchmark, candidate, reference) -> 1.0/0.0`` (defaults to reward.score_text).
ScoreFn = Callable[[str, str, Any], float]


def answers_agree(benchmark: str, a: str, b: str) -> bool:
    """Whether two answer strings are equivalent under the benchmark's grader.

    Reuses the eval scorer's own extractors (choice letter / boxed-or-numeric math value
    via ``math_equal`` / extracted code / else stripped text) — so "agree" means exactly
    what "correct" means. Mirrors the merged HERO self-consistency predicate (#139).
    """
    from trinity.orchestration import reward as _reward

    key = _reward.resolve_benchmark(benchmark)
    if key in _reward.CHOICE_BENCHMARKS:
        la = _reward.extract_choice_letter(a)
        return la is not None and la == _reward.extract_choice_letter(b)
    if key in _reward.MATH_BENCHMARKS:
        na = _reward.extract_boxed(a) or _reward.extract_last_number(a)
        nb = _reward.extract_boxed(b) or _reward.extract_last_number(b)
        return na is not None and nb is not None and _reward.math_equal(na, nb)
    if key in _reward.CODE_BENCHMARKS:
        ca, cb = _reward.extract_code(a).strip(), _reward.extract_code(b).strip()
        return bool(ca) and ca == cb
    return bool(a.strip()) and a.strip() == b.strip()


def plurality_answer(benchmark: str, model_answers: Mapping[str, Any]) -> str | None:
    """Plurality-vote representative answer over the pool's cached answers.

    Clusters the models' (non-empty) answers by :func:`answers_agree` and returns a
    representative of the largest cluster. Ties break deterministically toward the cluster
    whose earliest model sorts first, so the vote is reproducible. Returns None if no
    model produced a non-empty answer.
    """
    entries = [(m, "" if model_answers[m] is None else str(model_answers[m]))
               for m in sorted(model_answers)]
    entries = [(m, a) for m, a in entries if a.strip()]
    if not entries:
        return None
    clusters: list[list[tuple[str, str]]] = []
    for m, a in entries:
        for c in clusters:
            if answers_agree(benchmark, c[0][1], a):
                c.append((m, a))
                break
        else:
            clusters.append([(m, a)])
    # Largest cluster; on ties the earliest-created cluster wins (entries are in sorted-
    # model order and clusters preserve first-seen order, so smaller index = earlier model).
    best = max(range(len(clusters)), key=lambda i: (len(clusters[i]), -i))
    return clusters[best][0][1]


@dataclass(frozen=True)
class EnsembleSummary:
    """The realizable pool-ensemble (plurality-vote) baseline over one benchmark."""

    benchmark: str
    n_questions: int
    models: list[str]
    ensemble_accuracy: float
    per_model_accuracy: dict[str, float]
    best_single_model: str | None
    best_single: float
    oracle_any: float
    ensemble_vs_best_single: float
    per_item: list[int]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_questions": self.n_questions,
            "models": list(self.models),
            "ensemble_accuracy": self.ensemble_accuracy,
            "per_model_accuracy": dict(self.per_model_accuracy),
            "best_single_model": self.best_single_model,
            "best_single": self.best_single,
            "oracle_any": self.oracle_any,
            "ensemble_vs_best_single": self.ensemble_vs_best_single,
            "per_item": list(self.per_item),
        }


def analyze(
    items: Sequence[Mapping[str, Any]],
    *,
    benchmark: str | None = None,
    score_fn: ScoreFn | None = None,
) -> EnsembleSummary:
    """Compute the plurality-vote ensemble baseline over cached-answer ``items``.

    Each item is a benchmark item with ``model_answers`` (name -> answer) + a
    ``correct_answer`` reference (the shape ``analysis.agreement`` consumes). For each
    item the plurality vote is graded, and per-model / oracle accuracies are computed for
    context. ``score_fn`` defaults to ``reward.score_text``; inject one for offline tests.

    Raises:
        ValueError: If two items disagree about the model pool. A ragged pool would grade
            each model's accuracy over its own item count while the oracle spans every
            item, so a model present in only some items could report
            ``best_single > oracle_any`` (an impossible invariant). As in
            ``analysis.agreement``/``complementarity``/``sampling``, a ragged pool is
            rejected rather than silently biased.
    """
    from trinity.orchestration import reward as _reward

    score: ScoreFn = score_fn or _reward.score_text
    per_model_correct: dict[str, list[int]] = {}
    per_item: list[int] = []
    oracle: list[int] = []
    resolved = benchmark or "?"
    n = 0
    expected_models: list[str] | None = None
    for item in items:
        answers = item.get("model_answers") or {}
        if not answers:
            continue
        current_models = sorted(answers)
        if expected_models is None:
            expected_models = current_models
        elif current_models != expected_models:
            raise ValueError(
                f"item {item.get('question_id', '?')!r} has models {current_models}, "
                f"expected {expected_models}; a ragged pool grades per-model and oracle "
                "accuracy over different denominators (best_single could exceed the oracle)"
            )
        bench = benchmark or str(item.get("benchmark") or "")
        resolved = bench
        ref = item.get("correct_answer")
        n += 1
        model_ok: list[int] = []
        for m, a in answers.items():
            ok = int(score(bench, "" if a is None else str(a), ref) >= 0.5)
            per_model_correct.setdefault(m, []).append(ok)
            model_ok.append(ok)
        oracle.append(1 if any(model_ok) else 0)
        rep = plurality_answer(bench, answers)
        per_item.append(int(score(bench, rep, ref) >= 0.5) if rep is not None else 0)

    models = sorted(per_model_correct)
    if n == 0 or not models:
        return EnsembleSummary(resolved, 0, [], 0.0, {}, None, 0.0, 0.0, 0.0, [])
    per_model_acc = {m: sum(v) / len(v) for m, v in ((m, per_model_correct[m]) for m in models)}
    best_model = max(models, key=lambda m: per_model_acc[m])
    best_single = per_model_acc[best_model]
    ensemble_acc = sum(per_item) / len(per_item)
    return EnsembleSummary(
        benchmark=resolved,
        n_questions=n,
        models=list(models),
        ensemble_accuracy=ensemble_acc,
        per_model_accuracy=per_model_acc,
        best_single_model=best_model,
        best_single=best_single,
        oracle_any=sum(oracle) / len(oracle),
        ensemble_vs_best_single=ensemble_acc - best_single,
        per_item=per_item,
    )


def render(summary: EnsembleSummary, *, trinity_accuracy: float | None = None) -> str:
    """Markdown: the ensemble vs each single vs oracle, and the R3 verdict if TRINITY given."""
    out = ["# Multi-agent ensemble baseline (SPEC R3)\n"]
    if summary.n_questions == 0:
        return "".join(out) + "\n_(no cached-answer items found)_\n"

    out.append(f"n = {summary.n_questions} questions, pool = {summary.models}\n")
    out.append("| system | accuracy |")
    out.append("|---|---|")
    out.append(f"| **ensemble (plurality vote)** | **{summary.ensemble_accuracy:.3f}** |")
    for m in summary.models:
        star = " ← best" if m == summary.best_single_model else ""
        out.append(f"| single: {m}{star} | {summary.per_model_accuracy[m]:.3f} |")
    out.append(f"| oracle (any model) | {summary.oracle_any:.3f} |")
    if trinity_accuracy is not None:
        out.append(f"| TRINITY | {trinity_accuracy:.3f} |")

    out.append(f"\n- ensemble − best single = {summary.ensemble_vs_best_single:+.3f}")
    if trinity_accuracy is not None:
        holds = trinity_accuracy > summary.ensemble_accuracy
        out.append(f"- **R3** (TRINITY > best multi-agent baseline): "
                   f"{'✅ HOLDS' if holds else '❌'} "
                   f"({trinity_accuracy:.3f} vs ensemble {summary.ensemble_accuracy:.3f})")
    else:
        out.append("- **R3**: TRINITY must beat this ensemble baseline "
                   "(supply a TRINITY accuracy to render the verdict).")
    return "\n".join(out) + "\n"
