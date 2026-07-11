"""Offline format audit: which models produce parseable answers, per benchmark.

Why this exists
---------------
``docs/JOURNAL.md`` (2026-06-25) diagnoses the training bottleneck as **format,
not routing**: "*when the policy emits a valid workflow, the routed worker solves
it essentially every time*", but most proposals never parse. The same failure
mode applies to the worker pool — a worker whose answer cannot be extracted
scores 0 no matter how good the reasoning was.

Correctness and *parse rate* are different things, and the pipeline only reports
correctness. A worker can be right but unparseable (a false negative the grader
cannot rescue), and that is invisible in an accuracy number. This module reports
the parse rate — the fraction of cached answers from which
:func:`trinity.orchestration.reward.has_answer` can extract an answer at all —
broken down per model and per benchmark, so a contributor can see *which* model
is losing points to format rather than to reasoning.

It reads the ``model_answers`` already cached in a built benchmark, so it costs
nothing. ``has_answer`` is the exact predicate the grader and the multi-turn
committed-answer selection use, so "parseable here" means "scorable there".

Pure / deterministic / no network / no GPU / no torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "ModelFormatStats",
    "FormatAudit",
    "audit_items",
]

# Predicate: (benchmark, text) -> is an answer extractable? Defaults to the grader's.
HasAnswerFn = Callable[[str, str], bool]

_DEFAULT_BENCHMARK = "math500"


def _default_has_answer(benchmark: str, text: str) -> bool:
    """The grader's own format-validity predicate (imported lazily)."""
    from trinity.orchestration.reward import has_answer

    return bool(has_answer(benchmark, text))


@dataclass
class ModelFormatStats:
    """Parse-rate tally for one model (optionally within one benchmark)."""

    model: str
    n_answers: int = 0
    n_parseable: int = 0
    n_empty: int = 0            # blank / missing completions (a subset of unparseable)

    @property
    def n_unparseable(self) -> int:
        """Answers present but with no extractable value."""
        return self.n_answers - self.n_parseable

    @property
    def parse_rate(self) -> float:
        """Fraction of this model's answers that are extractable (0 when none)."""
        return self.n_parseable / self.n_answers if self.n_answers else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "n_answers": self.n_answers,
            "n_parseable": self.n_parseable,
            "n_unparseable": self.n_unparseable,
            "n_empty": self.n_empty,
            "parse_rate": self.parse_rate,
        }


@dataclass
class FormatAudit:
    """Parse-rate audit over a set of graded benchmark items.

    ``per_model`` aggregates across every benchmark seen; ``per_benchmark_model``
    keeps the ``benchmark -> model -> stats`` split so a model that is fine on
    math but mangles MCQ letters is visible.
    """

    per_model: dict[str, ModelFormatStats] = field(default_factory=dict)
    per_benchmark_model: dict[str, dict[str, ModelFormatStats]] = field(default_factory=dict)
    n_items: int = 0

    @property
    def overall_parse_rate(self) -> float:
        """Parse rate pooled across every (item, model) answer seen."""
        total = sum(s.n_answers for s in self.per_model.values())
        ok = sum(s.n_parseable for s in self.per_model.values())
        return ok / total if total else 0.0

    def worst_model(self) -> str | None:
        """The model with the lowest parse rate (ties broken by name).

        ``None`` when nothing was audited. This is the model to look at first: it
        is losing the most answers to format rather than to reasoning.
        """
        scored = [s for s in self.per_model.values() if s.n_answers]
        if not scored:
            return None
        return min(scored, key=lambda s: (s.parse_rate, s.model)).model

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_items": self.n_items,
            "overall_parse_rate": self.overall_parse_rate,
            "worst_model": self.worst_model(),
            "per_model": {m: s.to_dict() for m, s in sorted(self.per_model.items())},
            "per_benchmark_model": {
                b: {m: s.to_dict() for m, s in sorted(ms.items())}
                for b, ms in sorted(self.per_benchmark_model.items())
            },
        }


def audit_items(
    items: Iterable[Mapping[str, Any]],
    *,
    has_answer_fn: HasAnswerFn | None = None,
) -> FormatAudit:
    """Audit the parse rate of the cached ``model_answers`` in built items.

    Each item is the protocol shape from ``scripts/build_benchmark.py``:
    ``benchmark`` + ``model_answers`` (``model -> answer text``). An item with no
    cached answers (e.g. a ``live``-split item) contributes nothing.

    Args:
        items: Built benchmark items.
        has_answer_fn: Format predicate; defaults to the grader's ``has_answer``.

    Returns:
        A :class:`FormatAudit`.
    """
    check = has_answer_fn or _default_has_answer
    audit = FormatAudit()

    def _stat(store: dict[str, ModelFormatStats], model: str) -> ModelFormatStats:
        if model not in store:
            store[model] = ModelFormatStats(model=model)
        return store[model]

    for item in items:
        answers: Mapping[str, Any] = item.get("model_answers") or {}
        if not answers:
            continue
        audit.n_items += 1
        benchmark = str(item.get("benchmark") or _DEFAULT_BENCHMARK)
        bench_store = audit.per_benchmark_model.setdefault(benchmark, {})

        for model, answer in answers.items():
            text = "" if answer is None else str(answer)
            parseable = bool(text.strip()) and check(benchmark, text)
            for store in (audit.per_model, bench_store):
                s = _stat(store, model)
                s.n_answers += 1
                if parseable:
                    s.n_parseable += 1
                if not text.strip():
                    s.n_empty += 1

    return audit
