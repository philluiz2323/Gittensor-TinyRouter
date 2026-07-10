"""Per-question model agreement over the cached benchmark answers.

Why this exists
---------------
``docs/JOURNAL.md`` (2026-06-25, "dead gradient") records that GRPO computes its
advantage *within* a question's rollout group, but on math500 the strong workers
solve (or fail) a given question consistently regardless of which one routing
picks. Within-group reward variance is then ~0, every advantage is 0, and the
update skips every sample. The recorded follow-up is to *"train on the
disagreement subset (contested questions)"* — the questions where the choice of
worker actually flips correctness. ``ROADMAP.md`` likewise lists the
"disagreement rate" as a Phase-2 deliverable.

``scripts/oracle_ceiling.py`` already reports the disagreement rate as a single
scalar and computes the debiased ceiling / bootstrap CIs from a solve matrix.
Two things were missing, and this module supplies exactly those:

1. **A zero-cost path to that matrix.** ``oracle_ceiling --collect`` re-queries
   every model over the network. But the built benchmark already stores one
   cached answer per (question, model) in ``item["model_answers"]``. Grading
   those cached answers reproduces the same solve matrix for **$0**.
2. **The identity of the contested questions.** A scalar rate cannot be used to
   build a training pool; :func:`contested_ids` returns the question ids where
   the models disagree, which is the subset the JOURNAL asks us to train on.

:func:`to_oracle_matrix` emits precisely the schema
``scripts/oracle_ceiling.py --analyze`` consumes, so the full statistical layer
(cross-fit oracle, bootstrap CIs, verdict) is reused rather than duplicated
here. This module deliberately computes only the cheap, selection-relevant
counts it needs; it is not a second statistics implementation.

Pure / deterministic / no network / no GPU / no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "QuestionAgreement",
    "AgreementSummary",
    "grade_item",
    "grade_items",
    "contested_ids",
    "summarize",
    "to_oracle_matrix",
]

# A scorer takes (benchmark, candidate_answer, reference) and returns 1.0 / 0.0.
ScoreFn = Callable[[str, str, Any], float]

_DEFAULT_BENCHMARK = "math500"


def _adapter_score(benchmark: str, candidate: str, reference: Any) -> float:
    """Grade one cached answer through the benchmark's adapter.

    Lazily imported: the adapter registry pulls in optional dataset loaders, and
    this module must stay importable on a box without them.
    """
    from trinity.adapters import get_adapter

    return float(get_adapter(benchmark).score_output(candidate, reference))


@dataclass(frozen=True)
class QuestionAgreement:
    """How the model pool did on ONE question.

    Attributes:
        question_id: The protocol ``question_id`` of the item.
        benchmark: Benchmark this question belongs to.
        per_model_correct: Model name -> ``1`` if its cached answer was graded
            correct, else ``0``. Every question in a run must carry the same
            model keys.
    """

    question_id: str
    benchmark: str
    per_model_correct: dict[str, int]

    @property
    def models(self) -> list[str]:
        """Model names, in a stable sorted order."""
        return sorted(self.per_model_correct)

    @property
    def n_models(self) -> int:
        """How many models answered this question."""
        return len(self.per_model_correct)

    @property
    def n_correct(self) -> int:
        """How many models got this question right."""
        return sum(self.per_model_correct.values())

    @property
    def is_contested(self) -> bool:
        """True iff at least one model solved it and at least one did not.

        These are the only questions on which the routing choice can change the
        reward, so they are the ones that carry a GRPO gradient.
        """
        return 0 < self.n_correct < self.n_models

    @property
    def is_unanimous_correct(self) -> bool:
        """True iff every model solved it (no routing signal: reward is always 1)."""
        return self.n_models > 0 and self.n_correct == self.n_models

    @property
    def is_unanimous_wrong(self) -> bool:
        """True iff no model solved it (no routing signal: reward is always 0)."""
        return self.n_models > 0 and self.n_correct == 0


@dataclass(frozen=True)
class AgreementSummary:
    """Aggregate agreement counts over a graded benchmark.

    ``oracle_any`` is the fraction of questions at least one model solves — the
    naive routing ceiling. ``headroom`` is how much of that a perfect router could
    add over always picking the single best model. Both are the raw, undebiased
    quantities; the winner's-curse-corrected versions live in
    ``scripts/oracle_ceiling.py``, which this module feeds via
    :func:`to_oracle_matrix`.
    """

    n_questions: int
    models: list[str]
    n_contested: int
    n_unanimous_correct: int
    n_unanimous_wrong: int
    disagreement_rate: float
    per_model_accuracy: dict[str, float]
    best_single_model: str | None
    best_single_accuracy: float
    oracle_any: float
    headroom: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, for CLI output and report files."""
        return {
            "n_questions": self.n_questions,
            "models": list(self.models),
            "n_contested": self.n_contested,
            "n_unanimous_correct": self.n_unanimous_correct,
            "n_unanimous_wrong": self.n_unanimous_wrong,
            "disagreement_rate": self.disagreement_rate,
            "per_model_accuracy": dict(self.per_model_accuracy),
            "best_single_model": self.best_single_model,
            "best_single_accuracy": self.best_single_accuracy,
            "oracle_any": self.oracle_any,
            "headroom": self.headroom,
        }


def grade_item(
    item: Mapping[str, Any],
    *,
    score_fn: ScoreFn | None = None,
    use_cached_scores: bool = True,
) -> QuestionAgreement:
    """Grade one on-disk benchmark item into a :class:`QuestionAgreement`.

    The item is the protocol shape written by ``scripts/build_benchmark.py``:
    ``question_id`` / ``benchmark`` / ``correct_answer`` / ``model_answers``, and
    optionally a pre-computed ``model_scores`` map.

    Args:
        item: One benchmark item.
        score_fn: Grader taking ``(benchmark, candidate, reference)``. Defaults to
            the benchmark adapter's ``score_output``.
        use_cached_scores: When the item already carries ``model_scores`` for a
            model, trust it instead of re-grading that answer.

    Returns:
        The per-model correctness record for this question.

    Raises:
        KeyError: If the item has no ``correct_answer``.
    """
    scorer = score_fn or _adapter_score
    benchmark = str(item.get("benchmark") or _DEFAULT_BENCHMARK)
    reference = item["correct_answer"]
    answers: Mapping[str, Any] = item.get("model_answers") or {}
    cached: Mapping[str, Any] = (item.get("model_scores") or {}) if use_cached_scores else {}

    per_model: dict[str, int] = {}
    for model, answer in answers.items():
        if model in cached:
            per_model[model] = int(float(cached[model]) >= 0.5)
            continue
        text = "" if answer is None else str(answer)
        per_model[model] = int(scorer(benchmark, text, reference) >= 0.5)

    qid = str(item.get("question_id", ""))
    return QuestionAgreement(question_id=qid, benchmark=benchmark, per_model_correct=per_model)


def grade_items(
    items: Iterable[Mapping[str, Any]],
    *,
    score_fn: ScoreFn | None = None,
    use_cached_scores: bool = True,
) -> list[QuestionAgreement]:
    """Grade every item, skipping those with no cached answers.

    An item with an empty ``model_answers`` carries no information about the pool
    (typically a ``live`` split item, which is intentionally left uncached), so it
    is dropped rather than counted as a unanimous failure.

    Raises:
        ValueError: If two graded questions disagree about the model set. A ragged
            pool would silently bias every rate below, so it is an error.
    """
    records: list[QuestionAgreement] = []
    expected: list[str] | None = None
    for item in items:
        if not (item.get("model_answers") or {}):
            continue
        rec = grade_item(item, score_fn=score_fn, use_cached_scores=use_cached_scores)
        if expected is None:
            expected = rec.models
        elif rec.models != expected:
            raise ValueError(
                f"question {rec.question_id!r} has models {rec.models}, expected {expected}"
            )
        records.append(rec)
    return records


def contested_ids(records: Sequence[QuestionAgreement]) -> list[str]:
    """Question ids where the pool disagrees (some model right, some wrong).

    This is the "disagreement subset" the JOURNAL prescribes for GRPO: on every
    other question the reward is constant across routing choices, so the
    within-group advantage is identically zero and the sample contributes no
    gradient.
    """
    return [r.question_id for r in records if r.is_contested]


def summarize(records: Sequence[QuestionAgreement]) -> AgreementSummary:
    """Aggregate graded questions into pool-level agreement counts.

    An empty input yields an all-zero summary rather than raising, so a caller can
    report "nothing cached yet" without special-casing.
    """
    n = len(records)
    if n == 0:
        return AgreementSummary(
            n_questions=0, models=[], n_contested=0, n_unanimous_correct=0,
            n_unanimous_wrong=0, disagreement_rate=0.0, per_model_accuracy={},
            best_single_model=None, best_single_accuracy=0.0, oracle_any=0.0, headroom=0.0,
        )

    models = records[0].models
    n_contested = sum(r.is_contested for r in records)
    n_unanimous_correct = sum(r.is_unanimous_correct for r in records)
    n_unanimous_wrong = sum(r.is_unanimous_wrong for r in records)

    per_model_accuracy = {
        m: sum(r.per_model_correct[m] for r in records) / n for m in models
    }
    best_model: str | None = None
    best_acc = 0.0
    for m in models:  # sorted, so ties resolve deterministically
        if per_model_accuracy[m] > best_acc:
            best_model, best_acc = m, per_model_accuracy[m]
    if best_model is None and models:
        best_model = models[0]  # every model scored 0.0

    oracle_any = sum(r.n_correct > 0 for r in records) / n
    return AgreementSummary(
        n_questions=n,
        models=list(models),
        n_contested=n_contested,
        n_unanimous_correct=n_unanimous_correct,
        n_unanimous_wrong=n_unanimous_wrong,
        disagreement_rate=n_contested / n,
        per_model_accuracy=per_model_accuracy,
        best_single_model=best_model,
        best_single_accuracy=best_acc,
        oracle_any=oracle_any,
        headroom=max(0.0, oracle_any - best_acc),
    )


def to_oracle_matrix(
    records: Sequence[QuestionAgreement],
    *,
    benchmark: str | None = None,
) -> dict[str, Any]:
    """Emit the matrix JSON that ``oracle_ceiling.py --analyze`` consumes.

    Each cached answer is a single deterministic sample, so every ``per_model``
    cell is a length-1 list (``K = 1``). That matches
    ``oracle_ceiling.matrix_to_tensor``, which requires a uniform ``K`` across all
    (question, model) cells.

    Args:
        records: Graded questions, all sharing one model set.
        benchmark: Name to stamp on the matrix. Defaults to the first record's.

    Returns:
        ``{"benchmark": str, "n_samples": 1, "tasks": [{"id", "per_model"}, ...]}``
    """
    bench = benchmark or (records[0].benchmark if records else _DEFAULT_BENCHMARK)
    tasks = [
        {
            "id": r.question_id,
            "per_model": {m: [r.per_model_correct[m]] for m in r.models},
        }
        for r in records
    ]
    return {"benchmark": bench, "n_samples": 1, "tasks": tasks}
