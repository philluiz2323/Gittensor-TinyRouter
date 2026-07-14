"""BIG-Bench Hard (BBH) reasoning adapter — multiple-choice + exact-match.

BBH is a suite of 27 challenging multi-step reasoning tasks (logical deduction,
causal judgement, multistep arithmetic, tracking shuffled objects, word sorting,
boolean expressions, ...). Each subtask is graded in one of two formats:

* **multiple_choice** — the gold target is a parenthesised option letter such as
  ``"(B)"`` (subtasks go up to ``(G)``); the candidate is graded by extracting its
  chosen letter and comparing.
* **exact_match** — the gold target is a short free-form string (a number, a word,
  a boolean like ``"True"``/``"Yes"``/``"valid"``, or a space-separated list); the
  candidate is graded by normalized string equality of its final answer.

Scoring is **pure text** — there is no code execution and no sandbox, so this
adapter reuses the shared choice extractor (:func:`reward.extract_choice_letter`)
and adds no untrusted-execution surface. Because BBH mixes the two formats across
subtasks, it cannot be dispatched through a single ``reward.score_text`` family key;
the adapter carries the per-task ``answer_type`` in its structured reference and
grades with :func:`score_bbh` directly (issue #269).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from trinity.types import Task

from .base import BenchmarkAdapter, TaskType
from .registry import register_adapter

__all__ = [
    "BENCHMARK",
    "SUBTASKS",
    "BBHReference",
    "build_bbh_prompt",
    "score_bbh",
    "load_bbh_tasks",
    "BBHAdapter",
    "register_bbh_adapter",
]

#: Canonical benchmark name this adapter registers under.
BENCHMARK = "bbh"

#: HuggingFace dataset id (per-subtask config, ``test`` split).
_HF_DATASET = "lukaemon/bbh"

#: The 27 BBH subtasks and their answer format. ``multiple_choice`` subtasks have a
#: parenthesised option-letter gold target; ``exact_match`` subtasks have a short
#: free-form gold target.
SUBTASKS: dict[str, str] = {
    "boolean_expressions": "exact_match",
    "causal_judgement": "exact_match",
    "date_understanding": "multiple_choice",
    "disambiguation_qa": "multiple_choice",
    "dyck_languages": "exact_match",
    "formal_fallacies": "exact_match",
    "geometric_shapes": "multiple_choice",
    "hyperbaton": "multiple_choice",
    "logical_deduction_three_objects": "multiple_choice",
    "logical_deduction_five_objects": "multiple_choice",
    "logical_deduction_seven_objects": "multiple_choice",
    "movie_recommendation": "multiple_choice",
    "multistep_arithmetic_two": "exact_match",
    "navigate": "exact_match",
    "object_counting": "exact_match",
    "penguins_in_a_table": "multiple_choice",
    "reasoning_about_colored_objects": "multiple_choice",
    "ruin_names": "multiple_choice",
    "salient_translation_error_detection": "multiple_choice",
    "snarks": "multiple_choice",
    "sports_understanding": "exact_match",
    "temporal_sequences": "multiple_choice",
    "tracking_shuffled_objects_three_objects": "multiple_choice",
    "tracking_shuffled_objects_five_objects": "multiple_choice",
    "tracking_shuffled_objects_seven_objects": "multiple_choice",
    "web_of_lies": "exact_match",
    "word_sorting": "exact_match",
}

_MULTIPLE_CHOICE = "multiple_choice"
_EXACT_MATCH = "exact_match"


# --------------------------------------------------------------------------- #
# Structured reference schema
# --------------------------------------------------------------------------- #
@dataclass
class BBHReference:
    """The structured ``reference`` for a BBH task (stored as ``Task.answer``)."""

    answer: str
    answer_type: str = _EXACT_MATCH
    subtask: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (the on-disk / hidden-benchmark form)."""
        return {"answer": self.answer, "answer_type": self.answer_type, "subtask": self.subtask}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBHReference":
        """Rebuild a :class:`BBHReference` from :meth:`to_dict` output."""
        return cls(
            answer=str(data.get("answer", "")),
            answer_type=str(data.get("answer_type", _EXACT_MATCH)),
            subtask=str(data.get("subtask", "")),
        )

    def is_valid(self) -> bool:
        """A reference is gradable iff it carries a gold answer of a known type."""
        return bool(self.answer.strip()) and self.answer_type in (_MULTIPLE_CHOICE, _EXACT_MATCH)


# --------------------------------------------------------------------------- #
# Prompt format
# --------------------------------------------------------------------------- #
def build_bbh_prompt(question: str, answer_type: str) -> str:
    """Render the BBH prompt with an explicit, parseable final-answer instruction."""
    if answer_type == _MULTIPLE_CHOICE:
        instruction = (
            "Think step by step, then end with 'Answer: (X)' where X is the letter of "
            "the correct option."
        )
    else:
        instruction = (
            "Think step by step, then end with 'Answer: <answer>' giving only the final "
            "answer, exactly as it should be written."
        )
    return f"{question.strip()}\n\n{instruction}"


# --------------------------------------------------------------------------- #
# Scoring (pure text; no execution)
# --------------------------------------------------------------------------- #
_ANSWER_LEAD = re.compile(r"(?:final\s+)?answer\s*(?:is|:)\s*(.+)", re.IGNORECASE | re.DOTALL)


def _final_answer_segment(text: str) -> str:
    """Pull the answer portion out of a (possibly chatty) exact-match output.

    Prefers the text after an explicit ``"answer is"`` / ``"answer:"`` lead; else the
    last non-empty line (final answers come last). Returns ``""`` for empty input.
    """
    if not text:
        return ""
    m = _ANSWER_LEAD.search(text)
    seg = m.group(1) if m else next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
    # Keep only the first line of the captured segment (the answer proper).
    return seg.strip().splitlines()[0].strip() if seg.strip() else ""


def _normalize_exact(text: str) -> str:
    """Normalise a free-form answer for tolerant exact comparison.

    Lower-cases, strips surrounding quotes/brackets/terminal punctuation, and
    collapses internal whitespace — only formatting noise, never content.
    """
    s = str(text).strip().lower()
    s = s.strip(".\"'`()[]{} \t\n")
    s = re.sub(r"\s+", " ", s)
    return s


def score_bbh(candidate: str, reference: Any) -> float:
    """Binary reward for a BBH answer, dispatching on the reference's answer type.

    * ``multiple_choice`` — compare the option letter extracted from the candidate to
      the letter of the gold target (``"(B)"`` -> ``"B"``).
    * ``exact_match`` — normalized string equality of the candidate's final answer
      against the gold target.

    ``reference`` is the :class:`BBHReference` dict; a bare string reference is treated
    as an exact-match target.
    """
    from trinity.orchestration.reward import extract_choice_letter

    if isinstance(reference, dict):
        gold = str(reference.get("answer", ""))
        answer_type = str(reference.get("answer_type", _EXACT_MATCH))
    else:
        gold, answer_type = str(reference), _EXACT_MATCH
    if not gold.strip():
        return 0.0

    if answer_type == _MULTIPLE_CHOICE:
        cand_letter = extract_choice_letter(candidate)
        gold_letter = extract_choice_letter(gold)
        return 1.0 if cand_letter is not None and cand_letter == gold_letter else 0.0

    cand = _normalize_exact(_final_answer_segment(candidate))
    return 1.0 if cand and cand == _normalize_exact(gold) else 0.0


# --------------------------------------------------------------------------- #
# Loader (HuggingFace + offline toy fallback)
# --------------------------------------------------------------------------- #
def _row_get(row: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return row[k]
        except TypeError:
            break
    return default


def _make_task(subtask: str, index: int, question: str, target: str) -> Task:
    """Normalise one BBH row into a :class:`Task`."""
    answer_type = SUBTASKS.get(subtask, _EXACT_MATCH)
    ref = BBHReference(answer=str(target), answer_type=answer_type, subtask=subtask)
    return Task(
        task_id=f"bbh-{subtask}-{index}",
        benchmark=BENCHMARK,
        prompt=build_bbh_prompt(str(question), answer_type),
        answer=ref.to_dict(),
        meta={"source": _HF_DATASET, "subtask": subtask, "answer_type": answer_type,
              "task_type": TaskType.MCQ.value if answer_type == _MULTIPLE_CHOICE else TaskType.MATH.value},
    )


def _hf_bbh(split: str) -> list[Task] | None:
    """Load BBH across its subtasks from HuggingFace, or ``None`` on total failure."""
    try:
        from datasets import load_dataset
    except Exception:
        return None

    tasks: list[Task] = []
    for subtask in SUBTASKS:
        try:
            ds = load_dataset(_HF_DATASET, subtask, split=split or "test")
        except Exception:
            continue
        for i, row in enumerate(ds):
            question = _row_get(row, "input", "question", default="")
            target = _row_get(row, "target", "answer", default="")
            if not question or target in (None, ""):
                continue
            tasks.append(_make_task(subtask, i, str(question), str(target)))
    return tasks or None


def _toy_bbh() -> list[Task]:
    """A tiny, self-contained BBH-style set (one of each format) for offline smoke."""
    return [
        _make_task(
            "date_understanding", 0,
            "Yesterday was 01/01/2021. What is the date today?\n"
            "Options:\n(A) 01/01/2021\n(B) 01/02/2021\n(C) 12/31/2020",
            "(B)",
        ),
        _make_task(
            "boolean_expressions", 0,
            "Evaluate the result of the following boolean expression: not (True and False).",
            "True",
        ),
        _make_task(
            "object_counting", 0,
            "I have a chair, two tables, and a lamp. How many objects do I have?",
            "4",
        ),
    ]


def load_bbh_tasks(split: str, max_items: int | None, seed: int = 0) -> list[Task]:
    """Load BBH as a deterministic list of :class:`Task` across its subtasks.

    Tries HuggingFace (lazy/guarded, per-subtask); on total failure falls back to the
    built-in toy set. Applies a ``seed``-seeded shuffle and truncates to ``max_items``,
    so repeated calls with identical arguments return identical lists.
    """
    import random

    tasks = _hf_bbh(split) or _toy_bbh()
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class BBHAdapter(BenchmarkAdapter):
    """BIG-Bench Hard: reasoning question in, short textual answer out.

    Grading is pure text (:func:`score_bbh`) — multiple-choice by option letter,
    exact-match by normalized string equality — so there is no code execution and no
    sandbox. The per-task answer format lives in the structured reference; the
    benchmark-level :meth:`task_type` is nominal (BBH mixes MCQ and free-form).
    """

    name = BENCHMARK

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return load_bbh_tasks(split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return score_bbh(output, reference)

    def task_type(self) -> TaskType:
        # Nominal: BBH mixes multiple-choice and exact-match; the per-task format is
        # carried in the reference and honoured by score_bbh.
        return TaskType.MCQ

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": task.meta.get("task_type", TaskType.MCQ.value),
            "meta": dict(task.meta),
        }

    def score_trajectory(self, traj) -> float:
        # BBH is not a reward.score_text family, so committed-answer selection (which
        # keys on has_answer) doesn't apply; grade the final answer directly.
        return self.score_output(traj.final_answer or "", traj.task.answer)


def register_bbh_adapter() -> None:
    """Register the BBH adapter (idempotent-friendly)."""
    from .registry import is_registered

    if not is_registered(BENCHMARK):
        register_adapter(BENCHMARK, BBHAdapter())
