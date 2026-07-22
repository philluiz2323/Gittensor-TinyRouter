"""DROP (Discrete Reasoning Over Paragraphs) reading-comprehension adapter.

DROP presents a passage and a question that requires *discrete* reasoning over the
passage — arithmetic, counting, sorting, or selecting spans. Answers are numbers,
dates, or **sets of text spans**, and are graded by DROP's official metric: an
exact-match plus a normalized token-level **F1** (with article/punctuation removal and
number normalization), taking the best score over the set of acceptable gold answers.

Scoring is **pure text** — no code execution, no sandbox. Because DROP answers are
short free-form strings graded by F1 (not a single option letter), this adapter carries
the gold answer set in its structured reference and grades with :func:`score_drop`
directly rather than through a ``reward.score_text`` family key (issue #275).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from trinity.types import Task

from .base import BenchmarkAdapter, TaskType
from .registry import register_adapter

__all__ = [
    "BENCHMARK",
    "DropReference",
    "build_drop_prompt",
    "drop_em_f1",
    "score_drop",
    "load_drop_tasks",
    "DropAdapter",
    "register_drop_adapter",
]

#: Canonical benchmark name this adapter registers under.
BENCHMARK = "drop"

#: HuggingFace dataset id.
_HF_DATASET = "ucinlp/drop"


# --------------------------------------------------------------------------- #
# Structured reference schema
# --------------------------------------------------------------------------- #
@dataclass
class DropReference:
    """The structured ``reference`` for a DROP task (stored as ``Task.answer``).

    ``gold_answers`` is the set of acceptable answer strings (DROP collects multiple
    validated annotations; a candidate is correct if it matches any of them). A
    multi-span answer is stored as a single string with its spans space-joined.
    """

    gold_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (the on-disk / hidden-benchmark form)."""
        return {"gold_answers": list(self.gold_answers)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DropReference":
        """Rebuild a :class:`DropReference` from :meth:`to_dict` output."""
        return cls(gold_answers=[str(a) for a in (data.get("gold_answers", []) or [])])

    def is_valid(self) -> bool:
        """A reference is gradable iff it carries at least one non-empty gold answer."""
        return any(str(a).strip() for a in self.gold_answers)


# --------------------------------------------------------------------------- #
# Prompt format
# --------------------------------------------------------------------------- #
def build_drop_prompt(passage: str, question: str) -> str:
    """Render the DROP passage/question prompt with a concise-answer instruction."""
    return (
        f"Passage:\n{passage.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "Reason over the passage, then end with 'Answer: <answer>' giving only the "
        "concise final answer (a number, a date, or the exact span(s))."
    )


# --------------------------------------------------------------------------- #
# DROP metric (EM + token-F1), pure text — the official normalization
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")
#: Matches only the ``"answer is"`` / ``"answer:"`` LEAD, deliberately WITHOUT a
#: greedy ``(.+)`` capture: a ``(.+)`` under ``DOTALL`` consumes to end-of-string, so
#: ``finditer`` yields a single first-lead match and the intended "take the last lead"
#: silently reads the chain-of-thought instead (#327/#340). Locating each lead
#: separately lets the caller slice the text after the *last* one.
_ANSWER_LEAD = re.compile(r"(?:final\s+)?answer\s*(?:is\s*:?|:)\s*", re.IGNORECASE)


def _final_answer_segment(text: str) -> str:
    """Pull the answer portion out of a (possibly chatty) output.

    Prefers the text after the **last** explicit ``"answer is"`` / ``"answer:"`` lead
    (final answers come last), skipping a trailing lead with no content after it; else
    the last non-empty line. Returns ``""`` for empty input.
    """
    if not text:
        return ""
    for lead in reversed(list(_ANSWER_LEAD.finditer(text))):
        after = text[lead.end():].strip()
        if after:
            first = after.splitlines()[0].strip()
            if first:
                return first
    last_line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
    return last_line.strip()


#: Surrounding punctuation stripped from a token, EXCLUDING the signs ``+``/``-`` — a
#: leading sign is part of a number's value, not wrapping noise.
_STRIP_EDGE = "".join(c for c in string.punctuation if c not in "+-")


def _normalize_token(raw: str) -> str:
    """Normalise one token: number-normalize if numeric, else strip its punctuation.

    Punctuation is stripped *per token* and skipped for numbers, so ``"16.0"`` and
    ``"$16"`` stay a single number. Two formats of the same number normalize equal to its
    value: a **thousands separator** is removed (``"1,000" == "1000"``, and ``"1,234.5"``
    is no longer corrupted to ``"12345"``) and a **leading sign is preserved** (``"-5"``
    stays negative, so it does not falsely match ``"5"``) — matching DROP's ``_normalize``,
    as this docstring already promised but the old ``strip(string.punctuation)`` (which
    dropped the ``-`` and left commas to break ``float()``) did not deliver."""
    core = raw.strip(_STRIP_EDGE)
    try:
        return str(float(core.replace(",", "")))
    except ValueError:
        return _PUNCT.sub("", raw)


def _split_internal_hyphens(token: str) -> list[str]:
    """Split ``token`` on INTERNAL hyphens, matching DROP's ``re.split(" |-", ...)``.

    DROP's official ``_normalize`` tokenizes on hyphens as well as whitespace, so a
    hyphenated span/range (``"1994-1995"``, ``"20-yard"``) and the same answer written
    with spaces are equal. A LEADING sign is *not* split off, preserving this module's
    deliberate negative-number strictness (``"-5"`` must not match ``"5"``): only a
    hyphen with content before it is a separator.

    Args:
        token: One whitespace-delimited token.

    Returns:
        The token's hyphen-separated parts (empties dropped), the leading sign kept on
        the first part.
    """
    if "-" not in token:
        return [token]
    sign, body = ("", token)
    if body[:1] in ("+", "-"):
        sign, body = body[0], body[1:]
    parts = [p for p in body.split("-") if p]
    if not parts:
        return [token] if not sign else []
    parts[0] = sign + parts[0]
    return parts


def _normalize_tokens(text: str) -> list[str]:
    """DROP answer normalization -> token list: lower-case, strip articles, split on
    whitespace and internal hyphens, then per-token punctuation/number normalization;
    empty tokens are dropped."""
    s = _ARTICLES.sub(" ", str(text).lower())
    raw = [sub for tok in s.split() for sub in _split_internal_hyphens(tok)]
    return [tok for tok in (_normalize_token(r) for r in raw) if tok]


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    """Bag-of-tokens F1 between two token lists (SQuAD/DROP style)."""
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def drop_em_f1(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    """Return ``(exact_match, token_f1)`` of ``prediction`` over a gold answer set.

    Each is the best score across ``gold_answers`` after DROP normalization: EM is
    ``1.0`` iff the normalized token strings are identical; F1 is the bag-of-tokens F1.
    """
    pred_tokens = _normalize_tokens(prediction)
    pred_str = " ".join(pred_tokens)
    best_em, best_f1 = 0.0, 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_tokens(gold)
        if not gold_tokens:
            continue
        best_em = max(best_em, 1.0 if pred_str == " ".join(gold_tokens) else 0.0)
        best_f1 = max(best_f1, _f1(pred_tokens, gold_tokens))
    return best_em, best_f1


def score_drop(candidate: str, reference: Any) -> float:
    """Binary reward for a DROP answer: ``1.0`` iff the token-F1 over the gold set is 1.0.

    ``reference`` is the :class:`DropReference` dict; a bare string/list reference is
    treated as the gold answer(s). The binary threshold (``F1 == 1.0``) credits an
    answer that contains all and only the gold tokens (order-independent), which
    subsumes exact match and the correct multi-span multiset.
    """
    if isinstance(reference, dict):
        gold = [str(a) for a in (reference.get("gold_answers", []) or [])]
    elif isinstance(reference, (list, tuple)):
        gold = [str(a) for a in reference]
    else:
        gold = [str(reference)]
    if not any(g.strip() for g in gold):
        return 0.0
    _em, f1 = drop_em_f1(_final_answer_segment(candidate), gold)
    return 1.0 if f1 >= 1.0 - 1e-9 else 0.0


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


def _render_gold(answer_obj: Any) -> list[str]:
    """Turn a DROP ``answers_spans``/answer object into acceptable answer strings.

    Accepts the HF shape ``{"spans": [...], "types": [...]}`` (spans joined into one
    answer string), a bare list of spans, or a plain string.
    """
    if isinstance(answer_obj, dict):
        spans = answer_obj.get("spans") or []
        if spans:
            return [" ".join(str(s) for s in spans)]
        number = answer_obj.get("number")
        if number not in (None, ""):
            return [str(number)]
        return []
    if isinstance(answer_obj, (list, tuple)):
        return [" ".join(str(s) for s in answer_obj)] if answer_obj else []
    return [str(answer_obj)] if str(answer_obj).strip() else []


def _make_task(index: int, passage: str, question: str, gold_answers: list[str]) -> Task:
    ref = DropReference(gold_answers=gold_answers)
    return Task(
        task_id=f"drop-{index}",
        benchmark=BENCHMARK,
        prompt=build_drop_prompt(passage, question),
        answer=ref.to_dict(),
        meta={"source": _HF_DATASET, "task_type": TaskType.MATH.value},
    )


def _hf_drop(split: str) -> list[Task] | None:
    """Load DROP from HuggingFace, or ``None`` on any failure."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset(_HF_DATASET, split=split or "validation")
    except Exception:
        return None

    tasks: list[Task] = []
    for i, row in enumerate(ds):
        passage = _row_get(row, "passage", default="")
        question = _row_get(row, "question", default="")
        gold = _render_gold(_row_get(row, "answers_spans", "answer", default=None))
        if not passage or not question or not gold:
            continue
        tasks.append(_make_task(i, str(passage), str(question), gold))
    return tasks or None


def _toy_drop() -> list[Task]:
    """A tiny, self-contained DROP-style set for offline smoke (number + span)."""
    passage = (
        "In the town, 12 houses were built in 2018 and 9 more in 2019. "
        "The mayor, Ana Ruiz, opened a new library in 2019."
    )
    return [
        _make_task(0, passage, "How many houses were built in 2018 and 2019 combined?", ["21"]),
        _make_task(1, passage, "Who opened the new library?", ["Ana Ruiz"]),
    ]


def load_drop_tasks(split: str, max_items: int | None, seed: int = 0) -> list[Task]:
    """Load DROP as a deterministic list of :class:`Task`.

    Tries HuggingFace (lazy/guarded); on any failure falls back to the built-in toy
    set. Applies a ``seed``-seeded shuffle and truncates to ``max_items``, so repeated
    calls with identical arguments return identical lists.
    """
    import random

    tasks = _hf_drop(split) or _toy_drop()
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class DropAdapter(BenchmarkAdapter):
    """DROP: passage + discrete-reasoning question in, short textual answer out.

    Graded by DROP's EM/token-F1 metric (:func:`score_drop`) — pure text, no code
    execution, no sandbox. ``task_type`` is nominal (free-form answer, ``MATH``-family).
    """

    name = BENCHMARK

    def load_tasks(self, split: str, max_items: int | None, seed: int = 0) -> list[Task]:
        return load_drop_tasks(split, max_items, seed=seed)

    def build_prompt(self, task: Task) -> str:
        return task.prompt

    def score_output(self, output: str, reference: Any) -> float:
        return score_drop(output, reference)

    def task_type(self) -> TaskType:
        return TaskType.MATH

    def serialize_task(self, task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "benchmark": task.benchmark,
            "prompt": task.prompt,
            "reference": task.answer,
            "task_type": TaskType.MATH.value,
            "meta": dict(task.meta),
        }

    def score_trajectory(self, traj) -> float:
        # DROP is not a reward.score_text family; grade the final answer directly.
        return self.score_output(traj.final_answer or "", traj.task.answer)


def register_drop_adapter() -> None:
    """Register the DROP adapter (idempotent-friendly)."""
    from .registry import is_registered

    if not is_registered(BENCHMARK):
        register_adapter(BENCHMARK, DropAdapter())
