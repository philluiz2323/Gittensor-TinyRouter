"""Offline data-quality audit of a built benchmark.

Why this exists
---------------
``scripts/verify_benchmark.py`` checks a built benchmark's *integrity* (its hash
matches the committed manifest), and ``trinity.format_audit`` checks whether the
cached *answers* parse. Neither checks the *questions* themselves, yet a benchmark
with data-quality defects silently corrupts every downstream number:

* **duplicate ``question_id``** — the per-query maps
  (``EvalResult.per_query_binary``, the oracle solve matrix) are keyed by id, so a
  collision makes one question overwrite another and the counts lie;
* **duplicate question text** — the same question under two ids double-weights it
  in the accuracy mean and inflates any agreement/oracle statistic;
* **missing prompt or reference answer** — an unscoreable item that still counts
  toward the denominator, dragging accuracy down for a reason unrelated to the
  router.

This module reports exactly those defects from the built items, per benchmark, so
they are caught before a paid eval is run on a broken set. It reads the items
only; it costs nothing and changes nothing.

Pure / deterministic / no network / no GPU / no torch.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "BenchmarkQuality",
    "DatasetQualityReport",
    "audit_dataset",
    "normalize_question",
]

_DEFAULT_BENCHMARK = "math500"
_WS = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Canonicalize question text for near-duplicate detection.

    Lowercases and collapses whitespace, so two items that differ only in casing
    or spacing are recognised as the same question. Not a semantic match — just
    enough to catch the common copy/paste duplicate.
    """
    return _WS.sub(" ", str(text).strip().lower())


@dataclass
class BenchmarkQuality:
    """Data-quality tally for one benchmark within a built set."""

    benchmark: str
    n_items: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_questions: int = 0
    missing_prompt: int = 0
    missing_answer: int = 0

    @property
    def ok(self) -> bool:
        """True iff no defect was found for this benchmark."""
        return not (
            self.duplicate_ids or self.duplicate_questions
            or self.missing_prompt or self.missing_answer
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_items": self.n_items,
            "duplicate_ids": list(self.duplicate_ids),
            "duplicate_questions": self.duplicate_questions,
            "missing_prompt": self.missing_prompt,
            "missing_answer": self.missing_answer,
            "ok": self.ok,
        }


@dataclass
class DatasetQualityReport:
    """Whole-set data-quality report, plus the per-benchmark split."""

    n_items: int = 0
    per_benchmark: dict[str, BenchmarkQuality] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff every benchmark is defect-free."""
        return all(b.ok for b in self.per_benchmark.values())

    @property
    def n_problems(self) -> int:
        """Total defective items/ids across the set (a rough severity count)."""
        n = 0
        for b in self.per_benchmark.values():
            n += len(b.duplicate_ids) + b.duplicate_questions + b.missing_prompt + b.missing_answer
        return n

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_items": self.n_items,
            "ok": self.ok,
            "n_problems": self.n_problems,
            "per_benchmark": {b: q.to_dict() for b, q in sorted(self.per_benchmark.items())},
        }


def audit_dataset(items: Iterable[Mapping[str, Any]]) -> DatasetQualityReport:
    """Audit built benchmark items for data-quality defects, per benchmark.

    Each item is the protocol shape from ``scripts/build_benchmark.py``:
    ``question_id`` / ``question_text`` / ``correct_answer`` / ``benchmark``.

    Duplicate ids and duplicate (normalized) questions are detected **within each
    benchmark** (the same id across two benchmarks is not a collision, since the
    eval keys by benchmark + id). Returns a :class:`DatasetQualityReport`.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    total = 0
    for item in items:
        total += 1
        benchmark = str(item.get("benchmark") or _DEFAULT_BENCHMARK)
        grouped.setdefault(benchmark, []).append(item)

    report = DatasetQualityReport(n_items=total)
    for benchmark, group in grouped.items():
        q = BenchmarkQuality(benchmark=benchmark, n_items=len(group))

        # ``or ""`` + the blank filter mirror the question_text path below: a
        # missing/``None`` question_id is "no id", not an id collision. Without the
        # guard, ``str(None)`` groups null-id items under a spurious ``"None"``
        # duplicate id (which also inflates the total quality-flag count).
        id_counts = Counter(
            str(it.get("question_id") or "") for it in group
            if str(it.get("question_id") or "").strip()
        )
        q.duplicate_ids = sorted(i for i, c in id_counts.items() if c > 1)

        # ``or ""`` (not the get-default) so a present ``question_text: None`` is
        # treated as blank, like the ``correct_answer`` path below — otherwise
        # ``str(None)`` is the truthy ``"None"``, which hides the missing prompt and
        # collides two None items into a spurious duplicate.
        text_counts = Counter(
            normalize_question(it.get("question_text") or "") for it in group
            if str(it.get("question_text") or "").strip()
        )
        # Count the EXTRA copies (a question appearing 3x contributes 2 duplicates).
        q.duplicate_questions = sum(c - 1 for c in text_counts.values() if c > 1)

        for it in group:
            if not str(it.get("question_text") or "").strip():
                q.missing_prompt += 1
            ref = it.get("correct_answer")
            if ref is None or (isinstance(ref, str) and not ref.strip()):
                q.missing_answer += 1

        report.per_benchmark[benchmark] = q

    return report
