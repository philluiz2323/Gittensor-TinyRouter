"""Offline analyses over already-cached benchmark answers (no API cost).

Everything here is pure python/numpy and reads artifacts that already exist on
disk, so a diagnostic never re-pays for model calls.
"""
from __future__ import annotations

from trinity.analysis.agreement import (
    AgreementSummary,
    QuestionAgreement,
    contested_ids,
    grade_item,
    grade_items,
    summarize,
    to_oracle_matrix,
)
from trinity.analysis.complementarity import (
    ComplementaritySummary,
    PerModelComplementarity,
    analyze,
    analyze_tensor,
)

__all__ = [
    "AgreementSummary",
    "QuestionAgreement",
    "contested_ids",
    "grade_item",
    "grade_items",
    "summarize",
    "to_oracle_matrix",
    "ComplementaritySummary",
    "PerModelComplementarity",
    "analyze",
    "analyze_tensor",
]
