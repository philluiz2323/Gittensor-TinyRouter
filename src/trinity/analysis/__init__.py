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
from trinity.analysis.convergence import (
    RunConvergence,
    analyze_run,
    analyze_runs,
    render,
)
from trinity.analysis.generalization import (
    GeneralizationGap,
    analyze_pair,
    overfit_verdict,
)
from trinity.analysis.significance import (
    InvariantSignificance,
    PairedComparison,
    assess_invariants,
    mcnemar,
    paired_bootstrap_ci,
    paired_diff_test,
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
    "RunConvergence",
    "analyze_run",
    "analyze_runs",
    "render",
    "GeneralizationGap",
    "analyze_pair",
    "overfit_verdict",
    "InvariantSignificance",
    "PairedComparison",
    "assess_invariants",
    "paired_bootstrap_ci",
    "paired_diff_test",
    "mcnemar",
]
