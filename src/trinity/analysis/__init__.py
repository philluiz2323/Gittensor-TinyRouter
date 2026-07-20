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
from trinity.analysis.ensemble import (
    EnsembleSummary,
    answers_agree,
    plurality_answer,
)
from trinity.analysis.ensemble import analyze as analyze_ensemble
from trinity.analysis.generalization import (
    GeneralizationGap,
    analyze_pair,
    overfit_verdict,
)
from trinity.analysis.reconcile import (
    PerModelReconciliation,
    ReconciliationSummary,
    reconcile,
    reconcile_files,
)
from trinity.analysis.sampling import (
    ModelSampling,
    SamplingSummary,
    solve_counts,
)
from trinity.analysis.sampling import analyze as analyze_sampling
from trinity.analysis.selective import (
    ModelSelective,
    SelectiveSummary,
    risk_coverage,
)
from trinity.analysis.selective import analyze as analyze_selective
from trinity.analysis.significance import (
    InvariantSignificance,
    PairedComparison,
    assess_invariants,
    mcnemar,
    paired_bootstrap_ci,
    paired_diff_test,
)
from trinity.analysis.union_oracle import (
    BenchmarkOracle,
    UnionOracleSummary,
    oracle_from_matrix,
    relative_error_reduction,
    union_oracle,
)
from trinity.analysis.transfer import (
    HELD_OUT,
    IN_DISTRIBUTION,
    BenchmarkMargin,
    CohortSummary,
    TransferSummary,
    classify,
)
from trinity.analysis.grader_audit import (
    BenchmarkAudit,
    ItemAudit,
    ProbeFinding,
    audit,
    audit_item,
    benchmark_kind,
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
    "EnsembleSummary",
    "answers_agree",
    "plurality_answer",
    "analyze_ensemble",
    "GeneralizationGap",
    "analyze_pair",
    "overfit_verdict",
    "PerModelReconciliation",
    "ReconciliationSummary",
    "reconcile",
    "reconcile_files",
    "ModelSampling",
    "SamplingSummary",
    "solve_counts",
    "analyze_sampling",
    "ModelSelective",
    "SelectiveSummary",
    "risk_coverage",
    "analyze_selective",
    "InvariantSignificance",
    "PairedComparison",
    "assess_invariants",
    "paired_bootstrap_ci",
    "paired_diff_test",
    "mcnemar",
    "BenchmarkOracle",
    "UnionOracleSummary",
    "oracle_from_matrix",
    "relative_error_reduction",
    "union_oracle",
    "BenchmarkAudit",
    "ItemAudit",
    "ProbeFinding",
    "audit",
    "audit_item",
    "benchmark_kind",
    "HELD_OUT",
    "IN_DISTRIBUTION",
    "BenchmarkMargin",
    "CohortSummary",
    "TransferSummary",
    "classify",
]
