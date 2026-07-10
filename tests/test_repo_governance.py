"""Offline tests for repository governance automation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_GOV = Path(__file__).resolve().parents[1] / "scripts" / "repo_governance"
sys.path.insert(0, str(_GOV))

issue_bot = importlib.import_module("issue_bot")
pr_bot = importlib.import_module("pr_bot")
paths = importlib.import_module("paths")


def test_issue_bot_labels_bug_and_requests_missing_sections():
    body = "## Summary\nSomething broke.\n"
    result = issue_bot.analyze_issue("[bug] broken gate", body)
    assert result.labels[0] == "bug"
    assert result.missing_fields
    assert result.comment is not None


def test_issue_bot_flags_protocol_issues_for_maintainer_review():
    body = (
        "## Summary\nThe hidden benchmark gate rejects valid receipts.\n"
        "## Goal\nFix pr_eval Gate 4.\n"
    )
    result = issue_bot.analyze_issue("Gate mismatch", body)
    assert "needs-maintainer-review" in result.labels
    assert "area:scoring" in result.labels


def test_pr_bot_flags_sensitive_scoring_paths():
    files = ["scripts/pr_eval.py", "README.md"]
    body = (
        "## Type\n\n"
        "- [x] General improvement\n\n"
        "## General improvement\n\n"
        "**What does this PR do?**\n\n"
        "Fix the UTC timestamp parser in the rate-limit gate.\n\n"
        "**Why is it needed?**\n\n"
        "Security gate must be timezone-safe.\n"
    )
    result = pr_bot.analyze_pr("fix: parse UTC timestamps", body, files)
    assert "scripts/pr_eval.py" in result.sensitive_paths
    assert "needs-maintainer-review" in result.labels
    assert "area:scoring" in result.labels
    assert result.comment is not None


def test_pr_bot_detects_unfilled_general_template():
    body = (
        "## Type\n\n"
        "- [x] General improvement\n\n"
        "## General improvement\n\n"
        "**What does this PR do?**\n\n"
        "<!-- Brief description -->\n"
    )
    result = pr_bot.analyze_pr("docs: tweak readme", body, ["README.md"])
    assert result.template_violations
    assert result.comment is not None


def test_paths_marks_configs_and_leaderboard_as_sensitive():
    sensitive, labels = paths.analyse_changed_paths(
        ["leaderboard.json", "configs/trinity.yaml", "src/trinity/optim/fitness.py"]
    )
    assert "leaderboard.json" in sensitive
    assert "configs/trinity.yaml" in sensitive
    assert "area:training" in labels
