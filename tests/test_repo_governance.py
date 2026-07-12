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


def test_pr_bot_does_not_flag_general_pr_that_keeps_template_routing_text():
    """Regression for #84: the shared PR template mentions routing-head text."""
    template = (
        Path(__file__).resolve().parents[1] / ".github" / "PULL_REQUEST_TEMPLATE.md"
    ).read_text(encoding="utf-8")
    body = template.split("## Routing head submission")[0] + (
        "\n## General improvement\n\n"
        "**What does this PR do?**\n\n"
        "Fix the loader split-policy fallback.\n\n"
        "**Why is it needed?**\n\n"
        "Training was silently using the toy set.\n"
    )
    result = pr_bot.analyze_pr("fix: loader split policy", body, ["src/trinity/adapters/loaders.py"])
    assert result.is_routing_submission is False
    assert "submission" not in result.labels
    assert not any("Routing submission template" in v for v in result.template_violations)


def test_pr_bot_detects_checked_routing_submission_checkbox():
    body = (
        "## Type\n\n"
        "- [x] **Routing head submission** — trained head\n\n"
        "**Benchmark:** math500\n"
        "**Miner name:** miner-a\n"
        "**Generation:** 1\n"
        "**Training method:** CMA-ES\n"
        "**Training cost:** $25.00\n"
    )
    result = pr_bot.analyze_pr("[submission] miner-a gen 1", body, ["submissions/miner-a/1/head_weights.npy"])
    assert result.is_routing_submission is True
    assert "submission" in result.labels


def test_pr_bot_detects_routing_submission_from_title_tag():
    body = (
        "## Type\n\n"
        "- [x] General improvement\n\n"
        "## General improvement\n\n"
        "**What does this PR do?**\n\n"
        "Docs only.\n"
    )
    result = pr_bot.analyze_pr("[submission] miner-a gen 1 — math500", body, ["README.md"])
    assert result.is_routing_submission is True


# --------------------------------------------------------------------------- #
# issue_bot: an empty section must not pass triage on its header words
# --------------------------------------------------------------------------- #
def test_issue_bot_flags_all_empty_bug_sections():
    # A [bug] issue with only the (empty) section headers must report every
    # section as missing -- the header text must not count as content, and one
    # empty section must not be "filled" by the next section's header.
    body = "**Describe the bug**\n\n**Expected vs actual**\n\n**To reproduce**\n"
    result = issue_bot.analyze_issue("[bug] crash", body)
    for field_name in ("bug description", "expected vs actual behavior", "reproduction steps"):
        assert field_name in result.missing_fields
    assert result.comment is not None


def test_issue_bot_accepts_a_filled_bug_report():
    body = (
        "**Describe the bug**\nThe scorer raises IndexError on empty input.\n\n"
        "**Expected vs actual**\nExpected a score; it crashes instead.\n\n"
        "**To reproduce**\nCall score_text with an empty candidate list.\n"
    )
    result = issue_bot.analyze_issue("[bug] crash", body)
    assert result.missing_fields == []


def test_issue_bot_empty_section_not_filled_by_next_header():
    # Only the middle section is filled; the empty first/last must be flagged.
    body = (
        "**Describe the bug**\n\n"
        "**Expected vs actual**\nExpected success, observed a hard crash instead.\n\n"
        "**To reproduce**\n"
    )
    result = issue_bot.analyze_issue("[bug] x", body)
    assert "bug description" in result.missing_fields
    assert "reproduction steps" in result.missing_fields
    assert "expected vs actual behavior" not in result.missing_fields
