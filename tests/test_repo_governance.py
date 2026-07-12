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
# Routing template: fields are filled INLINE (the shipped template's layout)
# --------------------------------------------------------------------------- #
def _inline_routing_body(cost: str = "$25.00") -> str:
    return (
        "## Type\n\n"
        "- [x] **Routing head submission** — trained head\n\n"
        "## Routing head submission\n\n"
        "**Benchmark:** math500\n"
        "**Miner name:** miner-a\n"
        "**Generation:** 1\n"
        "**Training method:** CMA-ES\n"
        f"**Training cost:** {cost}\n"
    )


def test_inline_filled_routing_submission_has_no_template_violation():
    # Regression: the PR template puts each value inline on the label line, and a
    # correctly-filled submission must NOT be nagged as incomplete.
    result = pr_bot.analyze_pr(
        "[submission] miner-a gen 1", _inline_routing_body(),
        ["submissions/miner-a/1/head_weights.npy"],
    )
    assert result.is_routing_submission is True
    assert not any("Routing submission template" in v for v in result.template_violations)


def test_next_line_filled_routing_submission_still_accepted():
    body = (
        "- [x] **Routing head submission**\n\n"
        "**Benchmark:**\nmath500\n"
        "**Miner name:**\nminer-a\n"
        "**Generation:**\n1\n"
        "**Training method:**\nCMA-ES\n"
        "**Training cost:**\n$25.00\n"
    )
    assert pr_bot._routing_section_filled(body) is True


def test_placeholder_routing_fields_are_flagged():
    # The unedited template defaults ("(math500 or mmlu)", "$XX.XX ...") are placeholders.
    body = (
        "- [x] **Routing head submission**\n\n"
        "**Benchmark:** (math500 or mmlu)\n"
        "**Miner name:** (your miner identity)\n"
        "**Generation:** (submission number)\n"
        "**Training method:** (CMA-ES or other)\n"
        "**Training cost:** $XX.XX (from cost ledger)\n"
    )
    assert pr_bot._routing_section_filled(body) is False
    result = pr_bot.analyze_pr("[submission] x", body, ["submissions/x/1/head_weights.npy"])
    assert any("Routing submission template" in v for v in result.template_violations)


def test_empty_routing_field_does_not_bleed_into_next_label():
    # An empty Benchmark field must count as unfilled, not be "filled" by the
    # following **Miner name:** label.
    body = (
        "- [x] **Routing head submission**\n\n"
        "**Benchmark:**\n\n"
        "**Miner name:** miner-a\n"
        "**Generation:** 1\n"
        "**Training method:** CMA-ES\n"
        "**Training cost:** $25.00\n"
    )
    assert pr_bot._field_value(body, r"\*\*benchmark:\*\*") is None
    assert pr_bot._routing_section_filled(body) is False
