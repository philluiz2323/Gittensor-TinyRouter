#!/usr/bin/env python3
"""Deterministic issue triage for TinyRouter repository governance."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field

from paths import PROTOCOL_KEYWORDS

__all__ = ["analyze_issue", "main"]

_BUG_SECTION_HINTS: tuple[tuple[str, str], ...] = (
    (r"describe the bug", "bug description"),
    (r"expected vs actual", "expected vs actual behavior"),
    (r"to reproduce|steps to reproduce", "reproduction steps"),
)

_ENHANCEMENT_SECTION_HINTS: tuple[tuple[str, str], ...] = (
    (r"##\s*summary|^\*\*summary\*\*", "summary"),
    (r"##\s*goal|^\*\*goal\*\*", "goal or motivation"),
)

#: Every section header, so one section's body never runs into the next header.
_ALL_SECTION_PATTERNS: tuple[str, ...] = tuple(
    pattern for pattern, _ in (*_BUG_SECTION_HINTS, *_ENHANCEMENT_SECTION_HINTS)
)


@dataclass(frozen=True)
class IssueAnalysis:
    """Outcome of a deterministic issue triage pass."""

    labels: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    needs_maintainer_review: bool = False
    comment: str | None = None


def _section_body(lines: list[str], header_idx: int) -> str:
    """Return the text between a section header and the next section header.

    Excludes the header line itself and stops at the next recognised section
    header, so a header's own words never count as its content and one empty
    section is not judged "filled" by the following section's header line.

    Args:
        lines: The issue body split into lines.
        header_idx: Index of the matched section-header line.

    Returns:
        The stripped body of the section (may be empty).
    """
    collected: list[str] = []
    for line in lines[header_idx + 1:]:
        if any(re.search(p, line, flags=re.IGNORECASE) for p in _ALL_SECTION_PATTERNS):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def _has_section(body: str, pattern: str) -> bool:
    if not re.search(pattern, body, flags=re.IGNORECASE | re.MULTILINE):
        return False
    # Require non-placeholder content in the section's OWN body — the lines after
    # its header and before the next section header. Measuring the header line (or
    # the following section's headers) let an empty section pass as filled.
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        if re.search(pattern, line, flags=re.IGNORECASE):
            section = _section_body(lines, idx)
            cleaned = re.sub(r"[#*_`\-\s]", "", section).lower()
            if len(cleaned) >= 12 and "tbd" not in cleaned and "todo" not in cleaned:
                return True
    return False


def _mentions_protocol(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in PROTOCOL_KEYWORDS)


def _base_type_label(title: str, body: str) -> str:
    title_l = title.strip().lower()
    body_l = body.lower()
    if title_l.startswith("[bug]"):
        return "bug"
    if "submission" in title_l or "routing head" in body_l:
        return "submission"
    if title_l.startswith("[docs]"):
        return "documentation"
    return "enhancement"


def _area_labels(title: str, body: str) -> list[str]:
    text = f"{title}\n{body}".lower()
    labels: list[str] = []
    if any(k in text for k in ("adapter", "benchmark", "dataset", "loader")):
        labels.append("area:adapters")
    if any(k in text for k in ("pr_eval", "scoring", "leaderboard", "gate")):
        labels.append("area:scoring")
    if any(k in text for k in ("cma", "train", "fitness", "sep-cmaes")):
        labels.append("area:training")
    if any(k in text for k in ("eval", "oracle", "reward", "session")):
        labels.append("area:orchestration")
    if any(k in text for k in ("workflow", "conductor", "fugu", "grpo")):
        labels.append("area:conductor")
    if any(k in text for k in ("ci", "workflow", "github actions", ".github")):
        labels.append("area:infra")
    return labels


def analyze_issue(title: str, body: str) -> IssueAnalysis:
    """Classify an issue and decide whether a guidance comment is needed."""
    title = title or ""
    body = body or ""
    labels = [_base_type_label(title, body)]
    labels.extend(_area_labels(title, body))

    needs_maintainer = _mentions_protocol(f"{title}\n{body}")
    if needs_maintainer:
        labels.append("needs-maintainer-review")

    missing: list[str] = []
    if labels[0] == "bug":
        for pattern, name in _BUG_SECTION_HINTS:
            if not _has_section(body, pattern):
                missing.append(name)
    else:
        for pattern, name in _ENHANCEMENT_SECTION_HINTS:
            if not _has_section(body, pattern):
                missing.append(name)

    comment = None
    if missing:
        bullets = "\n".join(f"- {item}" for item in missing)
        comment = (
            "Thanks for opening this issue. To help maintainers review it quickly, "
            "please add the missing details below:\n\n"
            f"{bullets}\n\n"
            "For bugs, include the command you ran, the config snippet, and any "
            "error output. For benchmark or scoring issues, note the benchmark name "
            "and affected file path if known."
        )

    # Stable label order without duplicates.
    deduped_labels = list(dict.fromkeys(labels))
    return IssueAnalysis(
        labels=deduped_labels,
        missing_fields=missing,
        needs_maintainer_review=needs_maintainer,
        comment=comment,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a GitHub issue for triage.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout.")
    args = parser.parse_args(argv)

    result = analyze_issue(args.title, args.body)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
