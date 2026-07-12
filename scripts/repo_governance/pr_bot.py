#!/usr/bin/env python3
"""Deterministic pull-request governance checks for TinyRouter."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field

from paths import analyse_changed_paths

__all__ = ["analyze_pr", "main"]

_PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"<!--\s*brief description\s*-->",
    r"\(math500 or mmlu\)",
    r"\(your miner identity\)",
    r"\$xx\.xx",
    r"\(submission number\)",
    r"\(cma-es or other\)",
)

_ROUTING_MARKERS: tuple[str, ...] = (
    "[submission]",
    "head_weights.npy",
    "svf_scales.npy",
    "receipt.json",
)

# Only a checked routing-head checkbox counts — not the phrase appearing in the
# shared PR template that every general-improvement PR is told to keep.
_ROUTING_CHECKED_RE = re.compile(
    r"-\s*\[x\]\s*\*\*routing head submission\*\*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PRAnalysis:
    """Outcome of a deterministic pull-request governance pass."""

    labels: list[str] = field(default_factory=list)
    sensitive_paths: list[str] = field(default_factory=list)
    template_violations: list[str] = field(default_factory=list)
    is_routing_submission: bool = False
    needs_maintainer_review: bool = False
    comment: str | None = None


def _contains_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in _PLACEHOLDER_PATTERNS)


def _routing_submission(body: str, *, title: str = "") -> bool:
    if _ROUTING_CHECKED_RE.search(body):
        return True
    combined = f"{title}\n{body}".lower()
    return any(marker in combined for marker in _ROUTING_MARKERS)


def _general_section_filled(body: str) -> bool:
    match = re.search(
        r"\*\*what does this pr do\?\*\*\s*\n+(.+?)(\n\*\*|\n---|\Z)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False
    content = match.group(1).strip()
    if not content or _contains_placeholder(content):
        return False
    return len(re.sub(r"[#*_`\-\s]", "", content)) >= 8


#: The routing-submission fields a miner must fill (as ``**Label:**`` regexes).
_ROUTING_FIELDS: tuple[str, ...] = (
    r"\*\*benchmark:\*\*",
    r"\*\*miner name:\*\*",
    r"\*\*generation:\*\*",
    r"\*\*training method:\*\*",
    r"\*\*training cost:\*\*",
)


def _field_value(body: str, label_re: str) -> str | None:
    """Return the value a submitter supplied for a ``**Label:**`` field.

    The value may sit **inline** on the label line — which is exactly how
    ``.github/PULL_REQUEST_TEMPLATE.md`` lays the routing fields out
    (``**Benchmark:** math500``) — or on the next non-blank line. An empty field
    returns ``None``, and the search never bleeds into the following ``**...**``
    label, so a blank field is not mistaken as filled by the next label.

    Args:
        body: The full PR description.
        label_re: A case-insensitive regex matching the ``**Label:**`` marker.

    Returns:
        The stripped value, or ``None`` when the label is absent or the field is
        blank.
    """
    m = re.search(label_re, body, flags=re.IGNORECASE)
    if not m:
        return None
    rest = body[m.end():]
    newline = rest.find("\n")
    inline = (rest if newline == -1 else rest[:newline]).strip()
    if inline:
        return inline
    for line in rest.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # The next bold label means this field was left empty.
        return None if re.match(r"\*\*.+?\*\*", stripped) else stripped
    return None


def _routing_section_filled(body: str) -> bool:
    """True iff every routing field carries a real (non-placeholder) value.

    Values may be inline (the shipped template's layout) or on the following
    line; a blank field or a leftover template placeholder counts as unfilled.
    """
    for label_re in _ROUTING_FIELDS:
        value = _field_value(body, label_re)
        if value is None or _contains_placeholder(value):
            return False
    return True


def analyze_pr(title: str, body: str, changed_files: list[str]) -> PRAnalysis:
    """Classify a PR, validate template usage, and flag sensitive paths."""
    title = title or ""
    body = body or ""
    sensitive, area_labels = analyse_changed_paths(changed_files)

    labels = list(area_labels)
    routing = _routing_submission(body, title=title)
    if routing:
        labels.append("submission")

    violations: list[str] = []
    if routing:
        if not _routing_section_filled(body):
            violations.append(
                "Routing submission template is incomplete (benchmark, miner, generation, "
                "training method, or training cost still looks like a placeholder)."
            )
    elif not _general_section_filled(body):
        violations.append(
            "General improvement section is missing a filled-in answer under "
            "'**What does this PR do?**'."
        )

    needs_maintainer = bool(sensitive) or routing
    if needs_maintainer:
        labels.append("needs-maintainer-review")

    comment_parts: list[str] = []
    if sensitive:
        joined = "\n".join(f"- `{path}`" for path in sensitive)
        comment_parts.append(
            "This PR touches sensitive scoring or protocol paths:\n\n"
            f"{joined}\n\n"
            "A maintainer review is required before merge."
        )
    if violations:
        comment_parts.append(
            "Template compliance issues detected:\n\n"
            + "\n".join(f"- {item}" for item in violations)
        )

    deduped_labels = list(dict.fromkeys(labels))
    return PRAnalysis(
        labels=deduped_labels,
        sensitive_paths=sensitive,
        template_violations=violations,
        is_routing_submission=routing,
        needs_maintainer_review=needs_maintainer,
        comment="\n\n".join(comment_parts) if comment_parts else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a GitHub pull request.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--files", default="[]", help="JSON array of changed file paths.")
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout.")
    args = parser.parse_args(argv)

    files = json.loads(args.files)
    result = analyze_pr(args.title, args.body, files)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
