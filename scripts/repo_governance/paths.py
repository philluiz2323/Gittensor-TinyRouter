"""Shared path rules for repository governance automation."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass

# Paths that affect scoring, hidden benchmarks, or frozen protocol.
SENSITIVE_PATHS: tuple[str, ...] = (
    "scripts/pr_eval.py",
    "scripts/build_benchmark.py",
    "scripts/benchmark_protocol.py",
    "leaderboard.json",
    "configs/",
    "docs/BENCHMARK_PROTOCOL.md",
    "submissions/",
)

# Prefix -> label applied when a PR touches files under that prefix.
PATH_AREA_LABELS: tuple[tuple[str, str], ...] = (
    ("src/trinity/adapters/", "area:adapters"),
    ("scripts/pr_eval.py", "area:scoring"),
    ("scripts/build_benchmark.py", "area:benchmark"),
    ("scripts/benchmark_protocol.py", "area:benchmark"),
    ("src/trinity/optim/", "area:training"),
    ("src/trinity/orchestration/", "area:orchestration"),
    ("src/trinity/coordinator/", "area:coordinator"),
    ("src/trinity/fugu/", "area:conductor"),
    (".github/", "area:infra"),
    ("scripts/repo_governance/", "area:infra"),
    ("tests/", "area:tests"),
    ("docs/", "area:docs"),
)

PROTOCOL_KEYWORDS: frozenset[str] = frozenset(
    {
        "benchmark protocol",
        "hidden benchmark",
        "frozen protocol",
        "pr_eval",
        "leaderboard",
        "anti-cheat",
        "rate limit",
        "submission gate",
    }
)


@dataclass(frozen=True)
class PathMatch:
    """A changed file matched against governance rules."""

    path: str
    sensitive: bool
    area_labels: tuple[str, ...]


def _normalise_path(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def is_sensitive_path(path: str) -> bool:
    """Return True when ``path`` touches scoring or frozen-protocol surfaces."""
    norm = _normalise_path(path)
    for pattern in SENSITIVE_PATHS:
        if pattern.endswith("/"):
            if norm.startswith(pattern) or ("/" + pattern) in ("/" + norm):
                return True
        elif norm == pattern or norm.endswith("/" + pattern):
            return True
        elif fnmatch.fnmatch(norm, pattern):
            return True
    return False


def area_labels_for_path(path: str) -> tuple[str, ...]:
    """Return area labels implied by a changed file path."""
    norm = _normalise_path(path)
    labels: list[str] = []
    for prefix, label in PATH_AREA_LABELS:
        if norm.startswith(prefix) or fnmatch.fnmatch(norm, prefix):
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def analyse_changed_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(sensitive_paths, sorted_unique_area_labels)``."""
    sensitive: list[str] = []
    labels: list[str] = []
    for path in paths:
        if is_sensitive_path(path):
            sensitive.append(_normalise_path(path))
        labels.extend(area_labels_for_path(path))
    return sensitive, sorted(set(labels))
