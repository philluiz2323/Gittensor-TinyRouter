#!/usr/bin/env python3
"""Ensure governance labels exist before bots apply them."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

_API = "https://api.github.com"

REQUIRED_LABELS: tuple[dict[str, str], ...] = (
    {"name": "area:adapters", "color": "1d76db", "description": "Benchmark adapters and loaders"},
    {"name": "area:scoring", "color": "b60205", "description": "PR eval, leaderboard, and scoring gates"},
    {"name": "area:benchmark", "color": "5319e7", "description": "Hidden benchmark builder and protocol"},
    {"name": "area:training", "color": "0e8a16", "description": "CMA-ES training and fitness"},
    {"name": "area:orchestration", "color": "fbca04", "description": "Session loop, reward, and eval harness"},
    {"name": "area:coordinator", "color": "006b75", "description": "Coordinator encoder, head, and policy"},
    {"name": "area:conductor", "color": "d93f0b", "description": "Fugu / conductor workflow"},
    {"name": "area:infra", "color": "ededed", "description": "CI, workflows, and repository automation"},
    {"name": "area:tests", "color": "c2e0c6", "description": "Unit and harness tests"},
    {"name": "area:docs", "color": "bfdadc", "description": "Documentation changes"},
    {"name": "submission", "color": "f9d0c4", "description": "Routing head competition submission"},
    {
        "name": "needs-maintainer-review",
        "color": "e99695",
        "description": "Touches scoring, protocol, or submission surfaces",
    },
    {"name": "documentation", "color": "0075ca", "description": "Documentation-only change"},
)


def _request(method: str, url: str, token: str, payload: dict | None = None) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def ensure_labels(repo: str, token: str) -> None:
    for spec in REQUIRED_LABELS:
        url = f"{_API}/repos/{repo}/labels/{spec['name'].replace(':', '%3A')}"
        status, _ = _request("GET", url, token)
        if status == 200:
            continue
        create_url = f"{_API}/repos/{repo}/labels"
        status, body = _request("POST", create_url, token, spec)
        if status not in (200, 201):
            raise RuntimeError(f"Failed to create label {spec['name']}: {status} {body}")


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required.", file=sys.stderr)
        return 1
    ensure_labels(repo, token)
    print(f"Ensured governance labels on {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
