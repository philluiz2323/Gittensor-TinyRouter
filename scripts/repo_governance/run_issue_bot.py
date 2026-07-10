#!/usr/bin/env python3
"""Apply issue-bot decisions through the GitHub REST API."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from issue_bot import analyze_issue

_API = "https://api.github.com"


def _request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
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
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _fetch_issue(repo: str, issue_number: int, token: str) -> dict:
    url = f"{_API}/repos/{repo}/issues/{issue_number}"
    return _request("GET", url, token)


def _add_labels(repo: str, issue_number: int, labels: list[str], token: str) -> None:
    if not labels:
        return
    url = f"{_API}/repos/{repo}/issues/{issue_number}/labels"
    _request("POST", url, token, {"labels": labels})


def _post_comment(repo: str, issue_number: int, body: str, token: str) -> None:
    url = f"{_API}/repos/{repo}/issues/{issue_number}/comments"
    _request("POST", url, token, {"body": body})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run issue-bot against a live GitHub issue.")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token and not args.dry_run:
        print("GITHUB_TOKEN is required unless --dry-run is set.", file=sys.stderr)
        return 1

    issue = _fetch_issue(args.repo, args.issue_number, token) if token else {}
    title = issue.get("title", "")
    body = issue.get("body") or ""

    analysis = analyze_issue(title, body)
    print(json.dumps(analysis.__dict__, indent=2))

    if args.dry_run:
        return 0

    try:
        _add_labels(args.repo, args.issue_number, analysis.labels, token)
        if analysis.comment:
            marker = "<!-- tinyrouter-issue-bot -->"
            _post_comment(
                args.repo,
                args.issue_number,
                f"{marker}\n\n{analysis.comment}",
                token,
            )
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
