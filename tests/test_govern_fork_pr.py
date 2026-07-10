"""Tests for fork-PR govern write-back permissions (issue #84, part 1)."""
from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
_GOV = _REPO / "scripts" / "repo_governance"
sys.path.insert(0, str(_GOV))

ensure_labels = importlib.import_module("ensure_labels")
run_pr_bot = importlib.import_module("run_pr_bot")


class TestEnsureLabelsPermissions:
    """Label bootstrap must not fail fork PR workflows on HTTP 403."""

    def test_ensure_labels_tolerates_403_on_create(self):
        def fake_request(method: str, url: str, token: str, payload=None):
            if method == "GET":
                return 404, ""
            if method == "POST":
                return 403, '{"message":"Resource not accessible by integration"}'
            return 500, ""

        with patch.object(ensure_labels, "_request", side_effect=fake_request):
            warnings = ensure_labels.ensure_labels("owner/repo", "token")
        assert warnings
        assert all("write permission" in w for w in warnings)

    def test_ensure_labels_main_exits_zero_with_warnings(self, capsys, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_TOKEN", "token")
        with patch.object(
            ensure_labels,
            "ensure_labels",
            return_value=["Cannot create label 'area:infra'"],
        ):
            code = ensure_labels.main()
        assert code == 0
        err = capsys.readouterr().err
        assert "warning:" in err


class TestRunPrBotWriteBack:
    """PR-bot analysis must succeed even when label write-back is forbidden."""

    def test_run_pr_bot_returns_zero_on_403(self, monkeypatch):
        import urllib.error

        pr_payload = json.dumps({"title": "fix: test", "body": "body"}).encode()

        def fake_urlopen(req):
            if "/pulls/" in req.full_url:
                return io.BytesIO(pr_payload)
            raise urllib.error.HTTPError(
                req.full_url,
                403,
                "Forbidden",
                hdrs=None,
                fp=io.BytesIO(b'{"message":"forbidden"}'),
            )

        monkeypatch.setattr(run_pr_bot.subprocess, "check_output", lambda *a, **k: "README.md\n")
        monkeypatch.setattr(run_pr_bot.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        code = run_pr_bot.main(
            [
                "--repo",
                "owner/repo",
                "--pr-number",
                "1",
                "--base-sha",
                "abc",
                "--head-sha",
                "def",
            ]
        )
        assert code == 0
