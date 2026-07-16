"""Re-evaluating one PR is idempotent for the rate-limit gate.

`_record_attempt` consumes a daily slot as soon as Gate 1 passes, so a later
failure (a CI retry, or a transient GPU/API error during live scoring) leads the
maintainer to re-run `pr_eval` on the SAME PR. That re-run must not count the
attempt it already recorded and self-reject the PR as rate-limited. A DISTINCT PR
still counts, preserving the anti-probe intent. Offline: no GPU/network/torch.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pr_eval.py"
_spec = importlib.util.spec_from_file_location("pr_eval", _SCRIPT)
pr_eval = importlib.util.module_from_spec(_spec)
sys.modules["pr_eval"] = pr_eval
_spec.loader.exec_module(pr_eval)


def _stamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _lb_with_attempt(miner: str, pr: int, age_days: float = 0.05) -> dict:
    stamp = _stamp(time.time() - age_days * 86400)
    return {
        "benchmarks": {
            "math500": {
                "attempts": [{"miner": miner, "pr": pr, "timestamp": stamp}],
            }
        }
    }


def test_rerunning_the_same_pr_is_not_rate_limited():
    lb = _lb_with_attempt("alice", pr=42)
    assert pr_eval._check_rate_limit("alice", "math500", lb, current_pr=42) is None


def test_a_distinct_pr_still_counts():
    lb = _lb_with_attempt("alice", pr=42)
    err = pr_eval._check_rate_limit("alice", "math500", lb, current_pr=43)
    assert err is not None and "rate_limited" in err


def test_missing_pr_context_preserves_old_behavior():
    # Local preflight (no PR yet) or legacy callers count every attempt.
    lb = _lb_with_attempt("alice", pr=42)
    err = pr_eval._check_rate_limit("alice", "math500", lb)
    assert err is not None and "rate_limited" in err


def test_attempt_without_a_pr_field_still_counts():
    # An attempt whose PR is unknown cannot be identified as "the same PR", so
    # it must not be silently exempted.
    lb = {"benchmarks": {"math500": {"attempts": [
        {"miner": "alice", "timestamp": _stamp(time.time() - 3600)},
    ]}}}
    err = pr_eval._check_rate_limit("alice", "math500", lb, current_pr=42)
    assert err is not None and "rate_limited" in err


def test_prior_pr_plus_same_pr_rerun_counts_only_the_prior():
    # Alice already submitted PR 40 (a real prior submission) and is now being
    # re-evaluated on PR 42. Only PR 40 counts -> at the limit -> still blocked,
    # but it is the prior submission doing it, not PR 42's own recorded attempt.
    now = time.time()
    lb = {"benchmarks": {"math500": {"attempts": [
        {"miner": "alice", "pr": 40, "timestamp": _stamp(now - 0.5 * 86400)},
        {"miner": "alice", "pr": 42, "timestamp": _stamp(now - 3600)},
    ]}}}
    assert pr_eval._check_rate_limit("alice", "math500", lb, current_pr=42) is not None
    # Remove the genuine prior; now only PR 42's own attempt remains -> allowed.
    lb["benchmarks"]["math500"]["attempts"] = [
        {"miner": "alice", "pr": 42, "timestamp": _stamp(now - 3600)},
    ]
    assert pr_eval._check_rate_limit("alice", "math500", lb, current_pr=42) is None


def test_record_attempt_is_idempotent_per_pr(tmp_path, monkeypatch):
    lb_path = tmp_path / "leaderboard.json"
    lb_path.write_text(json.dumps({"benchmarks": {}}))
    monkeypatch.setattr(pr_eval, "_REPO", tmp_path)

    pr_eval._record_attempt("math500", "alice", 1, 42)
    pr_eval._record_attempt("math500", "alice", 1, 42)  # re-run same PR
    attempts = json.loads(lb_path.read_text())["benchmarks"]["math500"]["attempts"]
    assert len(attempts) == 1

    pr_eval._record_attempt("math500", "alice", 2, 43)  # distinct PR records
    attempts = json.loads(lb_path.read_text())["benchmarks"]["math500"]["attempts"]
    assert len(attempts) == 2
