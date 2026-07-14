"""Offline integrity + status for ``leaderboard.json`` (the committed competition record).

``scripts/pr_eval.py`` WRITES ``leaderboard.json`` (``_update_leaderboard`` appends a
win to ``history`` and bumps ``best_*``; ``_record_attempt`` appends to the rate-limit
``attempts`` ledger) — but nothing ever re-reads it to check the writes are internally
consistent, even though ``docs/CI.md`` lists it as a **sensitive path** and
``gates.check_rate_limit`` trusts its ``attempts`` ledger verbatim. This is the same
"writer exists, verifier absent" hole ``scripts/verify_benchmark.py`` (#174) closed for
the benchmark manifest; this closes it for the leaderboard (a named ``ROADMAP.md`` infra
priority: "leaderboard automation").

:func:`verify_leaderboard` cross-checks the committed record and returns a list of
problems (empty = clean). It is **read-only** — it detects tampering (an inflated
``best_score``, a score above the achievable ``oracle_ceiling``, a ``best_*`` pointer
that doesn't match the winning history entry, a **truncated ``attempts`` ledger** that
would let a past winner defeat the weekly rate limit, duplicate/mis-owned PRs, or
non-monotone timestamps) and never modifies anything. It **reuses**
``gates.parse_utc_timestamp`` / ``gates.rate_limit_entries`` and the ``RATE_LIMIT_*``
constants, so it can never drift from the gate that trusts the file. Pure python — no
torch, no network.
"""
from __future__ import annotations

from typing import Any, Mapping, TypeGuard

from trinity.submission.constants import RATE_LIMIT_MAX_SUBMISSIONS, RATE_LIMIT_WINDOW_DAYS
from trinity.submission.gates import parse_utc_timestamp, rate_limit_entries

__all__ = ["verify_leaderboard", "leaderboard_report", "headroom_captured"]

_HISTORY_FIELDS = ("miner", "generation", "score", "pr", "timestamp")
_ATTEMPT_FIELDS = ("miner", "pr", "timestamp")
_TOL = 1e-9


def _is_num(x: Any) -> TypeGuard[float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_list(x: Any) -> list[Any]:
    """``x`` if it is a JSON array, else ``[]`` (a non-list is flagged, not iterated)."""
    return x if isinstance(x, list) else []


def _in_unit(x: Any) -> bool:
    return _is_num(x) and -_TOL <= x <= 1.0 + _TOL


def headroom_captured(entry: Mapping[str, Any]) -> float | None:
    """Fraction of the routable headroom the leader captured, or None if undefined.

    ``(best_score - best_single_model) / (oracle_ceiling - best_single_model)`` — how far
    the current leader closed the gap from the best fixed single model to the routing
    oracle. None when any term is missing or the denominator is non-positive.
    """
    bsm, oc, bs = entry.get("best_single_model"), entry.get("oracle_ceiling"), entry.get("best_score")
    if _is_num(bsm) and _is_num(oc) and _is_num(bs):
        denom = oc - bsm
        return (bs - bsm) / denom if denom > _TOL else None
    return None


def _verify_bench(name: str, entry: Any, problems: list[str]) -> None:
    tag = f"benchmarks.{name}"
    if not isinstance(entry, dict):
        problems.append(f"{tag}: not a JSON object")
        return

    # A tampered record can carry a non-list `history`/`attempts` (a scalar where an
    # array belongs); flag it and treat as empty rather than crashing on iteration.
    raw_history = entry.get("history")
    if raw_history is not None and not isinstance(raw_history, list):
        problems.append(f"{tag}.history is not a JSON array")
    history = _as_list(raw_history)
    raw_attempts = entry.get("attempts")
    if raw_attempts is not None and not isinstance(raw_attempts, list):
        problems.append(f"{tag}.attempts is not a JSON array")

    # score fields in [0, 1]; a best_score above the oracle ceiling is impossible.
    bs = entry.get("best_score")
    if not _in_unit(bs):
        problems.append(f"{tag}.best_score {bs!r} not a number in [0, 1]")
    for key in ("baseline_random", "best_single_model", "oracle_ceiling"):
        v = entry.get(key)
        if v is not None and not _in_unit(v):
            problems.append(f"{tag}.{key} {v!r} not a number in [0, 1]")
    oc = entry.get("oracle_ceiling")
    if _is_num(bs) and _is_num(oc) and float(bs) > float(oc) + _TOL:
        problems.append(f"{tag}.best_score {bs} exceeds oracle_ceiling {oc} (physically impossible)")

    # history entries: required fields, merged=true, valid score, non-decreasing time.
    max_score: float | None = None
    max_entry: Mapping[str, Any] | None = None
    prev_ts: float | None = None
    for i, h in enumerate(history):
        if not isinstance(h, dict):
            problems.append(f"{tag}.history[{i}]: not a JSON object")
            continue
        for f in _HISTORY_FIELDS:
            if f not in h:
                problems.append(f"{tag}.history[{i}] missing '{f}'")
        if h.get("merged") is not True:
            problems.append(f"{tag}.history[{i}] is not merged=true")
        sc = h.get("score")
        if _is_num(sc):
            if not _in_unit(sc):
                problems.append(f"{tag}.history[{i}].score {sc} not in [0, 1]")
            if max_score is None or float(sc) > max_score:
                max_score, max_entry = float(sc), h
        ts = parse_utc_timestamp(h.get("timestamp", ""))
        if ts is None:
            problems.append(f"{tag}.history[{i}] bad timestamp {h.get('timestamp')!r}")
        elif prev_ts is not None and ts + _TOL < prev_ts:
            problems.append(f"{tag}.history[{i}] timestamp goes backwards")
        if ts is not None:
            prev_ts = ts

    # best_* must equal the winning (max-score) history entry; empty history -> unclaimed.
    if history:
        if _is_num(bs) and max_score is not None and abs(float(bs) - max_score) > _TOL:
            problems.append(f"{tag}.best_score {bs} != max history score {max_score} (tampered?)")
        if max_entry is not None:
            for lb_key, h_key in (("best_miner", "miner"), ("best_generation", "generation"),
                                  ("best_pr", "pr")):
                if entry.get(lb_key) != max_entry.get(h_key):
                    problems.append(
                        f"{tag}.{lb_key} {entry.get(lb_key)!r} != winning history {h_key} "
                        f"{max_entry.get(h_key)!r}"
                    )
    elif entry.get("best_miner") is not None:
        problems.append(f"{tag}.best_miner {entry.get('best_miner')!r} set but history is empty")

    # attempts ledger: fields, no duplicate (miner, pr), one owner per pr, monotone time.
    attempts = rate_limit_entries(entry)
    seen_pairs: set = set()
    pr_owner: dict = {}
    prev_ats: float | None = None
    for i, a in enumerate(attempts):
        if not isinstance(a, dict):
            problems.append(f"{tag}.attempts[{i}]: not a JSON object")
            continue
        for f in _ATTEMPT_FIELDS:
            if f not in a:
                problems.append(f"{tag}.attempts[{i}] missing '{f}'")
        pair = (a.get("miner"), a.get("pr"))
        if pair in seen_pairs:
            problems.append(f"{tag}.attempts[{i}] duplicate (miner, pr) {pair}")
        seen_pairs.add(pair)
        pr, miner = a.get("pr"), a.get("miner")
        if pr is not None:
            if pr in pr_owner and pr_owner[pr] != miner:
                problems.append(f"{tag}: pr {pr} claimed by both {pr_owner[pr]!r} and {miner!r}")
            pr_owner.setdefault(pr, miner)
        ats = parse_utc_timestamp(a.get("timestamp", ""))
        if ats is None:
            problems.append(f"{tag}.attempts[{i}] bad timestamp {a.get('timestamp')!r}")
        elif prev_ats is not None and ats + _TOL < prev_ats:
            problems.append(f"{tag}.attempts[{i}] timestamp goes backwards")
        if ats is not None:
            prev_ats = ats

    # every win must have a recorded attempt (a truncated ledger = rate-limit bypass).
    # Only enforced when the explicit `attempts` ledger exists as a list (else it falls
    # back to history and the check is vacuous).
    if isinstance(raw_attempts, list):
        att_pairs = {(a.get("miner"), a.get("pr")) for a in raw_attempts if isinstance(a, dict)}
        for i, h in enumerate(history):
            if isinstance(h, dict) and (h.get("miner"), h.get("pr")) not in att_pairs:
                problems.append(
                    f"{tag}.history[{i}] win ({h.get('miner')!r}, pr {h.get('pr')}) missing from "
                    f"attempts ledger (rate-limit bypass?)"
                )


def verify_leaderboard(lb: Mapping[str, Any]) -> list[str]:
    """Cross-check a parsed ``leaderboard.json`` for integrity. Empty list = clean.

    Reuses the same rate-limit parser/ledger accessor the gate uses, so the checks can't
    diverge from ``gates.check_rate_limit``. See the module docstring for the failure
    modes it catches.
    """
    if not isinstance(lb, dict):
        return ["leaderboard is not a JSON object"]
    problems: list[str] = []
    benches = lb.get("benchmarks")
    if not isinstance(benches, dict):
        return ["missing or invalid 'benchmarks' object"]

    newest: float | None = None
    for name, entry in benches.items():
        _verify_bench(name, entry, problems)
        if isinstance(entry, dict):
            for coll in ("history", "attempts"):
                for e in _as_list(entry.get(coll)):
                    if isinstance(e, dict):
                        ts = parse_utc_timestamp(e.get("timestamp", ""))
                        if ts is not None and (newest is None or ts > newest):
                            newest = ts

    up = lb.get("updated_at")
    if up is not None:
        ts = parse_utc_timestamp(up)
        if ts is None:
            problems.append(f"updated_at {up!r} is not a valid UTC timestamp")
        elif newest is not None and ts + _TOL < newest:
            problems.append(f"updated_at {up} is older than the newest entry timestamp")
    return problems


def leaderboard_report(lb: Mapping[str, Any], *, now: float | None = None) -> str:
    """Markdown status report: per-benchmark frontier + (if ``now`` given) rate status.

    ``now`` is a UTC epoch used only to count each miner's submissions inside the
    ``RATE_LIMIT_WINDOW_DAYS`` window (the same computation ``check_rate_limit`` does);
    pass it for a deterministic report, or leave None to omit the rate-limit section.
    """
    benches = lb.get("benchmarks") if isinstance(lb, dict) else None
    out = ["# Leaderboard status\n"]
    if not isinstance(benches, dict) or not benches:
        return "".join(out) + "\n_(no benchmarks)_\n"

    out.append("| benchmark | best | random | best-single | oracle | headroom captured | leader (pr) |")
    out.append("|---|---|---|---|---|---|---|")
    for name, e in benches.items():
        if not isinstance(e, dict):
            continue
        hc = headroom_captured(e)
        hc_s = f"{hc:.1%}" if hc is not None else "—"
        leader = f"{e.get('best_miner')} (#{e.get('best_pr')})" if e.get("best_miner") else "—"
        out.append(
            f"| {name} | {e.get('best_score')} | {e.get('baseline_random')} | "
            f"{e.get('best_single_model')} | {e.get('oracle_ceiling')} | {hc_s} | {leader} |"
        )

    if now is not None:
        cutoff = now - RATE_LIMIT_WINDOW_DAYS * 86400
        out.append(f"\n## Rate-limit status (window {RATE_LIMIT_WINDOW_DAYS}d, "
                   f"max {RATE_LIMIT_MAX_SUBMISSIONS}/miner)\n")
        for name, e in benches.items():
            if not isinstance(e, dict):
                continue
            recent: dict[str, int] = {}
            for a in rate_limit_entries(e):
                ts = parse_utc_timestamp(a.get("timestamp", "")) if isinstance(a, dict) else None
                if ts is not None and ts > cutoff and a.get("miner"):
                    recent[a["miner"]] = recent.get(a["miner"], 0) + 1
            status = ", ".join(f"{m}: {c}" for m, c in sorted(recent.items())) or "none"
            out.append(f"- **{name}**: {status}")
    return "\n".join(out) + "\n"
