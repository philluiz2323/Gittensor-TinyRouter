"""Canonical cost-ledger hash chain for OpenRouter token accounting.

The training/eval stack appends one JSONL line per LLM call when
``TRINITY_COST_LEDGER`` is set. Each line carries a tamper-evident hash
``h = sha256(prev_h + payload)`` where ``payload`` is a **fixed-format**
compact JSON object::

    {"m":"<model>","p":<prompt_tok>,"c":<completion_tok>}

Every writer and verifier must hash the same ``payload`` string. Using
``json.dumps(..., sort_keys=True)`` in the verifier while the writer emits
compact key order makes every legitimate ledger fail verification.

This module is the single source of truth for payload formatting, hashing,
chain verification, and safe line parsing. It is importable from both
``src/trinity`` and ``scripts/`` without torch/GPU dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

__all__ = [
    "LedgerEntry",
    "ledger_payload",
    "ledger_entry_hash",
    "format_ledger_line",
    "parse_ledger_line",
    "verify_ledger_chain",
    "verify_ledger_chain_text",
    "read_ledger_entries",
    "tip_hash_from_text",
    "append_ledger_entry",
    "summarize_token_usage",
]


@dataclass(frozen=True)
class LedgerEntry:
    """One decoded cost-ledger record (hash field excluded)."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    line_number: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def ledger_payload(model: str, prompt_tokens: int, completion_tokens: int) -> str:
    """Return the canonical pre-hash payload for one ledger entry.

    Args:
        model: Short model slug (e.g. ``qwen3.5-35b-a3b``).
        prompt_tokens: Prompt token count from the API response.
        completion_tokens: Completion token count from the API response.

    Returns:
        Compact JSON string with keys ``m``, ``p``, ``c`` in fixed order.
    """
    short = model.rsplit("/", 1)[-1]
    pt = int(prompt_tokens)
    ct = int(completion_tokens)
    return f'{{"m":"{short}","p":{pt},"c":{ct}}}'


def ledger_entry_hash(
    prev_hash: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    """Compute the chained SHA-256 digest for one ledger entry.

    Args:
        prev_hash: Hash from the previous line (empty string for the first).
        model: Short model slug.
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.

    Returns:
        Hex digest linking this entry to ``prev_hash``.
    """
    payload = ledger_payload(model, prompt_tokens, completion_tokens)
    return hashlib.sha256((prev_hash + payload).encode()).hexdigest()


def format_ledger_line(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prev_hash: str = "",
) -> str:
    """Serialize one ledger JSONL line including its hash field.

    Args:
        model: Short model slug.
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        prev_hash: Previous line hash (empty for genesis entry).

    Returns:
        A single JSON object string suitable for appending to the ledger file.
    """
    short = model.rsplit("/", 1)[-1]
    pt = int(prompt_tokens)
    ct = int(completion_tokens)
    digest = ledger_entry_hash(prev_hash, short, pt, ct)
    return f'{{"m":"{short}","p":{pt},"c":{ct},"h":"{digest}"}}'


def parse_ledger_line(line: str, *, line_number: int = 0) -> tuple[LedgerEntry, str | None]:
    """Parse one ledger JSONL line.

    Args:
        line: Raw line text (may include trailing newline).
        line_number: 1-based line index for error messages.

    Returns:
        ``(entry, hash_or_none)`` where ``hash_or_none`` is the ``h`` field if
        present.

    Raises:
        ValueError: On invalid JSON or missing required fields.
    """
    text = line.strip()
    if not text:
        raise ValueError(f"line {line_number}: empty line")
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number}: invalid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: expected JSON object")
    expected_h = record.pop("h", None)
    model = record.pop("m", None)
    if not isinstance(model, str) or not model:
        raise ValueError(f"line {line_number}: missing model field 'm'")
    try:
        pt = int(record.pop("p"))
        ct = int(record.pop("c"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: invalid token fields") from exc
    if record:
        raise ValueError(f"line {line_number}: unexpected fields {sorted(record)}")
    return LedgerEntry(model=model, prompt_tokens=pt, completion_tokens=ct, line_number=line_number), expected_h


def verify_ledger_chain_text(text: str) -> tuple[bool, int, str]:
    """Verify a hash chain from in-memory ledger text.

    Args:
        text: Full ledger file contents (JSONL).

    Returns:
        ``(valid, num_entries, error_message)``. ``error_message`` is empty when
        valid.
    """
    prev_hash = ""
    entries = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entry, expected_h = parse_ledger_line(raw, line_number=lineno)
        except ValueError as exc:
            return False, entries, str(exc)
        if expected_h is None:
            return False, entries, (
                f"line {lineno}: missing hash field 'h' "
                "(ledger entries must be written with hash-chain enabled)"
            )
        computed_h = ledger_entry_hash(
            prev_hash, entry.model, entry.prompt_tokens, entry.completion_tokens
        )
        if computed_h != expected_h:
            return False, entries, (
                f"line {lineno}: hash mismatch — expected {computed_h[:16]}..., "
                f"got {expected_h[:16]}... (chain broken or entry tampered)"
            )
        prev_hash = computed_h
        entries += 1
    return True, entries, ""


def verify_ledger_chain(path: str | Path) -> tuple[bool, int, str]:
    """Verify the hash-chain integrity of a cost ledger file.

    Args:
        path: Path to ``cost_ledger.jsonl``.

    Returns:
        ``(valid, num_entries, error_message)``.
    """
    with open(path, encoding="utf-8") as fh:
        return verify_ledger_chain_text(fh.read())


def read_ledger_entries(path: str | Path) -> list[LedgerEntry]:
    """Read all entries from a ledger without verifying the hash chain.

    Skips blank lines. Does not validate hashes — use
    :func:`verify_ledger_chain` first when integrity matters.

    Args:
        path: Path to ``cost_ledger.jsonl``.

    Returns:
        Parsed entries in file order.
    """
    out: list[LedgerEntry] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            entry, _ = parse_ledger_line(raw, line_number=lineno)
            out.append(entry)
    return out


def tip_hash_from_text(text: str) -> str:
    """Return the ``h`` field of the last non-empty ledger line, or ``""``.

    Used by the writer to continue the chain from the file tip. Unlike
    :func:`verify_ledger_chain_text`, this does **not** require the prefix to
    be valid — a broken earlier line must not reset later appends to genesis.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    try:
        _, tip = parse_ledger_line(lines[-1])
    except ValueError:
        return ""
    return tip or ""


def _tip_hash_from_handle(file_handle: TextIO) -> str:
    """Read tip hash from an in-memory or seekable text handle."""
    getvalue = getattr(file_handle, "getvalue", None)
    if callable(getvalue):
        return tip_hash_from_text(getvalue())
    try:
        pos = file_handle.tell()
        file_handle.seek(0)
        text = file_handle.read()
        file_handle.seek(pos)
        return tip_hash_from_text(text)
    except (OSError, ValueError):
        return ""


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Cross-platform exclusive lock via a sidecar ``.lock`` file."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def append_ledger_entry(
    path: str | Path,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    file_handle: TextIO | None = None,
) -> str:
    """Append one hash-chained entry to a ledger file.

    Best-effort helper mirroring :func:`openrouter_client._ledger_append` but
    usable from tests and scripts. When ``file_handle`` is provided, writes to
    that handle instead of opening ``path`` (for in-memory testing) and derives
    ``prev_hash`` from the handle's current contents.

    On disk, the tip is read under an exclusive sidecar lock so concurrent
    OpenRouter writers share one chain tip. The writer always links to the
    **last line's** ``h``, even if an earlier line fails verification — resetting
    to genesis after a break permanently fragments the ledger.

    Args:
        path: Ledger file path (read for previous hash unless handle given).
        model: Model slug written to the ``m`` field.
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        file_handle: Optional open append handle.

    Returns:
        The new entry hash ``h``.
    """
    p = Path(path)

    if file_handle is not None:
        prev_hash = _tip_hash_from_handle(file_handle)
        line = format_ledger_line(model, prompt_tokens, completion_tokens, prev_hash=prev_hash)
        digest = json.loads(line)["h"]
        file_handle.write(line + "\n")
        return digest

    lock_path = Path(str(p) + ".lock")
    with _exclusive_lock(lock_path):
        prev_hash = ""
        if p.exists():
            try:
                prev_hash = tip_hash_from_text(p.read_text(encoding="utf-8"))
            except OSError:
                prev_hash = ""
        line = format_ledger_line(model, prompt_tokens, completion_tokens, prev_hash=prev_hash)
        digest = json.loads(line)["h"]
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return digest


def summarize_token_usage(entries: list[LedgerEntry]) -> dict[str, tuple[int, int, int]]:
    """Aggregate prompt/completion/call counts per model slug.

    Args:
        entries: Parsed ledger rows.

    Returns:
        ``model -> (prompt_tokens, completion_tokens, num_calls)``.
    """
    per: dict[str, list[int]] = {}
    for entry in entries:
        acc = per.setdefault(entry.model, [0, 0, 0])
        acc[0] += entry.prompt_tokens
        acc[1] += entry.completion_tokens
        acc[2] += 1
    return {m: (v[0], v[1], v[2]) for m, v in per.items()}
