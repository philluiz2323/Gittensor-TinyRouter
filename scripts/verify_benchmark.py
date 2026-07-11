#!/usr/bin/env python3
"""Verify a built hidden benchmark against its committed hash.txt / meta.json.

`scripts/build_benchmark.py` writes a public `hash.txt` + `meta.json` and prints
"Add this hash to the repo's benchmark_hashes.txt to prove the benchmark ... has not
been modified since creation" — and `docs/BENCHMARK_PROTOCOL.md` promises they "let
anyone verify the hidden benchmark has not changed". But nothing consumed them: no
script recomputes the manifest hash or checks a build against its manifest. This is
that missing verifier.

Two modes:

  # Full — decrypt the splits, recompute the manifest hash, and check it against BOTH
  # hash.txt and meta.json (+ per-split counts / ids / disjointness / sealed seed).
  python scripts/verify_benchmark.py --dir data/benchmarks/math500 --password ...
        (or set BENCHMARK_PASSWORD)

  # Offline self-consistency — validate meta.json against the frozen protocol using
  # only the committed public file (no password, no questions):
  python scripts/verify_benchmark.py --meta data/benchmarks/math500/meta.json

Add `--append benchmark_hashes.txt` to record a verified hash, finally making the
builder's instruction real. Verification reuses the canonical
`benchmark_protocol.manifest_hash` / `build_manifest`, so the verifier can never drift
from the builder's hashing. Pure/offline apart from the optional AES-GCM decryption.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts/ modules

import benchmark_protocol as protocol  # noqa: E402  (needs the sys.path insert above)

_SPLIT_FILES = {"eval": "eval.json", "audit": "audit.json", "live": "live.json"}


def _derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256 key derivation — identical to build_benchmark/pr_eval."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)


def _decrypt_file(path: Path, password: str) -> dict:
    """Decrypt one AES-256-GCM benchmark split file into its JSON dict."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    combined = base64.b64decode(Path(path).read_text().strip())
    salt, nonce, ct = combined[:16], combined[16:28], combined[28:]
    plain = AESGCM(_derive_key(password, salt)).decrypt(nonce, ct, None)
    return json.loads(plain.decode("utf-8"))


def load_splits(bench_dir: str | Path, password: str) -> tuple[dict[str, list], list[str]]:
    """Decrypt the three split files -> ``(splits, problems)``.

    Each file is ``{"seed", "count", "items": [...]}``; a missing file, a wrong sealed
    seed, or a ``count`` that disagrees with ``len(items)`` is recorded as a problem
    (the splits dict is still returned so the manifest check can run on what decoded).
    """
    bench_dir = Path(bench_dir)
    splits: dict[str, list] = {}
    problems: list[str] = []
    for name in protocol.SPLIT_ORDER:
        fp = bench_dir / _SPLIT_FILES[name]
        if not fp.exists():
            problems.append(f"missing split file {fp.name}")
            splits[name] = []
            continue
        data = _decrypt_file(fp, password)
        items = list(data.get("items") or [])
        splits[name] = items
        if data.get("seed") != protocol.SEALED_SEED:
            problems.append(f"{fp.name}: seed {data.get('seed')!r} != sealed {protocol.SEALED_SEED}")
        if data.get("count") is not None and int(data["count"]) != len(items):
            problems.append(f"{fp.name}: count {data.get('count')} != len(items) {len(items)}")
    return splits, problems


def verify_dir(bench_dir: str | Path, password: str) -> list[str]:
    """Full verification of a built benchmark directory. Empty list = verified."""
    bench_dir = Path(bench_dir)
    meta_path = bench_dir / "meta.json"
    if not meta_path.exists():
        return [f"missing {meta_path.name}"]
    meta = json.loads(meta_path.read_text())

    problems: list[str] = []
    hash_path = bench_dir / "hash.txt"
    expected_hash: str | None = hash_path.read_text() if hash_path.exists() else None
    if expected_hash is None:
        problems.append("missing hash.txt")
    elif expected_hash.strip() != meta.get("content_hash"):
        problems.append(
            f"hash.txt {expected_hash.strip()} != meta content_hash {meta.get('content_hash')}"
        )

    splits, split_problems = load_splits(bench_dir, password)
    problems += split_problems
    problems += protocol.verify_manifest(meta, splits, expected_hash=expected_hash)
    return problems


def verify_meta_file(meta_path: str | Path) -> list[str]:
    """Offline self-consistency: validate meta.json alone (no password, no questions)."""
    meta = json.loads(Path(meta_path).read_text())
    return protocol.verify_meta_selfconsistent(meta)


def append_hash(path: str | Path, benchmark: str, content_hash: str) -> bool:
    """Append ``'<benchmark>\\t<hash>'`` to a benchmark_hashes.txt (idempotent).

    Returns True if a new line was written, False if it was already present.
    """
    p = Path(path)
    line = f"{benchmark}\t{content_hash}"
    if p.exists() and line in p.read_text().splitlines():
        return False
    with p.open("a") as fh:
        fh.write(line + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a built hidden benchmark's integrity.")
    ap.add_argument("--dir", default=None, help="benchmark dir for FULL verify (needs a password)")
    ap.add_argument("--meta", default=None, help="meta.json for OFFLINE self-consistency verify")
    ap.add_argument("--password", default=os.environ.get("BENCHMARK_PASSWORD", ""),
                    help="decryption password (or set BENCHMARK_PASSWORD)")
    ap.add_argument("--append", default=None, dest="append",
                    help="append the verified '<benchmark>\\t<hash>' to this file")
    args = ap.parse_args()

    if args.dir:
        if not args.password:
            print("ERROR: --dir needs --password or BENCHMARK_PASSWORD")
            sys.exit(2)
        problems = verify_dir(args.dir, args.password)
        meta_path = Path(args.dir) / "meta.json"
        mode = f"full: {args.dir}"
    elif args.meta:
        problems = verify_meta_file(args.meta)
        meta_path = Path(args.meta)
        mode = f"self-consistency: {args.meta}"
    else:
        print("ERROR: pass --dir <benchmark_dir> (full) or --meta <meta.json> (offline)")
        sys.exit(2)

    if problems:
        print(f"FAIL [{mode}] — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    # Reached only when verification passed, so meta.json exists and is well-formed
    # (a missing/corrupt manifest is reported as a problem above). Reading it eagerly
    # before this point crashed the verifier on the very case it exists to report —
    # a build missing meta.json — instead of printing the FAIL report.
    meta = json.loads(meta_path.read_text())
    print(f"OK [{mode}] — {meta.get('benchmark')} verified, hash {meta.get('content_hash')}")
    if args.append:
        wrote = append_hash(args.append, str(meta.get("benchmark")), str(meta.get("content_hash")))
        print(f"  {'appended to' if wrote else 'already in'} {args.append}")


if __name__ == "__main__":
    main()
