"""Offline tests for the benchmark integrity verifier.

Covers the pure verify helpers in ``scripts/benchmark_protocol.py`` (which reuse the
canonical ``manifest_hash``/``build_manifest``, so the verifier can't drift from the
builder) and the ``scripts/verify_benchmark.py`` CLI functions, including a full
AES-GCM encrypt -> verify -> tamper round-trip gated on ``cryptography``.

No network, no GPU: synthetic splits, and (for the round-trip) a local encrypt that
mirrors ``build_benchmark``'s format via the verifier's own key-derivation.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


protocol = _load("benchmark_protocol")
vb = _load("verify_benchmark")


def _item(qid, text, ans, benchmark="math500"):
    return {"question_id": qid, "benchmark": benchmark, "task_type": "math",
            "question_text": text, "correct_answer": ans, "model_answers": {}}


def _splits():
    return {
        "eval": [_item("q1", "1+1?", "2"), _item("q2", "2+2?", "4")],
        "audit": [_item("q3", "3+3?", "6")],
        "live": [_item("q4", "4+4?", "8")],
    }


def _meta(splits):
    return protocol.build_manifest("math500", splits, seed=protocol.SEALED_SEED, created_at="t")


# --------------------------------------------------------------------------- #
# verify_meta_selfconsistent
# --------------------------------------------------------------------------- #
def test_selfconsistent_clean_meta_has_no_problems():
    assert protocol.verify_meta_selfconsistent(_meta(_splits())) == []


def test_selfconsistent_flags_wrong_protocol_version():
    m = {**_meta(_splits()), "protocol_version": 2}
    assert any("protocol_version" in p for p in protocol.verify_meta_selfconsistent(m))


def test_selfconsistent_flags_wrong_seed_and_split_order():
    m = _meta(_splits())
    assert any("seed" in p for p in protocol.verify_meta_selfconsistent({**m, "seed": 12345}))
    assert any("split_order" in p
               for p in protocol.verify_meta_selfconsistent({**m, "split_order": ["a", "b"]}))


def test_selfconsistent_flags_bad_hash_and_count_mismatch():
    m = _meta(_splits())
    assert any("content_hash" in p
               for p in protocol.verify_meta_selfconsistent({**m, "content_hash": "xyz"}))
    bad = {**m, "counts": {**m["counts"], "eval": 99}}
    assert any("counts[eval]" in p for p in protocol.verify_meta_selfconsistent(bad))


def test_selfconsistent_flags_overlapping_splits():
    m = _meta(_splits())
    m = {**m, "question_ids": {**m["question_ids"], "audit": ["q1"]}}  # q1 also in eval
    assert any("overlap" in p for p in protocol.verify_meta_selfconsistent(m))


# --------------------------------------------------------------------------- #
# verify_manifest (needs the actual items)
# --------------------------------------------------------------------------- #
def test_verify_manifest_clean():
    s = _splits()
    assert protocol.verify_manifest(_meta(s), s, expected_hash=_meta(s)["content_hash"]) == []


def test_verify_manifest_detects_tampered_question():
    s = _splits()
    meta = _meta(s)                     # hash over the ORIGINAL text
    s["eval"][0]["question_text"] = "TAMPERED"
    probs = protocol.verify_manifest(meta, s)
    assert any("content_hash" in p for p in probs)


def test_verify_manifest_detects_split_reshuffle():
    s = _splits()
    meta = _meta(s)
    moved = {"eval": s["eval"] + s["audit"], "audit": [], "live": s["live"]}  # q3 -> eval
    probs = protocol.verify_manifest(meta, moved)
    assert probs  # hash + counts + ids all disagree


def test_verify_manifest_flags_wrong_hash_txt():
    s = _splits()
    probs = protocol.verify_manifest(_meta(s), s, expected_hash="0" * 64)
    assert any("hash.txt" in p for p in probs)


# --------------------------------------------------------------------------- #
# CLI: self-consistency file mode + append
# --------------------------------------------------------------------------- #
def test_verify_meta_file(tmp_path):
    mp = tmp_path / "meta.json"
    mp.write_text(json.dumps(_meta(_splits())))
    assert vb.verify_meta_file(mp) == []
    mp.write_text(json.dumps({**_meta(_splits()), "seed": 1}))
    assert any("seed" in p for p in vb.verify_meta_file(mp))


def test_append_hash_is_idempotent(tmp_path):
    f = tmp_path / "benchmark_hashes.txt"
    assert vb.append_hash(f, "math500", "abc") is True
    assert vb.append_hash(f, "math500", "abc") is False   # already present
    assert vb.append_hash(f, "mmlu", "def") is True
    assert f.read_text().splitlines() == ["math500\tabc", "mmlu\tdef"]


# --------------------------------------------------------------------------- #
# Full encrypt -> verify -> tamper round-trip (needs cryptography)
# --------------------------------------------------------------------------- #
def _encrypt(data: dict, password: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce = os.urandom(16), os.urandom(12)
    ct = AESGCM(vb._derive_key(password, salt)).encrypt(nonce, json.dumps(data).encode("utf-8"), None)
    return base64.b64encode(salt + nonce + ct).decode()


def _write_benchmark(d: Path, splits: dict, password: str) -> dict:
    meta = _meta(splits)
    for name, fname in (("eval", "eval.json"), ("audit", "audit.json"), ("live", "live.json")):
        data = {"seed": protocol.SEALED_SEED, "count": len(splits[name]), "items": splits[name]}
        (d / fname).write_text(_encrypt(data, password))
    (d / "hash.txt").write_text(meta["content_hash"] + "\n")
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def test_full_verify_roundtrip_and_tamper(tmp_path):
    pytest.importorskip("cryptography")
    _write_benchmark(tmp_path, _splits(), "pw")
    assert vb.verify_dir(tmp_path, "pw") == []          # clean build verifies

    # tamper the eval split (change a question), re-encrypt just that file.
    bad = _splits()
    bad["eval"][0]["question_text"] = "TAMPERED"
    (tmp_path / "eval.json").write_text(
        _encrypt({"seed": protocol.SEALED_SEED, "count": 2, "items": bad["eval"]}, "pw")
    )
    probs = vb.verify_dir(tmp_path, "pw")
    assert probs and any("content_hash" in p for p in probs)


def test_full_verify_flags_count_and_seed_tamper(tmp_path):
    pytest.importorskip("cryptography")
    _write_benchmark(tmp_path, _splits(), "pw")
    # re-encrypt live.json with a wrong sealed seed and a lying count.
    (tmp_path / "live.json").write_text(
        _encrypt({"seed": 999, "count": 5, "items": _splits()["live"]}, "pw")
    )
    probs = vb.verify_dir(tmp_path, "pw")
    assert any("seed" in p for p in probs) and any("count" in p for p in probs)
