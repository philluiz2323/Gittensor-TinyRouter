"""Offline tests for the verify_benchmark CLI entry point (issue #191).

``verify_dir`` reports a missing ``meta.json`` as a clean problem, but ``main()``
used to re-read ``meta.json`` unconditionally before the failure branch, so a build
missing its manifest crashed with ``FileNotFoundError`` instead of reporting it.
These drive ``main()`` directly (the pure helpers are covered elsewhere).

Pure ``pathlib``/``json`` — no password, no decryption, no torch/network: the
missing-manifest and self-consistency paths return before any AES-GCM work.
"""
from __future__ import annotations

import importlib.util
import json
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


vb = _load("verify_benchmark")
protocol = _load("benchmark_protocol")


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["verify_benchmark.py", *argv])
    vb.main()


def test_missing_meta_reports_cleanly_not_crash(monkeypatch, capsys, tmp_path):
    # A build directory with no meta.json: verify_dir returns ["missing meta.json"],
    # and main() must print a FAIL report and exit 1 — never raise FileNotFoundError.
    bench = tmp_path / "math500"
    bench.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["--dir", str(bench), "--password", "pw"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "missing meta.json" in out


def test_dir_without_password_exits_2(monkeypatch, capsys, tmp_path):
    bench = tmp_path / "math500"
    bench.mkdir()
    monkeypatch.delenv("BENCHMARK_PASSWORD", raising=False)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["--dir", str(bench)])
    assert exc.value.code == 2


def test_no_mode_exits_2(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, [])
    assert exc.value.code == 2


def test_self_consistency_bad_meta_reports_cleanly(monkeypatch, capsys, tmp_path):
    # A syntactically-valid but protocol-inconsistent meta.json should FAIL-report,
    # exercising the --meta branch of main() through to the problem print.
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"benchmark": "math500"}))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["--meta", str(meta_path)])
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_append_requires_full_dir_verification(monkeypatch, capsys, tmp_path):
    splits = {
        "eval": [{"question_id": "q1", "benchmark": "math500", "task_type": "math",
                  "question_text": "1+1?", "correct_answer": "2", "model_answers": {}}],
        "audit": [],
        "live": [],
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(protocol.build_manifest(
        "math500", splits, seed=protocol.SEALED_SEED, created_at="t",
    )))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["--meta", str(meta_path), "--append", str(tmp_path / "hashes.txt")])
    assert exc.value.code == 2
    assert "full --dir" in capsys.readouterr().out
