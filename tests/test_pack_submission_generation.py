"""Offline tests for pack_submission generation auto-detect (issue #187).

Pure ``pathlib`` — no torch / GPU / network. Guards against the old
``len(existing) + 1`` count, which silently overwrote an existing generation
whenever the numbering had a gap or a stray entry.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "pack_submission", _REPO / "scripts" / "pack_submission.py"
)
pack_submission = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pack_submission)
next_generation = pack_submission.next_generation


def test_missing_dir_is_generation_one(tmp_path: Path) -> None:
    assert next_generation(tmp_path / "alice") == 1


def test_empty_dir_is_generation_one(tmp_path: Path) -> None:
    d = tmp_path / "alice"
    d.mkdir()
    assert next_generation(d) == 1


def test_contiguous_generations(tmp_path: Path) -> None:
    d = tmp_path / "alice"
    for g in ("1", "2", "3"):
        (d / g).mkdir(parents=True)
    assert next_generation(d) == 4


def test_gap_does_not_overwrite_existing_generation(tmp_path: Path) -> None:
    # gens 1 and 3 present (gen 2 was thrown out). Counting entries would give
    # 2 + 1 == 3 and clobber the real gen 3; max+1 must give 4.
    d = tmp_path / "alice"
    (d / "1").mkdir(parents=True)
    (d / "3").mkdir(parents=True)
    assert next_generation(d) == 4


def test_ignores_non_numeric_and_file_entries(tmp_path: Path) -> None:
    d = tmp_path / "alice"
    (d / "1").mkdir(parents=True)
    (d / "2").mkdir()
    (d / "README.md").write_text("notes")  # stray file
    (d / ".DS_Store").write_text("")        # stray hidden file
    (d / "draft").mkdir()                    # non-numeric dir
    assert next_generation(d) == 3


def test_single_high_generation(tmp_path: Path) -> None:
    # Only gen 7 kept around; next is 8, never 2.
    d = tmp_path / "alice"
    (d / "7").mkdir(parents=True)
    assert next_generation(d) == 8
