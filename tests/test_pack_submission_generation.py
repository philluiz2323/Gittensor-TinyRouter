"""Offline tests for pack_submission generation auto-detect (issue #187).

Pure ``pathlib`` — no torch / GPU / network. Guards against the old
``len(existing) + 1`` count, which silently overwrote an existing generation
whenever the numbering had a gap or a stray entry.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
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


# --------------------------------------------------------------------------- #
# best_theta fallback selection: latest generation by INTEGER, not string order
# ('best_theta_gen11' must beat 'best_theta_gen9'). Same integer-vs-string
# generation bug class as #262.
# --------------------------------------------------------------------------- #
_theta_generation = pack_submission._theta_generation
extract_head_and_svf = pack_submission.extract_head_and_svf


def test_theta_generation_orders_by_integer_not_string():
    names = ["best_theta_gen2.npy", "best_theta_gen9.npy", "best_theta_gen11.npy"]
    latest = max((Path(n) for n in names), key=_theta_generation)
    assert latest.name == "best_theta_gen11.npy"       # not the string-max gen9


def test_theta_generation_no_number_falls_back_to_name():
    # names without an encoded generation keep the previous lexicographic order.
    a, b = Path("best_theta.npy"), Path("best_theta_final.npy")
    assert _theta_generation(a) == (-1, "best_theta.npy")
    assert max((a, b), key=_theta_generation).name == "best_theta_final.npy"


def _write_theta(path: Path, value: float) -> None:
    """A valid-length θ filled with a marker so we can tell which file was loaded."""
    from trinity.coordinator import params as P
    spec = P.make_spec(n_a=6, d_h=1024, n_svf=P.DEFAULT_N_SVF)
    np.save(str(path), np.full(spec.n_total, value, dtype=np.float64))


def test_extract_prefers_canonical_best_theta(tmp_path: Path):
    _write_theta(tmp_path / "best_theta.npy", 1.0)
    _write_theta(tmp_path / "best_theta_gen9.npy", 9.0)
    head_W, _ = extract_head_and_svf(tmp_path)
    assert float(head_W[0, 0]) == pytest.approx(1.0)   # canonical wins when present


def test_extract_fallback_picks_latest_generation(tmp_path: Path):
    # No canonical file -> the glob fallback must pack gen 11, not the string-max gen 9.
    for gen in (2, 9, 11):
        _write_theta(tmp_path / f"best_theta_gen{gen}.npy", float(gen))
    head_W, _ = extract_head_and_svf(tmp_path)
    assert float(head_W[0, 0]) == pytest.approx(11.0)


def test_extract_no_theta_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_head_and_svf(tmp_path)
