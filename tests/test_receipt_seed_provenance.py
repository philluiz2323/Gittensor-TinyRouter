"""Seed provenance from training run to submission receipt (issue #109).

`trinity.train` records the run's seed in `summary.json`, and
`scripts/pack_submission.build_receipt` surfaces it in `receipt.json` rather than
silently defaulting to 0. No GPU / no network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.train import build_summary  # noqa: E402


def _load_pack_submission():
    spec = importlib.util.spec_from_file_location(
        "pack_submission", _REPO / "scripts" / "pack_submission.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pack_submission"] = mod
    spec.loader.exec_module(mod)
    return mod


def _summary_kwargs(**overrides):
    kwargs = dict(
        benchmark="math500",
        pool_models=["a", "b", "c"],
        n_total=13312,
        popsize=33,
        m_cma=16,
        generations=3,
        best_fitness=0.42,
        seed=7,
        run_dir="experiments/math500/demo",
    )
    kwargs.update(overrides)
    return kwargs


def _write_run(tmp_path: Path, summary: dict | None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "generation": g,
                    "gen_mean_fitness": 0.1 * g,
                    "gen_max_fitness": 0.2 * g,
                    "best_fitness": 0.2 * g,
                }
                for g in range(1, 4)
            ]
        )
    )
    return run_dir


# ---- train.py records the seed ---------------------------------------------
def test_build_summary_records_the_seed():
    assert build_summary(**_summary_kwargs(seed=7))["seed"] == 7


def test_build_summary_coerces_seed_to_int():
    seed = build_summary(**_summary_kwargs(seed=True))["seed"]
    assert seed == 1 and isinstance(seed, int)


def test_build_summary_is_json_serialisable_with_all_documented_keys():
    summary = build_summary(**_summary_kwargs())
    roundtripped = json.loads(json.dumps(summary))
    assert set(roundtripped) == {
        "benchmark", "pool", "n_total", "popsize", "m_cma",
        "generations", "best_fitness", "seed", "run_dir",
    }


# ---- pack_submission surfaces it -------------------------------------------
def test_receipt_seed_round_trips_from_summary(tmp_path):
    pack = _load_pack_submission()
    run_dir = _write_run(tmp_path, build_summary(**_summary_kwargs(seed=7)))
    assert pack.build_receipt(run_dir, "math500")["seed"] == 7


def test_receipt_preserves_an_explicit_zero_seed(tmp_path):
    pack = _load_pack_submission()
    run_dir = _write_run(tmp_path, build_summary(**_summary_kwargs(seed=0)))
    assert pack.build_receipt(run_dir, "math500")["seed"] == 0


def test_legacy_summary_without_seed_records_null_not_zero(tmp_path, capsys):
    """A run predating seed recording is honestly unknown, not a plausible 0."""
    pack = _load_pack_submission()
    legacy = build_summary(**_summary_kwargs())
    legacy.pop("seed")
    run_dir = _write_run(tmp_path, legacy)

    receipt = pack.build_receipt(run_dir, "math500")

    assert receipt["seed"] is None
    assert json.loads(json.dumps(receipt))["seed"] is None
    assert "no 'seed'" in capsys.readouterr().err


def test_missing_summary_records_null_seed(tmp_path, capsys):
    pack = _load_pack_submission()
    run_dir = _write_run(tmp_path, None)
    assert pack.build_receipt(run_dir, "math500")["seed"] is None
    assert "no 'seed'" in capsys.readouterr().err


def test_resolve_seed_helper_is_pure(capsys):
    pack = _load_pack_submission()
    assert pack._resolve_seed({"seed": 11}) == 11
    assert pack._resolve_seed({"seed": 0}) == 0
    assert pack._resolve_seed({}) is None
    capsys.readouterr()
