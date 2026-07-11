"""Offline tests for YAML config validation. No network, no GPU."""
from __future__ import annotations

from pathlib import Path

from trinity.config_check import (
    check_config_dir,
    check_models_config,
    check_trinity_config,
)

_REPO = Path(__file__).resolve().parents[1]


def _models(**over):
    cfg = {
        "openrouter": {"base_url": "https://x/api", "api_key_env": "OPENROUTER_API_KEY",
                       "timeout_s": 180, "max_retries": 10, "max_concurrency": 8},
        "pool": [{"name": "a", "id": "v/a"}, {"name": "b", "id": "v/b"}],
        "decoding": {"worker": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 4096}},
    }
    cfg.update(over)
    return cfg


def _trinity(**over):
    cfg = {
        "coordinator": {
            "head": {"n_a": 6, "n_models": 3, "n_roles": 3},
            "svf": {"enabled": True, "matrices": ["q_proj", "k_proj"]},
        },
        "session": {"max_turns": 5},
        "sep_cmaes": {"population_size": 33, "mu": 16, "sigma0": 0.1,
                      "generations": 60, "m_cma": 16},
    }
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# The real repo configs must validate
# ---------------------------------------------------------------------------
def test_the_repo_configs_are_valid():
    report = check_config_dir(_REPO / "configs")
    assert report.ok, report.problems


# ---------------------------------------------------------------------------
# models.yaml
# ---------------------------------------------------------------------------
def test_valid_models_config_has_no_problems():
    assert check_models_config(_models()) == []


def test_duplicate_pool_name_is_flagged():
    cfg = _models(pool=[{"name": "a", "id": "v/a"}, {"name": "a", "id": "v/b"}])
    probs = check_models_config(cfg)
    assert any("duplicate pool name" in p for p in probs)


def test_duplicate_pool_id_is_flagged():
    cfg = _models(pool=[{"name": "a", "id": "v/x"}, {"name": "b", "id": "v/x"}])
    assert any("duplicate pool id" in p for p in check_models_config(cfg))


def test_missing_pool_entry_fields_are_flagged():
    cfg = _models(pool=[{"name": "a"}, {"id": "v/b"}])
    probs = check_models_config(cfg)
    assert any("missing an 'id'" in p for p in probs)
    assert any("missing a 'name'" in p for p in probs)


def test_empty_pool_is_flagged():
    assert any("non-empty list" in p for p in check_models_config(_models(pool=[])))


def test_missing_openrouter_keys_are_flagged():
    cfg = _models(openrouter={"base_url": "https://x"})
    assert any("api_key_env" in p for p in check_models_config(cfg))


def test_bad_decoding_ranges_are_flagged():
    cfg = _models(decoding={"worker": {"temperature": 5, "top_p": 0, "max_tokens": -1}})
    probs = check_models_config(cfg)
    assert any("temperature" in p for p in probs)
    assert any("top_p" in p for p in probs)
    assert any("max_tokens" in p for p in probs)


# ---------------------------------------------------------------------------
# trinity.yaml
# ---------------------------------------------------------------------------
def test_valid_trinity_config_has_no_problems():
    assert check_trinity_config(_trinity()) == []


def test_head_action_space_cross_check():
    # n_a must equal n_models + n_roles.
    cfg = _trinity(coordinator={"head": {"n_a": 7, "n_models": 3, "n_roles": 3},
                                "svf": {"enabled": False}})
    assert any("n_a" in p and "must equal" in p for p in check_trinity_config(cfg))


def test_mu_above_population_is_flagged():
    cfg = _trinity(sep_cmaes={"population_size": 10, "mu": 20, "sigma0": 0.1})
    assert any("mu" in p for p in check_trinity_config(cfg))


def test_nonpositive_sigma_is_flagged():
    cfg = _trinity(sep_cmaes={"population_size": 10, "mu": 5, "sigma0": 0})
    assert any("sigma0" in p for p in check_trinity_config(cfg))


def test_enabled_svf_needs_matrices():
    cfg = _trinity(coordinator={"head": {"n_a": 6, "n_models": 3, "n_roles": 3},
                                "svf": {"enabled": True, "matrices": []}})
    assert any("svf.matrices" in p for p in check_trinity_config(cfg))


def test_bad_max_turns_is_flagged():
    assert any("max_turns" in p for p in check_trinity_config(_trinity(session={"max_turns": 0})))


# ---------------------------------------------------------------------------
# check_config_dir
# ---------------------------------------------------------------------------
def test_missing_file_is_reported_not_raised(tmp_path):
    report = check_config_dir(tmp_path)
    assert not report.ok
    assert any("file not found" in p for p in report.problems)


def test_unparseable_yaml_is_reported(tmp_path):
    (tmp_path / "models.yaml").write_text("pool: [unclosed")
    (tmp_path / "trinity.yaml").write_text("session: {max_turns: 5}")
    report = check_config_dir(tmp_path)
    assert any("could not parse YAML" in p for p in report.problems)


def test_report_to_dict():
    from trinity.config_check import ConfigReport

    r = ConfigReport(problems=["x: bad"])
    assert r.to_dict() == {"ok": False, "n_problems": 1, "problems": ["x: bad"]}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
