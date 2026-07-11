"""Offline structural validation of the repo's YAML configs.

Why this exists
---------------
``configs/models.yaml`` and ``configs/trinity.yaml`` drive every run, but a typo
in them fails late — often only after a GPU is up or a paid API call is made
(a duplicate pool ``name`` silently shadows a model; ``head.n_a`` that disagrees
with ``n_models + n_roles`` mis-sizes the CMA-ES search vector; ``mu`` above the
population is an invalid recombination). The submission-side ``validate_*``
helpers cover packs and receipts, but nothing checks the run configs themselves.

This module validates the self-contained structural invariants offline: required
keys, types, value ranges, uniqueness, and the cross-field relationships the code
relies on. It returns a list of human-readable problems (empty means valid) and
never raises on a bad config — a validator that crashes on the thing it validates
is useless.

Pure / deterministic / no network / no GPU / no torch. Only the stdlib + PyYAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeGuard

__all__ = [
    "ConfigReport",
    "check_models_config",
    "check_trinity_config",
    "check_config_dir",
]

# Decoding roles that, if present, must carry sane sampling params.
_DECODING_ROLES = ("thinker", "worker", "verifier")


@dataclass
class ConfigReport:
    """The problems found in one or more config files (empty == valid)."""

    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no problems were found."""
        return not self.problems

    def extend(self, prefix: str, problems: list[str]) -> None:
        """Absorb another check's problems, tagged with a source ``prefix``."""
        self.problems.extend(f"{prefix}: {p}" for p in problems)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {"ok": self.ok, "n_problems": len(self.problems), "problems": list(self.problems)}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_decoding(decoding: Any, problems: list[str]) -> None:
    if decoding is None:
        return
    if not isinstance(decoding, Mapping):
        problems.append("'decoding' must be a mapping of role -> params")
        return
    for role in _DECODING_ROLES:
        block = decoding.get(role)
        if block is None:
            continue  # roles are optional; only validate what is present
        if not isinstance(block, Mapping):
            problems.append(f"decoding.{role} must be a mapping")
            continue
        temp = block.get("temperature")
        if temp is not None and (not _is_number(temp) or not 0.0 <= temp <= 2.0):
            problems.append(f"decoding.{role}.temperature must be in [0, 2]; got {temp!r}")
        top_p = block.get("top_p")
        if top_p is not None and (not _is_number(top_p) or not 0.0 < top_p <= 1.0):
            problems.append(f"decoding.{role}.top_p must be in (0, 1]; got {top_p!r}")
        mt = block.get("max_tokens")
        if mt is not None and (not isinstance(mt, int) or isinstance(mt, bool) or mt <= 0):
            problems.append(f"decoding.{role}.max_tokens must be a positive int; got {mt!r}")


def check_models_config(cfg: Any) -> list[str]:
    """Validate a parsed ``models.yaml``; return a list of problems (empty == ok).

    Checks: the ``openrouter`` block has ``base_url`` + ``api_key_env`` and
    positive numeric ``timeout_s`` / ``max_retries`` / ``max_concurrency`` when
    present; ``pool`` is a non-empty list whose entries each have a ``name`` and
    ``id``, with **unique** names and ids (a duplicate silently shadows a model);
    and any ``decoding`` role block carries sane sampling params.
    """
    problems: list[str] = []
    if not isinstance(cfg, Mapping):
        return ["top-level config must be a mapping"]

    orc = cfg.get("openrouter")
    if not isinstance(orc, Mapping):
        problems.append("missing 'openrouter' block")
    else:
        for key in ("base_url", "api_key_env"):
            if not orc.get(key):
                problems.append(f"openrouter.{key} is required and must be non-empty")
        for key in ("timeout_s", "max_retries", "max_concurrency"):
            v = orc.get(key)
            if v is not None and (not _is_number(v) or v <= 0):
                problems.append(f"openrouter.{key} must be a positive number; got {v!r}")

    pool = cfg.get("pool")
    if not isinstance(pool, list) or not pool:
        problems.append("'pool' must be a non-empty list of models")
    else:
        names: list[str] = []
        ids: list[str] = []
        for i, entry in enumerate(pool):
            if not isinstance(entry, Mapping):
                problems.append(f"pool[{i}] must be a mapping with 'name' and 'id'")
                continue
            name, mid = entry.get("name"), entry.get("id")
            if not name:
                problems.append(f"pool[{i}] is missing a 'name'")
            else:
                names.append(str(name))
            if not mid:
                problems.append(f"pool[{i}] ({name!r}) is missing an 'id'")
            else:
                ids.append(str(mid))
        for label, values in (("name", names), ("id", ids)):
            dups = sorted({v for v in values if values.count(v) > 1})
            if dups:
                problems.append(f"duplicate pool {label}(s): {dups}")

    _check_decoding(cfg.get("decoding"), problems)
    return problems


def check_trinity_config(cfg: Any) -> list[str]:
    """Validate a parsed ``trinity.yaml``; return a list of problems (empty == ok).

    Checks the self-contained cross-field invariants the code relies on:
    ``head.n_a == n_models + n_roles`` (the action-space size that sizes the head
    and the CMA-ES vector); ``svf.matrices`` non-empty when SVF is enabled;
    ``sep_cmaes`` sanity (``0 < mu <= population_size``, positive ``sigma0`` /
    ``generations`` / ``m_cma``); and ``session.max_turns >= 1``.
    """
    problems: list[str] = []
    if not isinstance(cfg, Mapping):
        return ["top-level config must be a mapping"]

    head = cfg.get("coordinator", {}).get("head") if isinstance(cfg.get("coordinator"), Mapping) else None
    if isinstance(head, Mapping):
        n_a, n_models, n_roles = head.get("n_a"), head.get("n_models"), head.get("n_roles")
        # Narrow each individually so the arithmetic below is well-typed.
        if (_is_int(n_a) and _is_int(n_models) and _is_int(n_roles)
                and n_a != n_models + n_roles):
            problems.append(
                f"head.n_a ({n_a}) must equal n_models + n_roles "
                f"({n_models} + {n_roles} = {n_models + n_roles})"
            )

    svf = cfg.get("coordinator", {}).get("svf") if isinstance(cfg.get("coordinator"), Mapping) else None
    if isinstance(svf, Mapping) and svf.get("enabled"):
        mats = svf.get("matrices")
        if not isinstance(mats, list) or not mats:
            problems.append("svf.enabled is true but svf.matrices is empty")

    cma = cfg.get("sep_cmaes")
    if isinstance(cma, Mapping):
        pop, mu = cma.get("population_size"), cma.get("mu")
        if isinstance(pop, int) and pop <= 0:
            problems.append(f"sep_cmaes.population_size must be positive; got {pop}")
        if isinstance(mu, int) and isinstance(pop, int) and not (0 < mu <= pop):
            problems.append(f"sep_cmaes.mu ({mu}) must satisfy 0 < mu <= population_size ({pop})")
        for key in ("sigma0", "generations", "m_cma"):
            v = cma.get(key)
            if v is not None and (not _is_number(v) or v <= 0):
                problems.append(f"sep_cmaes.{key} must be positive; got {v!r}")

    session = cfg.get("session")
    if isinstance(session, Mapping):
        mt = session.get("max_turns")
        if mt is not None and (not isinstance(mt, int) or isinstance(mt, bool) or mt < 1):
            problems.append(f"session.max_turns must be an int >= 1; got {mt!r}")

    return problems


def check_config_dir(configs_dir: str | Path) -> ConfigReport:
    """Validate ``models.yaml`` and ``trinity.yaml`` under ``configs_dir``.

    A missing or unparseable file is itself a reported problem. Returns a
    :class:`ConfigReport`; never raises.
    """
    import yaml

    report = ConfigReport()
    root = Path(configs_dir)
    for name, checker in (("models.yaml", check_models_config),
                          ("trinity.yaml", check_trinity_config)):
        path = root / name
        if not path.exists():
            report.problems.append(f"{name}: file not found at {path}")
            continue
        try:
            cfg = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            report.problems.append(f"{name}: could not parse YAML ({exc})")
            continue
        report.extend(name, checker(cfg))
    return report
