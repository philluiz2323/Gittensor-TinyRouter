"""Offline consistency check for the pool's price tables and membership lists.

Why this exists
---------------
The repo keeps the model pool's pricing and membership in **five** places that must
agree, each carrying a hand-written "keep in sync" comment but with **nothing that
enforces it**:

- ``configs/models.yaml`` ``pool`` — the runtime source of truth for membership.
- ``trinity.llm.openrouter_pricing.OPENROUTER_POOL_PRICES`` — the self-declared "single
  source of truth" for per-model rates (``scripts/cost_report.py`` and the
  ``pack_submission`` receipt price ledgers through it).
- ``scripts/oracle_ceiling.py::_DEFAULT_PRICES`` — a duplicate price table.
- ``trinity.fugu.cost.PRICES`` — a duplicate price table.
- ``trinity.submission.constants.DEFAULT_POOL_MODELS`` — the membership gate 6 checks a
  submission receipt's ``pool_models`` against (``receipt_pool_models_invalid``).

When these drift, the failure is silent and costly: the three price tables each fall back
*differently* on an unknown model (blended mean / ``(0, 0)`` / a missing key), so a lagging
table makes tools disagree instead of erroring; a lagging ``DEFAULT_POOL_MODELS``
false-rejects every honest submission at gate 6; and a lagging price makes the packed
receipt cost wrong, which then trips the min-cost / ledger-reconcile gates. The
``models.yaml`` header even points maintainers at ``cost_report.py`` to add prices —
which is now only an *alias* of ``OPENROUTER_POOL_PRICES``, so following the doc edits the
wrong place.

``config_check`` validates the *within-YAML* structure of the configs; it never looks at
the Python-side price tables. This module fills that gap: it flags membership mismatches,
missing/stale price entries, cross-table price disagreements, and invalid (non-finite or
non-positive) prices. On the current repo every source agrees, so it passes today and only
fires on future drift.

Pure / deterministic / no network / no GPU / no torch. Only the stdlib + PyYAML. The core
:func:`check_pool_consistency` is pure over injected data; :func:`gather_sources` does the
(lazy) loading so importing this module pulls in nothing heavy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PriceTable",
    "PoolSources",
    "PoolConsistencyReport",
    "check_pool_consistency",
    "gather_sources",
    "render",
]

# Canonical price-table label (the self-declared single source of truth).
CANONICAL_LABEL = "openrouter_pricing.OPENROUTER_POOL_PRICES"
_YAML_LABEL = "configs/models.yaml pool"
_DPM_LABEL = "submission.constants.DEFAULT_POOL_MODELS"


@dataclass(frozen=True)
class PriceTable:
    """A named ``model -> (price_in, price_out)`` table (one of the duplicated copies)."""

    name: str
    prices: dict[str, tuple[float, float]]


@dataclass
class PoolSources:
    """The five pool-defining sources, gathered for a consistency check.

    ``yaml_pool`` is the membership source of truth (falls back to ``default_pool_models``
    if empty). ``canonical`` is ``OPENROUTER_POOL_PRICES``; ``duplicates`` are the other
    price tables that must match it.
    """

    yaml_pool: list[str]
    default_pool_models: list[str]
    canonical: PriceTable
    duplicates: list[PriceTable] = field(default_factory=list)

    @property
    def price_tables(self) -> list[PriceTable]:
        """The canonical table followed by every duplicate."""
        return [self.canonical, *self.duplicates]


@dataclass
class PoolConsistencyReport:
    """Problems found across the pool sources (empty ``problems`` == consistent)."""

    problems: list[str] = field(default_factory=list)
    pool: list[str] = field(default_factory=list)          # membership source of truth
    tables_checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff every source agrees."""
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "ok": self.ok,
            "n_problems": len(self.problems),
            "problems": list(self.problems),
            "pool": list(self.pool),
            "tables_checked": list(self.tables_checked),
        }


def _is_finite_positive(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _valid_pair(pair: Any) -> bool:
    return (isinstance(pair, (tuple, list)) and len(pair) == 2
            and _is_finite_positive(pair[0]) and _is_finite_positive(pair[1]))


def _dups(names: list[str]) -> list[str]:
    return sorted({n for n in names if names.count(n) > 1})


def _check_membership(sources: PoolSources, pool: list[str], problems: list[str]) -> None:
    """Membership lists must agree as sets (gate 6 compares ``pool_models`` as a set)."""
    dpm = sources.default_pool_models
    dup_yaml, dup_dpm = _dups(sources.yaml_pool), _dups(dpm)
    if dup_yaml:
        problems.append(f"{_YAML_LABEL}: duplicate pool name(s): {dup_yaml}")
    if dup_dpm:
        problems.append(f"{_DPM_LABEL}: duplicate model(s): {dup_dpm}")
    missing = sorted(set(pool) - set(dpm))
    extra = sorted(set(dpm) - set(pool))
    if missing:
        problems.append(f"{_DPM_LABEL} is missing pool model(s) {missing} — gate 6 would "
                        "reject an honest submission (receipt_pool_models_invalid)")
    if extra:
        problems.append(f"{_DPM_LABEL} lists non-pool model(s) {extra} not in {_YAML_LABEL}")


def _check_table_membership(pool: list[str], table: PriceTable, problems: list[str]) -> None:
    keys = set(table.prices)
    missing = sorted(set(pool) - keys)
    extra = sorted(keys - set(pool))
    if missing:
        problems.append(f"{table.name} is missing price(s) for pool model(s): {missing}")
    if extra:
        problems.append(f"{table.name} has stale/extra price entr(ies) not in the pool: {extra}")


def _check_price_validity(table: PriceTable, problems: list[str]) -> None:
    for model, pair in table.prices.items():
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            problems.append(f"{table.name}[{model!r}] must be an (in, out) price pair; got {pair!r}")
            continue
        for label, value in (("in", pair[0]), ("out", pair[1])):
            if not _is_finite_positive(value):
                problems.append(f"{table.name}[{model!r}] {label} price must be finite and "
                                f"> 0; got {value!r}")


def _check_price_agreement(pool: list[str], tables: list[PriceTable], problems: list[str]) -> None:
    """Every table that prices a pool model must agree on ``(in, out)`` (to 1e-6)."""
    for model in pool:
        by_price: dict[tuple[float, float], list[str]] = {}
        for t in tables:
            pair = t.prices.get(model)
            if pair is not None and _valid_pair(pair):
                key = (round(float(pair[0]), 6), round(float(pair[1]), 6))
                by_price.setdefault(key, []).append(t.name)
        if len(by_price) > 1:
            desc = "; ".join(f"{list(k)} in [{', '.join(v)}]" for k, v in sorted(by_price.items()))
            problems.append(f"price disagreement for {model!r}: {desc}")


def check_pool_consistency(sources: PoolSources) -> PoolConsistencyReport:
    """Cross-check the pool's membership lists and price tables; return a report.

    The membership source of truth is ``sources.yaml_pool`` (or ``default_pool_models`` if
    the YAML pool is empty). Checks: membership agreement (yaml pool vs
    ``DEFAULT_POOL_MODELS``, plus intra-list duplicates); each price table covers exactly
    the pool (no missing/stale entries); every price is finite and positive; and all tables
    agree on each model's ``(in, out)`` rate. Never raises.
    """
    pool = list(sources.yaml_pool) or list(sources.default_pool_models)
    tables = sources.price_tables
    report = PoolConsistencyReport(pool=pool, tables_checked=[t.name for t in tables])
    problems = report.problems

    if not pool:
        problems.append(f"no pool membership found ({_YAML_LABEL} and {_DPM_LABEL} both empty)")
        return report

    _check_membership(sources, pool, problems)
    for table in tables:
        _check_table_membership(pool, table, problems)
        _check_price_validity(table, problems)
    _check_price_agreement(pool, tables, problems)
    return report


def _load_yaml_pool(models_yaml: Path) -> list[str]:
    import yaml

    doc = yaml.safe_load(models_yaml.read_text())
    pool = doc.get("pool") if isinstance(doc, dict) else None
    if not isinstance(pool, list):
        return []
    return [str(e["name"]) for e in pool if isinstance(e, dict) and e.get("name")]


def _load_script_attr(path: Path, module_name: str, attr: str) -> Any:
    """Import a stand-alone script by path (registered so dataclasses resolve) and read an attr."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, attr, None)


def gather_sources(repo_root: str | Path) -> PoolSources:
    """Load the five pool sources from a repo checkout (lazy imports; never at module load).

    ``configs/models.yaml`` is parsed; the price tables and ``DEFAULT_POOL_MODELS`` are
    imported from their modules; ``_DEFAULT_PRICES`` is read from the ``oracle_ceiling.py``
    script by path. A source that cannot be loaded contributes an empty table (which the
    consistency check then flags as a mismatch, rather than crashing).
    """
    root = Path(repo_root)
    from trinity.fugu.cost import PRICES as fugu_prices
    from trinity.llm.openrouter_pricing import OPENROUTER_POOL_PRICES as canonical_prices
    from trinity.submission.constants import DEFAULT_POOL_MODELS as default_pool

    oc_prices = _load_script_attr(
        root / "scripts" / "oracle_ceiling.py", "oracle_ceiling", "_DEFAULT_PRICES"
    ) or {}

    return PoolSources(
        yaml_pool=_load_yaml_pool(root / "configs" / "models.yaml"),
        default_pool_models=[str(m) for m in default_pool],
        canonical=PriceTable(CANONICAL_LABEL, dict(canonical_prices)),
        duplicates=[
            PriceTable("oracle_ceiling._DEFAULT_PRICES", dict(oc_prices)),
            PriceTable("fugu.cost.PRICES", dict(fugu_prices)),
        ],
    )


def render(report: PoolConsistencyReport) -> str:
    """Markdown status report: the pool, the tables checked, and any drift found."""
    out = ["# Pool price/membership consistency\n"]
    out.append(f"pool ({len(report.pool)}): {', '.join(report.pool) or '(none)'}")
    out.append(f"price tables checked: {', '.join(report.tables_checked) or '(none)'}\n")
    if report.ok:
        out.append("**OK** — every pool source (membership + all price tables) agrees.")
    else:
        out.append(f"**DRIFT** — {len(report.problems)} problem(s):")
        out.extend(f"  - {p}" for p in report.problems)
    return "\n".join(out) + "\n"
