"""Reachability levels L0/L1/L2 for the oracle ceiling (ORACLE §2.2, §6, §9).

``docs/ORACLE_CEILING_DIAGNOSTIC.md`` §2.2 names the diagnostic's **false-negative**
failure mode: TRINITY is not "pick one model", it is ``(model, role)`` decisions over up
to 5 turns, so an oracle computed from single-turn Worker correctness alone is a *strict
lower bound* on the reachable ceiling. A thin gap measured that way can read as "routing
is hopeless" when multi-turn collaboration would in fact have helped. §9's defense table
lists the fix as *"L0/L1/L2 reachability; L0 is a lower bound"*.

That fix was never built, and the shipped code says so in two places:

* ``scripts/oracle_ceiling.py`` exposes ``--level`` with ``choices=["L0"]`` and the help
  text *"L1/L2 are future"*.
* Its verdict's INCONCLUSIVE branch advises *"Widen the reachability level (L1/L2) or
  collect more samples"* — advice nothing can currently act on.

Meanwhile §6 defines the deciding quantity as ``H = routing_headroom`` measured **"at the
widest reachability level (L2 if run, else L1)"**. Since neither can be run, every verdict
today rests on L0 while the decision rule believes it is reading something wider. This
module supplies the missing layer.

**The three levels** (§2.2):

======  =======================================================================
Level   What it measures
======  =======================================================================
``L0``  single-turn Worker — cheapest; a strict lower bound on the ceiling
``L1``  best-role-per-model — each model as a single answerer in any role
``L2``  short multi-turn probe — sampled ``(model, role)`` sequences within the
        5-turn budget; **sampled, not exhaustive**, so it carries its own CI
======  =======================================================================

**What is and is not monotone — this matters for the guard.** Each level's option set
*contains* the one below it (L1 lets a model answer in any role, Worker included; L2
allows sequences that include the single-turn cases). A max over a superset can only grow,
so ``routing_oracle`` is **monotone non-decreasing** in the level, and so is
``best_single``. The *headroom*, being their difference, is **not** monotone — widening
reachability can lift the best single answerer as much as it lifts the ceiling.
:func:`analyze` therefore checks monotonicity of the **oracle** (a violation means a
measurement bug, since it is mathematically impossible) and never assumes the headroom
moves in a particular direction. §2.2's phrase *"only rules out routing if L1 and L2 are
also thin"* is implemented literally, by reading each measured level's own headroom, not
by extrapolating from L0's.

**Scope.** This is the *analysis* layer and is fully offline: it consumes per-level
``oracle_matrix`` artifacts and reuses the canonical
:func:`trinity.analysis.union_oracle.oracle_from_matrix` decoder, so it cannot drift from
the shipped schema. Actually *collecting* an L1 or L2 matrix needs live role-loop /
trajectory calls, exactly as collecting L0 already does; that is out of scope here and
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trinity.analysis.union_oracle import BenchmarkOracle, oracle_from_matrix

__all__ = [
    "LEVEL_ORDER",
    "LEVEL_DESCRIPTIONS",
    "THIN_HEADROOM",
    "LevelOracle",
    "ReachabilitySummary",
    "analyze",
    "render",
]

#: Reachability levels, narrowest first (ORACLE §2.2).
LEVEL_ORDER: tuple[str, ...] = ("L0", "L1", "L2")

LEVEL_DESCRIPTIONS: dict[str, str] = {
    "L0": "single-turn Worker (strict lower bound)",
    "L1": "best-role-per-model (single answerer, any role)",
    "L2": "short multi-turn probe (sampled, carries its own CI)",
}

#: ORACLE §6: headroom is "small" when its CI upper bound is <= ~0.02.
THIN_HEADROOM = 0.02

#: Levels that are sampled rather than exhaustive, so a point estimate alone is not
#: enough to support a verdict (§2.2 fix 1: "reported with its own CI").
_SAMPLED_LEVELS = frozenset({"L2"})


@dataclass(frozen=True)
class LevelOracle:
    """One reachability level's oracle, plus whether it can carry a verdict."""

    level: str
    oracle: BenchmarkOracle
    ci: tuple[float, float] | None
    is_sampled: bool

    @property
    def headroom(self) -> float:
        """``routing_oracle - best_single`` at this level."""
        return self.oracle.headroom

    @property
    def routing_oracle(self) -> float:
        return self.oracle.routing_oracle

    @property
    def is_thin(self) -> bool:
        """Thin per §6: CI upper bound <= 0.02 if a CI is known, else the point estimate.

        Falling back to the point estimate is deliberately the *less* conservative read,
        so :attr:`verdict_supported` refuses to call a sampled level thin without a CI.
        """
        upper = self.ci[1] if self.ci is not None else self.headroom
        return upper <= THIN_HEADROOM

    @property
    def verdict_supported(self) -> bool:
        """Can this level support a 'routing is hopeless' claim on its own?

        A sampled level (L2) without a CI cannot: §2.2 requires it be reported with one,
        because a point estimate from a handful of sequences is not evidence of absence.
        """
        return not (self.is_sampled and self.ci is None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "level": self.level,
            "description": LEVEL_DESCRIPTIONS.get(self.level, ""),
            "routing_oracle": self.routing_oracle,
            "best_single": self.oracle.best_single,
            "headroom": self.headroom,
            "ci_95": list(self.ci) if self.ci is not None else None,
            "is_sampled": self.is_sampled,
            "is_thin": self.is_thin,
            "verdict_supported": self.verdict_supported,
            "oracle": self.oracle.to_dict(),
        }


@dataclass(frozen=True)
class ReachabilitySummary:
    """Per-level oracles + the §6 widest-level reading and its integrity guards."""

    levels: list[LevelOracle]
    widest_level: str | None
    widest_headroom: float | None
    widest_ci: tuple[float, float] | None
    monotonicity_violations: list[str]
    can_rule_out_routing: bool
    verdict: str
    message: str

    @property
    def levels_measured(self) -> list[str]:
        return [lv.level for lv in self.levels]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "levels_measured": self.levels_measured,
            "widest_level": self.widest_level,
            "widest_headroom": self.widest_headroom,
            "widest_ci_95": list(self.widest_ci) if self.widest_ci is not None else None,
            "monotonicity_violations": list(self.monotonicity_violations),
            "can_rule_out_routing": self.can_rule_out_routing,
            "verdict": self.verdict,
            "message": self.message,
            "levels": [lv.to_dict() for lv in self.levels],
        }


def _level_key(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        raise ValueError(
            f"unknown reachability level {level!r}; expected one of {list(LEVEL_ORDER)}"
        ) from None


def _coerce_ci(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    lo: Any
    hi: Any
    if isinstance(raw, Mapping):
        lo, hi = raw.get("ci_lo"), raw.get("ci_hi")
        if lo is None or hi is None:
            raise ValueError(
                f"CI mapping must carry both 'ci_lo' and 'ci_hi', got {raw!r}"
            )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        lo, hi = raw[0], raw[1]
    else:
        raise ValueError(f"CI must be a (lo, hi) pair or a mapping, got {raw!r}")
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        raise ValueError(f"CI bounds must be numbers, got {raw!r}") from None
    if hi_f < lo_f:
        raise ValueError(f"CI bounds are inverted: lo={lo_f} > hi={hi_f}")
    return (lo_f, hi_f)


def analyze(
    matrices_by_level: Mapping[str, dict],
    *,
    threshold: float = 0.5,
    cis: Mapping[str, Any] | None = None,
) -> ReachabilitySummary:
    """Compute the oracle at each supplied reachability level and apply §6's rule.

    Parameters
    ----------
    matrices_by_level:
        ``{level: oracle_matrix}`` for any subset of :data:`LEVEL_ORDER`. Each matrix uses
        the canonical ``oracle_matrix`` schema and is decoded by the merged
        :func:`~trinity.analysis.union_oracle.oracle_from_matrix`, so an L1 matrix simply
        carries composite ``"model/role"`` keys in ``per_model`` — no new schema.
    threshold:
        Solve-probability threshold, forwarded to the canonical decoder.
    cis:
        Optional ``{level: (lo, hi)}`` (or ``{level: {"ci_lo","ci_hi"}}``) headroom CIs.
        Required for a sampled level (L2) to support a verdict.

    Returns
    -------
    ReachabilitySummary
        Levels narrowest-first, the widest level's headroom, any monotonicity violations,
        and whether "routing is hopeless" is supportable.

    Raises
    ------
    ValueError
        If a level name is unknown or a supplied CI is malformed.
    """
    cis = cis or {}
    for level in cis:
        _level_key(level)

    levels: list[LevelOracle] = []
    for level in sorted(matrices_by_level, key=_level_key):
        oracle = oracle_from_matrix(matrices_by_level[level], threshold=threshold)
        levels.append(
            LevelOracle(
                level=level,
                oracle=oracle,
                ci=_coerce_ci(cis.get(level)),
                is_sampled=level in _SAMPLED_LEVELS,
            )
        )

    violations = _monotonicity_violations(levels)

    if not levels:
        return ReachabilitySummary(
            levels=[],
            widest_level=None,
            widest_headroom=None,
            widest_ci=None,
            monotonicity_violations=violations,
            can_rule_out_routing=False,
            verdict="NO_DATA",
            message="No reachability levels supplied; nothing to conclude.",
        )

    widest = levels[-1]
    verdict, can_rule_out, message = _verdict(levels, widest, violations)

    return ReachabilitySummary(
        levels=levels,
        widest_level=widest.level,
        widest_headroom=widest.headroom,
        widest_ci=widest.ci,
        monotonicity_violations=violations,
        can_rule_out_routing=can_rule_out,
        verdict=verdict,
        message=message,
    )


def _monotonicity_violations(levels: Sequence[LevelOracle]) -> list[str]:
    """Flag any drop in ``routing_oracle`` as the level widens.

    Widening reachability strictly adds options, so a max over that set cannot fall. A
    drop is therefore not a finding about routing — it is evidence that one of the
    matrices was mis-collected, and it must not be quietly averaged into a verdict.
    Deliberately NOT applied to the headroom, which is a difference of two monotone
    quantities and so is under no such constraint.
    """
    out: list[str] = []
    for prev, cur in zip(levels, levels[1:]):
        if cur.routing_oracle < prev.routing_oracle - 1e-12:
            out.append(
                f"{cur.level} routing_oracle {cur.routing_oracle:.4f} < "
                f"{prev.level} {prev.routing_oracle:.4f}: impossible, since {cur.level}'s "
                f"option set contains {prev.level}'s — check matrix collection"
            )
    return out


def _verdict(
    levels: Sequence[LevelOracle],
    widest: LevelOracle,
    violations: Sequence[str],
) -> tuple[str, bool, str]:
    """§2.2 fix 2 + §6: only the widest MEASURED level can rule routing out."""
    if violations:
        return (
            "INCONSISTENT",
            False,
            "Reachability levels are not monotone in the oracle, which is impossible if "
            "both were collected correctly. Fix collection before reading any verdict: "
            + "; ".join(violations),
        )

    if not widest.verdict_supported:
        return (
            "NEEDS_CI",
            False,
            f"{widest.level} is a sampled probe and no CI was supplied. ORACLE §2.2 "
            "requires a sampled level be reported with its own CI; a point estimate from "
            "a few sequences cannot establish absence of headroom.",
        )

    if not widest.is_thin:
        return (
            "HEADROOM_REMAINS",
            False,
            f"Headroom at {widest.level} (the widest level measured) is "
            f"{widest.headroom:.4f}, above the {THIN_HEADROOM} 'thin' bar. Routing is not "
            "ruled out.",
        )

    # Widest measured level is thin. Whether that settles it depends on how wide we got.
    if widest.level == LEVEL_ORDER[-1]:
        return (
            "POOL_BOUND",
            True,
            f"Headroom is thin at {widest.level}, the widest reachability level defined. "
            "Routing is ruled out on this pool: the lever is the pool, not the router.",
        )

    missing = [lv for lv in LEVEL_ORDER[_level_key(widest.level) + 1 :]]
    return (
        "LOWER_BOUND_ONLY",
        False,
        f"Headroom is thin at {widest.level}, but {widest.level} is a LOWER BOUND on the "
        f"reachable ceiling — {', '.join(missing)} were not measured. ORACLE §2.2: a thin "
        "gap only rules routing out if the wider levels are also thin. Measure "
        f"{missing[0]} before concluding routing cannot help.",
    )


def render(summary: ReachabilitySummary) -> str:
    """Markdown: one row per level, then the widest-level verdict."""
    out = ["# Oracle reachability levels (ORACLE §2.2)\n"]
    if not summary.levels:
        out.append("_No reachability levels supplied._\n")
        return "\n".join(out)

    out.append("| Level | What it measures | Oracle | Best single | Headroom | 95% CI |")
    out.append("| --- | --- | ---: | ---: | ---: | :---: |")
    for lv in summary.levels:
        ci = f"[{lv.ci[0]:.4f}, {lv.ci[1]:.4f}]" if lv.ci is not None else "—"
        out.append(
            f"| `{lv.level}` | {LEVEL_DESCRIPTIONS.get(lv.level, '')} | "
            f"{lv.routing_oracle:.4f} | {lv.oracle.best_single:.4f} | "
            f"{lv.headroom:.4f} | {ci} |"
        )

    out.append("")
    out.append(f"**Widest level measured:** `{summary.widest_level}`")
    out.append(f"**Verdict:** `{summary.verdict}` — {summary.message}")
    if summary.monotonicity_violations:
        out.append("")
        out.append("**Integrity violations:**")
        for v in summary.monotonicity_violations:
            out.append(f"- {v}")
    out.append("")
    return "\n".join(out)
