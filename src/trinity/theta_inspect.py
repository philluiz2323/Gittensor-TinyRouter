"""Offline diagnostics for a trained parameter vector (theta).

Why this exists
---------------
A submission's theta is the 13,312-dim CMA-ES vector that packs the linear head
(``n_a x d_h`` = 6,144) and the SVF scales (7,168), split by
:func:`trinity.coordinator.params.unpack`. ``submission.schema`` validates a theta
for *shape and finiteness* (a pass/fail gate), but tells a miner nothing about
whether training actually did anything.

Two silent failure modes are invisible until a paid eval comes back flat:

* the **head never moved** off its ``W = 0`` init, so routing is still the uniform
  policy — every logit is 0 and the argmax is arbitrary;
* the **SVF scales never moved** off their ``1.0`` init, so the SLM is unmodified —
  Transformer^2 adaptation contributed nothing.

Both look like a perfectly valid submission to a shape check. This module reports
the descriptive statistics that surface them — per-block norms, how far each block
moved from its init, how many SVF scales actually changed, and any non-finite
entry — so a contributor can see at a glance whether a run trained.

Pure numpy. No network, no GPU, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BlockStats", "ThetaReport", "inspect_theta"]

# A head weight this close to 0, or an SVF scale this close to 1, counts as "at init".
_ATOL = 1e-6


@dataclass(frozen=True)
class BlockStats:
    """Descriptive stats for one block of a theta (the head, or the SVF scales)."""

    name: str
    size: int
    l2_norm: float
    max_abs: float
    mean: float
    n_nonfinite: int
    n_at_init: int          # entries within _ATOL of this block's init value
    dist_from_init: float   # L2 distance from the block's init vector

    @property
    def n_moved(self) -> int:
        """Entries that moved off their init value (finite and off-init).

        A non-finite entry is neither at-init nor moved — it is a separate
        failure mode counted by :attr:`n_nonfinite`, so it must be excluded here.
        Deriving this as ``size - n_at_init`` alone would count every NaN/Inf as
        "moved" and falsely report an otherwise-at-init block as trained.
        """
        return self.size - self.n_at_init - self.n_nonfinite

    @property
    def frac_moved(self) -> float:
        """Fraction of the block that moved off init (0 for an untouched block)."""
        return self.n_moved / self.size if self.size else 0.0

    @property
    def at_init(self) -> bool:
        """True iff the whole block is still at its init (training did nothing)."""
        return self.n_moved == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "name": self.name, "size": self.size, "l2_norm": self.l2_norm,
            "max_abs": self.max_abs, "mean": self.mean, "n_nonfinite": self.n_nonfinite,
            "n_at_init": self.n_at_init, "n_moved": self.n_moved,
            "frac_moved": self.frac_moved, "dist_from_init": self.dist_from_init,
            "at_init": self.at_init,
        }


@dataclass(frozen=True)
class ThetaReport:
    """Whole-theta diagnostics with the head / SVF block breakdown."""

    n_total: int
    head: BlockStats
    svf: BlockStats
    n_nonfinite: int

    @property
    def finite(self) -> bool:
        """True iff every entry is finite (no NaN/Inf)."""
        return self.n_nonfinite == 0

    @property
    def head_trained(self) -> bool:
        """True iff the head moved off its ``W = 0`` init."""
        return not self.head.at_init

    @property
    def svf_trained(self) -> bool:
        """True iff any SVF scale moved off its ``1.0`` init."""
        return not self.svf.at_init

    @property
    def trained(self) -> bool:
        """True iff *either* block moved — a fully-at-init theta never trained."""
        return self.head_trained or self.svf_trained

    @property
    def warnings(self) -> list[str]:
        """Human-readable flags for the silent failure modes."""
        out: list[str] = []
        if not self.finite:
            out.append(f"{self.n_nonfinite} non-finite entr(y/ies) (NaN/Inf) in theta")
        if not self.head_trained:
            out.append("head is still at its W=0 init: routing is the uniform policy")
        if not self.svf_trained:
            out.append("SVF scales are still at their 1.0 init: the SLM was not adapted")
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_total": self.n_total,
            "finite": self.finite,
            "trained": self.trained,
            "head_trained": self.head_trained,
            "svf_trained": self.svf_trained,
            "warnings": self.warnings,
            "head": self.head.to_dict(),
            "svf": self.svf.to_dict(),
        }


def _block_stats(name: str, vec: np.ndarray, init_value: float) -> BlockStats:
    finite = np.isfinite(vec)
    n_nonfinite = int(vec.size - int(finite.sum()))
    safe = np.where(finite, vec, 0.0)
    at_init = np.abs(safe - init_value) <= _ATOL
    # Only finite entries can be "at init"; a NaN never counts as unchanged.
    at_init = at_init & finite
    init_vec = np.full_like(safe, init_value)
    return BlockStats(
        name=name,
        size=int(vec.size),
        l2_norm=float(np.linalg.norm(safe)),
        max_abs=float(np.abs(safe).max()) if vec.size else 0.0,
        mean=float(safe.mean()) if vec.size else 0.0,
        n_nonfinite=n_nonfinite,
        n_at_init=int(at_init.sum()),
        dist_from_init=float(np.linalg.norm(safe - init_vec)),
    )


def inspect_theta(theta: Any, spec: Any = None) -> ThetaReport:
    """Report descriptive diagnostics for a packed parameter vector.

    Args:
        theta: The flat parameter vector (array-like of length ``spec.n_total``).
        spec: A :class:`~trinity.coordinator.params.ParamSpec`. Defaults to the
            canonical spec from ``params.make_spec()``.

    Returns:
        A :class:`ThetaReport`. The head init is ``0`` and the SVF init is ``1.0``
        (``params.initial_theta``), so "at init" means that block never trained.

    Raises:
        ValueError: If ``theta`` is not 1-D or its length does not match
            ``spec.n_total`` — a shape mismatch is a hard error, not something to
            paper over with a partial report.
    """
    from trinity.coordinator import params as P

    spec = spec if spec is not None else P.make_spec()
    arr = np.asarray(theta, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"theta must be 1-D; got shape {arr.shape}")
    if arr.size != spec.n_total:
        raise ValueError(
            f"theta has {arr.size} entries but spec.n_total is {spec.n_total}"
        )

    head_W, svf_scales = P.unpack(arr, spec)
    head = _block_stats("head", np.asarray(head_W, dtype=float).ravel(), init_value=0.0)
    svf = _block_stats("svf", np.asarray(svf_scales, dtype=float).ravel(), init_value=1.0)
    return ThetaReport(
        n_total=int(arr.size), head=head, svf=svf,
        n_nonfinite=head.n_nonfinite + svf.n_nonfinite,
    )
