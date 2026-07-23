"""Role-ablation policies for SPEC R9 (``no_thinker`` / ``no_trirole``).

``docs/SPEC.md`` §1.3 invariant **R9** claims that removing each design component
hurts accuracy. The *verifier* for R9 is already merged --
``trinity.analysis.ablations.analyze`` consumes ``{full, no_svf, no_thinker,
no_trirole, last_token}`` -- but nothing in ``src/`` could ever *produce* those
numbers: the four variant names appear only inside docstrings and the example
JSON in ``scripts/ablations_report.py``. This module is the producer for the two
**role** ablations; the SVF / penultimate-token pair are a separate concern.

What an ablation means here
---------------------------
Both variants restrict the *role* group of the head's output. The agent (pool
model) group is left completely untouched -- R9's role ablations remove role
structure, not model routing.

``no_thinker``
    Drop the Thinker from the role set. The coordinator may still choose Worker
    or Verifier.
``no_trirole``
    Collapse to a *single* role, so there is no role differentiation at all --
    every turn is the same kind of turn.

Masking happens on the **logits, before the softmax**, which makes the result a
true restricted categorical. That is not the same thing as taking the full
model's decision and remapping it afterwards: if the un-ablated argmax picked
Thinker, a post-hoc remap has no way to know whether Worker or Verifier was
runner-up, so it cannot reconstruct the restricted argmax. Sampling has the same
problem -- the restricted distribution is renormalized over the survivors, not
the full distribution with one outcome deleted. Doing it at the logit level is
what makes these numbers comparable with the full model's.

[OUR CHOICE] ``no_trirole`` collapses to :attr:`~trinity.types.Role.WORKER`. The
SPEC names the ablation but does not say which role survives; Worker is the role
whose prompt asks for a direct solution, so "no role differentiation" reads most
naturally as "every turn is a plain solver turn". It is a constructor argument,
so the choice is overridable rather than baked in.

Import cost
-----------
This module imports **no torch** at module scope -- only numpy and
``trinity.types``. torch is imported lazily inside :meth:`AblatedPolicy.decide`,
exactly as ``policy.CoordinatorPolicy.decide`` does, so importing
``trinity.coordinator.ablations`` stays free.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from trinity.types import ROLE_ORDER, Role

__all__ = [
    "ROLE_ABLATIONS",
    "AblatedPolicy",
    "categorical_pick",
    "make_ablated_policy",
    "masked_role_probs",
    "permitted_roles",
    "role_mask",
]


#: Variant name -> the roles that survive the ablation. Keys are exactly the
#: names ``trinity.analysis.ablations`` expects to find in its input mapping.
ROLE_ABLATIONS: Mapping[str, tuple[Role, ...]] = {
    "no_thinker": (Role.WORKER, Role.VERIFIER),
    "no_trirole": (Role.WORKER,),
}


class _LogitSource(Protocol):
    """What :class:`AblatedPolicy` needs from the policy it wraps."""

    def __call__(self, transcript_text: str) -> tuple[np.ndarray, np.ndarray]: ...


def permitted_roles(variant: str) -> tuple[Role, ...]:
    """Return the roles that survive ``variant``.

    Parameters
    ----------
    variant:
        One of the keys of :data:`ROLE_ABLATIONS`.

    Raises
    ------
    KeyError
        If ``variant`` is not a known role ablation.
    """
    try:
        return ROLE_ABLATIONS[variant]
    except KeyError:
        raise KeyError(
            f"unknown role ablation {variant!r}; known: {sorted(ROLE_ABLATIONS)}"
        ) from None


def role_mask(permitted: Sequence[Role]) -> np.ndarray:
    """Boolean mask over :data:`~trinity.types.ROLE_ORDER`; ``True`` = keep.

    Raises
    ------
    ValueError
        If ``permitted`` is empty (an all-masked role group has no decision to
        make) or names a role outside ``ROLE_ORDER``.
    """
    if len(permitted) == 0:
        raise ValueError("permitted role set is empty; at least one role must survive")
    unknown = [r for r in permitted if r not in ROLE_ORDER]
    if unknown:
        raise ValueError(
            f"roles {unknown} are not in ROLE_ORDER ({[r.value for r in ROLE_ORDER]})"
        )
    keep = set(permitted)
    return np.array([r in keep for r in ROLE_ORDER], dtype=bool)


def masked_role_probs(role_logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over the permitted roles only; masked entries get probability 0.

    The surviving probabilities are renormalized to sum to 1, so this is the
    categorical the ablated coordinator actually draws from -- not the full
    distribution with entries zeroed.

    Parameters
    ----------
    role_logits:
        Raw role logits, shape ``(len(ROLE_ORDER),)``.
    mask:
        Boolean mask from :func:`role_mask`, same shape.

    Returns
    -------
    np.ndarray
        ``float64`` probabilities summing to 1, zero at every masked position.

    Raises
    ------
    ValueError
        On a shape mismatch or an all-``False`` mask.
    """
    z = np.asarray(role_logits, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    if z.shape != (len(ROLE_ORDER),):
        raise ValueError(
            f"role_logits must have shape ({len(ROLE_ORDER)},), got {z.shape}"
        )
    if m.shape != z.shape:
        raise ValueError(f"mask shape {m.shape} != role_logits shape {z.shape}")
    if not m.any():
        raise ValueError("mask removes every role; at least one must survive")

    probs = np.zeros_like(z)
    probs[m] = _softmax_over(z[m])
    return probs


def _softmax_over(kept: np.ndarray) -> np.ndarray:
    """Stable softmax over the surviving logits, degenerate cases included.

    Shifting by the max is taken over the KEPT entries only, so a masked
    ``+inf``-ish logit cannot move the survivors' scale. When every survivor is
    ``-inf`` (or the sum otherwise degenerates) the choice carries no
    information, so this falls back to uniform instead of emitting NaNs.
    """
    mx = kept.max()
    if not np.isfinite(mx):
        return np.full(kept.shape, 1.0 / kept.size)
    exp = np.exp(kept - mx)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(kept.shape, 1.0 / kept.size)
    return exp / total


def categorical_pick(
    logits: np.ndarray,
    *,
    sample: bool,
    rng: Any = None,
    mask: np.ndarray | None = None,
) -> int:
    """Pick an index from ``logits``, matching ``head.LinearHead.select``'s rule.

    ``sample=False`` takes the argmax (eval); ``sample=True`` draws from the
    softmax (training fitness) -- SPEC §4.3. When ``mask`` is given, both paths
    are restricted to the permitted positions.

    Parameters
    ----------
    logits:
        1-D logit vector.
    sample:
        Draw from the categorical instead of taking the argmax.
    rng:
        Optional :class:`numpy.random.Generator` for reproducible sampling.
        Ignored when ``sample=False``.
    mask:
        Optional boolean keep-mask, same shape as ``logits``.

    Raises
    ------
    TypeError
        If ``rng`` is not a numpy ``Generator``. A ``torch.Generator`` (which
        ``head.select`` accepts) is rejected loudly rather than silently ignored
        -- passing one and getting unseeded draws would look reproducible
        without being so.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    if z.size == 0:
        raise ValueError("logits is empty")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.shape != z.shape:
            raise ValueError(f"mask shape {m.shape} != logits shape {z.shape}")
        if not m.any():
            raise ValueError("mask removes every option")
    else:
        m = np.ones_like(z, dtype=bool)

    if not sample:
        # argmax restricted to the permitted positions.
        masked = np.where(m, z, -np.inf)
        return int(np.argmax(masked))

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a numpy.random.Generator (got "
            f"{type(rng).__module__}.{type(rng).__name__}); the ablation wrapper "
            "samples with numpy, not torch"
        )

    # Reuse the masked-softmax path so sampling and argmax see the same support.
    probs = np.zeros_like(z)
    probs[m] = _softmax_over(z[m])
    return int(rng.choice(z.size, p=probs))


class AblatedPolicy:
    """A :class:`~trinity.orchestration.session.Policy` with roles ablated.

    Satisfies the session's ``Policy`` protocol (``decide(transcript_text, *,
    sample, rng) -> (agent_idx, Role)``), so it drops straight into
    ``run_trajectory`` with no change to the session or submission path.

    Parameters
    ----------
    inner:
        The policy being ablated. Must expose ``.encoder`` and ``.head`` (as
        ``CoordinatorPolicy`` does) unless ``logits_fn`` is supplied.
    variant:
        A key of :data:`ROLE_ABLATIONS`.
    collapse_role:
        Overrides which single role ``no_trirole`` collapses to. Ignored by
        variants that keep more than one role.
    logits_fn:
        Optional seam returning ``(agent_logits, role_logits)`` as numpy arrays
        for a transcript. Supplying it avoids touching ``inner`` at all, which
        is how the ablation logic is exercised on CPU without loading the SLM.

    Notes
    -----
    Nothing here mutates ``inner``; the wrapper is additive and the un-ablated
    policy remains usable alongside it.
    """

    def __init__(
        self,
        inner: Any,
        variant: str,
        *,
        collapse_role: Role | None = None,
        logits_fn: _LogitSource | None = None,
    ) -> None:
        roles = permitted_roles(variant)
        if collapse_role is not None:
            if len(roles) != 1:
                raise ValueError(
                    f"collapse_role only applies to single-role ablations; "
                    f"{variant!r} keeps {[r.value for r in roles]}"
                )
            roles = (collapse_role,)
        self.inner = inner
        self.variant = variant
        self.permitted = roles
        self.mask = role_mask(roles)
        self._logits_fn = logits_fn

    @property
    def spec(self) -> Any:
        """Passthrough to the wrapped policy's ``ParamSpec``."""
        return getattr(self.inner, "spec", None)

    def configure(self, theta: np.ndarray, spec: Any = None) -> None:
        """Forward θ to the wrapped policy unchanged.

        The ablation restricts *decisions*, not parameters: θ keeps its full
        width so an ablated run is directly comparable with the full model's.
        """
        self.inner.configure(theta, spec)

    def _logits(self, transcript_text: str) -> tuple[np.ndarray, np.ndarray]:
        if self._logits_fn is not None:
            agent_logits, role_logits = self._logits_fn(transcript_text)
            return np.asarray(agent_logits), np.asarray(role_logits)

        encoder = getattr(self.inner, "encoder", None)
        head = getattr(self.inner, "head", None)
        if encoder is None or head is None:
            raise TypeError(
                f"{type(self.inner).__name__} exposes no .encoder/.head; pass "
                "logits_fn= to wrap a policy with a different shape"
            )

        # Imported only once the torch path is certain to be taken. Importing it
        # before this check would pull torch into ``sys.modules`` even on the
        # rejection path, which the repo's ``test_no_torch_imported`` guards
        # (in several test modules) would then see and fail on.
        import torch  # lazy: keeps module import torch-free (see module docstring)

        h = encoder.encode(transcript_text)
        h_t = torch.as_tensor(np.asarray(h, dtype=np.float32), device=head.weight.device)
        agent_logits, role_logits = head.forward(h_t)
        return (
            agent_logits.detach().to("cpu").float().numpy(),
            role_logits.detach().to("cpu").float().numpy(),
        )

    def decide(
        self, transcript_text: str, *, sample: bool = False, rng: Any = None
    ) -> tuple[int, Role]:
        """Pick ``(agent_idx, role)`` with the ablated role set."""
        agent_logits, role_logits = self._logits(transcript_text)
        agent_idx = categorical_pick(agent_logits, sample=sample, rng=rng)
        role_pos = categorical_pick(role_logits, sample=sample, rng=rng, mask=self.mask)
        return int(agent_idx), ROLE_ORDER[role_pos]

    def role_probs(self, transcript_text: str) -> np.ndarray:
        """The restricted role distribution for a transcript (diagnostics)."""
        _, role_logits = self._logits(transcript_text)
        return masked_role_probs(role_logits, self.mask)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        kept = ",".join(r.value for r in self.permitted)
        return f"AblatedPolicy(variant={self.variant!r}, roles=[{kept}])"


def make_ablated_policy(
    policy: Any,
    variant: str,
    *,
    collapse_role: Role | None = None,
    logits_fn: _LogitSource | None = None,
) -> AblatedPolicy:
    """Wrap ``policy`` in the named role ablation.

    Thin factory kept so callers name the variant rather than assembling masks
    by hand; see :data:`ROLE_ABLATIONS` for the known names.
    """
    return AblatedPolicy(
        policy, variant, collapse_role=collapse_role, logits_fn=logits_fn
    )
