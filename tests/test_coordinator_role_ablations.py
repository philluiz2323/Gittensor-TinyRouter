"""Role-ablation logic for SPEC R9 (``no_thinker`` / ``no_trirole``).

Pure-numpy tests: this file imports **no torch**, matching the module under
test. The parity check against the real ``LinearHead.select`` lives in
``test_torch_role_ablation_parity.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity.coordinator.ablations import (
    ROLE_ABLATIONS,
    AblatedPolicy,
    categorical_pick,
    make_ablated_policy,
    masked_role_probs,
    permitted_roles,
    role_mask,
)
from trinity.types import ROLE_ORDER, Role

#: Point child interpreters at *this* checkout's ``src``, not whatever copy of
#: trinity happens to be installed, so the subprocess probes below test the tree
#: they are running from.
_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _probe(code: str) -> str:
    """Run ``code`` in a fresh interpreter against this checkout; return stdout."""
    env = {
        **os.environ,
        "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    return out.stdout.strip()


N_MODELS = 3


def _stub_logits(agent, role):
    """A ``logits_fn`` seam returning fixed logits for any transcript."""

    def fn(_text: str):
        return np.asarray(agent, dtype=np.float64), np.asarray(role, dtype=np.float64)

    return fn


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_names_are_the_r9_variant_names():
    """The keys must be what the merged R9 verifier expects to be handed."""
    assert set(ROLE_ABLATIONS) == {"no_thinker", "no_trirole"}


def test_no_thinker_drops_only_the_thinker():
    assert ROLE_ABLATIONS["no_thinker"] == (Role.WORKER, Role.VERIFIER)
    assert Role.THINKER not in ROLE_ABLATIONS["no_thinker"]


def test_no_trirole_collapses_to_a_single_role():
    assert len(ROLE_ABLATIONS["no_trirole"]) == 1


def test_every_variant_keeps_at_least_one_role():
    for name, roles in ROLE_ABLATIONS.items():
        assert len(roles) >= 1, name


def test_permitted_roles_rejects_unknown_variant():
    with pytest.raises(KeyError, match="unknown role ablation"):
        permitted_roles("no_such_ablation")


# --------------------------------------------------------------------------
# role_mask
# --------------------------------------------------------------------------


def test_role_mask_marks_exactly_the_permitted_positions():
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    assert mask.tolist() == [r in (Role.WORKER, Role.VERIFIER) for r in ROLE_ORDER]
    assert mask.dtype == bool


def test_role_mask_rejects_empty_permitted_set():
    with pytest.raises(ValueError, match="empty"):
        role_mask(())


def test_role_mask_rejects_roles_outside_role_order():
    with pytest.raises(ValueError, match="not in ROLE_ORDER"):
        role_mask(("not_a_role",))


# --------------------------------------------------------------------------
# masked_role_probs
# --------------------------------------------------------------------------


def test_masked_probs_are_zero_on_masked_and_sum_to_one():
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    probs = masked_role_probs(np.array([5.0, 1.0, 0.5]), mask)
    assert probs[ROLE_ORDER.index(Role.THINKER)] == 0.0
    assert probs.sum() == pytest.approx(1.0)


def test_masked_probs_renormalize_rather_than_just_zeroing():
    """The survivors must be a proper categorical, not the full softmax clipped."""
    logits = np.array([5.0, 1.0, 0.5])
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    probs = masked_role_probs(logits, mask)

    full = np.exp(logits - logits.max())
    full /= full.sum()
    kept = full[1:] / full[1:].sum()
    assert probs[1:] == pytest.approx(kept)
    # ...and that is strictly larger than the clipped-but-unnormalized version.
    assert probs[1] > full[1]


def test_masked_probs_ignore_a_huge_masked_logit():
    """A masked entry must not shift the survivors' scale at all."""
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    base = masked_role_probs(np.array([0.0, 1.0, 0.5]), mask)
    huge = masked_role_probs(np.array([1e9, 1.0, 0.5]), mask)
    assert huge == pytest.approx(base)


def test_masked_probs_single_survivor_is_deterministic():
    mask = role_mask((Role.WORKER,))
    probs = masked_role_probs(np.array([9.0, -3.0, 4.0]), mask)
    expected = [0.0, 0.0, 0.0]
    expected[ROLE_ORDER.index(Role.WORKER)] = 1.0
    assert probs.tolist() == expected


def test_masked_probs_degenerate_all_neg_inf_falls_back_to_uniform():
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    probs = masked_role_probs(np.array([0.0, -np.inf, -np.inf]), mask)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert probs[1] == pytest.approx(0.5)


def test_masked_probs_rejects_wrong_shape():
    mask = role_mask((Role.WORKER,))
    with pytest.raises(ValueError, match="must have shape"):
        masked_role_probs(np.array([1.0, 2.0]), mask)


def test_masked_probs_rejects_all_false_mask():
    with pytest.raises(ValueError, match="every role"):
        masked_role_probs(np.zeros(3), np.zeros(3, dtype=bool))


# --------------------------------------------------------------------------
# categorical_pick
# --------------------------------------------------------------------------


def test_argmax_unmasked_matches_numpy_argmax():
    z = np.array([0.1, 2.5, -1.0, 0.7])
    assert categorical_pick(z, sample=False) == int(np.argmax(z))


def test_masked_argmax_picks_the_best_survivor():
    z = np.array([9.0, 1.0, 5.0])  # THINKER wins outright
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    assert categorical_pick(z, sample=False, mask=mask) == 2  # VERIFIER


def test_masked_argmax_is_not_a_post_hoc_remap_of_the_full_argmax():
    """The point of masking pre-softmax, stated as a test.

    Full-model argmax is THINKER. A post-hoc rule ("if Thinker, use the next
    role in ROLE_ORDER") would answer WORKER; the restricted argmax is VERIFIER
    because Verifier outscores Worker among the survivors.
    """
    z = np.array([9.0, 1.0, 5.0])
    assert int(np.argmax(z)) == ROLE_ORDER.index(Role.THINKER)
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    picked = ROLE_ORDER[categorical_pick(z, sample=False, mask=mask)]
    assert picked is Role.VERIFIER
    assert picked is not Role.WORKER


def test_sampling_never_draws_a_masked_option():
    rng = np.random.default_rng(0)
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    z = np.array([10.0, 0.0, 0.0])  # masked option has by far the largest logit
    draws = [categorical_pick(z, sample=True, rng=rng, mask=mask) for _ in range(400)]
    assert ROLE_ORDER.index(Role.THINKER) not in set(draws)


def test_sampling_follows_the_restricted_distribution():
    rng = np.random.default_rng(20260722)
    mask = role_mask((Role.WORKER, Role.VERIFIER))
    z = np.array([0.0, 1.0, 0.0])
    expected = masked_role_probs(z, mask)
    n = 20000
    draws = np.array([categorical_pick(z, sample=True, rng=rng, mask=mask) for _ in range(n)])
    freq = np.array([(draws == i).mean() for i in range(3)])
    assert freq == pytest.approx(expected, abs=0.02)


def test_sampling_is_reproducible_for_a_seeded_generator():
    a = [categorical_pick(np.array([0.3, 0.4, 0.3]), sample=True, rng=np.random.default_rng(7))
         for _ in range(5)]
    b = [categorical_pick(np.array([0.3, 0.4, 0.3]), sample=True, rng=np.random.default_rng(7))
         for _ in range(5)]
    assert a == b


def test_sampling_rejects_a_non_numpy_generator():
    """A torch.Generator would be silently ignored -- reject it loudly instead."""

    class _FakeTorchGenerator:
        pass

    _FakeTorchGenerator.__module__ = "torch._C"
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        categorical_pick(np.zeros(3), sample=True, rng=_FakeTorchGenerator())


def test_pick_rejects_empty_logits():
    with pytest.raises(ValueError, match="empty"):
        categorical_pick(np.array([]), sample=False)


def test_pick_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError, match="mask shape"):
        categorical_pick(np.zeros(3), sample=False, mask=np.ones(2, dtype=bool))


def test_pick_rejects_all_false_mask():
    with pytest.raises(ValueError, match="every option"):
        categorical_pick(np.zeros(3), sample=False, mask=np.zeros(3, dtype=bool))


# --------------------------------------------------------------------------
# AblatedPolicy
# --------------------------------------------------------------------------


def test_no_thinker_never_returns_thinker_over_many_transcripts():
    rng = np.random.default_rng(1)
    for _ in range(200):
        agent = rng.normal(size=N_MODELS)
        role = rng.normal(size=len(ROLE_ORDER)) * 5.0
        pol = AblatedPolicy(None, "no_thinker", logits_fn=_stub_logits(agent, role))
        _, chosen = pol.decide("t")
        assert chosen is not Role.THINKER


def test_no_trirole_always_returns_the_same_role():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(200):
        agent = rng.normal(size=N_MODELS)
        role = rng.normal(size=len(ROLE_ORDER)) * 5.0
        pol = AblatedPolicy(None, "no_trirole", logits_fn=_stub_logits(agent, role))
        seen.add(pol.decide("t")[1])
    assert seen == {Role.WORKER}


def test_ablation_leaves_agent_selection_untouched():
    """R9's role ablations remove role structure, not model routing."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        agent = rng.normal(size=N_MODELS)
        role = rng.normal(size=len(ROLE_ORDER))
        fn = _stub_logits(agent, role)
        full_choice = categorical_pick(agent, sample=False)
        for variant in ROLE_ABLATIONS:
            pol = AblatedPolicy(None, variant, logits_fn=fn)
            assert pol.decide("t")[0] == full_choice


def test_decide_returns_the_protocol_shape():
    pol = AblatedPolicy(None, "no_thinker", logits_fn=_stub_logits([0.0, 1.0, 0.0], [0, 1, 2]))
    out = pol.decide("t", sample=False, rng=None)
    assert isinstance(out, tuple) and len(out) == 2
    idx, role = out
    assert isinstance(idx, int) and isinstance(role, Role)


def test_decide_sampling_path_stays_inside_the_permitted_set():
    rng = np.random.default_rng(11)
    pol = AblatedPolicy(None, "no_thinker", logits_fn=_stub_logits([1.0, 0, 0], [9.0, 0, 0]))
    roles = {pol.decide("t", sample=True, rng=rng)[1] for _ in range(300)}
    assert Role.THINKER not in roles


def test_collapse_role_override_changes_the_surviving_role():
    pol = AblatedPolicy(
        None, "no_trirole", collapse_role=Role.VERIFIER,
        logits_fn=_stub_logits([0.0, 1.0, 0.0], [5.0, 5.0, 0.0]),
    )
    assert pol.decide("t")[1] is Role.VERIFIER


def test_collapse_role_rejected_on_a_multi_role_variant():
    with pytest.raises(ValueError, match="single-role"):
        AblatedPolicy(None, "no_thinker", collapse_role=Role.WORKER)


def test_unknown_variant_rejected_at_construction():
    with pytest.raises(KeyError, match="unknown role ablation"):
        AblatedPolicy(None, "no_svf")


def test_role_probs_exposes_the_restricted_distribution():
    pol = AblatedPolicy(None, "no_thinker", logits_fn=_stub_logits([0.0], [5.0, 1.0, 0.5]))
    probs = pol.role_probs("t")
    assert probs[ROLE_ORDER.index(Role.THINKER)] == 0.0
    assert probs.sum() == pytest.approx(1.0)


def test_configure_forwards_theta_to_the_wrapped_policy():
    class _Inner:
        def __init__(self):
            self.seen = None
            self.spec = "SPEC-OBJ"

        def configure(self, theta, spec=None):
            self.seen = (np.asarray(theta), spec)

    inner = _Inner()
    pol = AblatedPolicy(inner, "no_thinker", logits_fn=_stub_logits([0.0], [0, 1, 2]))
    theta = np.arange(4, dtype=np.float64)
    pol.configure(theta)
    assert inner.seen is not None
    assert inner.seen[0].tolist() == theta.tolist()
    assert pol.spec == "SPEC-OBJ"


def test_wrapper_does_not_mutate_the_inner_policy():
    class _Inner:
        pass

    inner = _Inner()
    before = dict(vars(inner))
    AblatedPolicy(inner, "no_thinker", logits_fn=_stub_logits([0.0], [0, 1, 2])).decide("t")
    assert vars(inner) == before


def test_missing_encoder_head_raises_a_clear_error():
    pol = AblatedPolicy(object(), "no_thinker")
    with pytest.raises(TypeError, match="logits_fn"):
        pol.decide("t")


def test_rejection_path_does_not_import_torch():
    """Rejecting a badly-shaped policy must not drag torch into ``sys.modules``.

    Several modules ship a ``test_no_torch_imported`` guard, and pytest runs
    files in alphabetical order -- so importing torch on this path would poison
    every guard that sorts after this file.
    """
    got = _probe(
        "import sys\n"
        "from trinity.coordinator.ablations import AblatedPolicy\n"
        "pol = AblatedPolicy(object(), 'no_thinker')\n"
        "try:\n"
        "    pol.decide('t')\n"
        "except TypeError:\n"
        "    pass\n"
        "print('torch' in sys.modules)\n"
    )
    assert got == "False", got


def test_make_ablated_policy_matches_direct_construction():
    fn = _stub_logits([0.0, 1.0], [1.0, 2.0, 0.0])
    a = make_ablated_policy(None, "no_thinker", logits_fn=fn)
    b = AblatedPolicy(None, "no_thinker", logits_fn=fn)
    assert a.permitted == b.permitted
    assert a.mask.tolist() == b.mask.tolist()
    assert a.decide("t") == b.decide("t")


# --------------------------------------------------------------------------
# import cost
# --------------------------------------------------------------------------


def test_importing_the_module_does_not_import_torch():
    """Run in a subprocess so the result is independent of test ordering."""
    got = _probe(
        "import sys; import trinity.coordinator.ablations as m; "
        "print('torch' in sys.modules)"
    )
    assert got == "False", got
