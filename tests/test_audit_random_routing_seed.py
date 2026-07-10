"""The audit's random-routing baseline must depend on (seed, task_id), not on
asyncio scheduling.

`run_audit` fans every task out through `asyncio.gather`. If all trajectories share
one policy rng, the turn-2+ draws are consumed in network-completion order, so the
"sealed", supposedly-reproducible audit number for `random_routing` varies run to
run. `_RandomAuditPolicy.decide` now draws from the per-trajectory rng that
`run_trajectory` passes through (seeded by `task_rng(seed, task_id)`); these tests
pin that contract. No torch/network needed — the policy is pure.
"""
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

_SCRIPT = _REPO / "scripts" / "audit_eval.py"
_spec = importlib.util.spec_from_file_location("audit_eval", _SCRIPT)
audit_eval = importlib.util.module_from_spec(_spec)
sys.modules["audit_eval"] = audit_eval
_spec.loader.exec_module(audit_eval)

from trinity.eval import task_rng  # noqa: E402

_RandomAuditPolicy = audit_eval._RandomAuditPolicy


def _sequence(policy, task_id, seed, n=5):
    """Simulate one trajectory's per-turn decisions from its own seeded rng."""
    rng = task_rng(seed, task_id)
    return [policy.decide("", rng=rng) for _ in range(n)]


def test_decide_uses_the_passed_rng_and_is_deterministic():
    policy = _RandomAuditPolicy(3)
    # Same (seed, task_id) -> identical decision sequence, regardless of the
    # instance's own rng state.
    assert _sequence(policy, "taskA", seed=42) == _sequence(policy, "taskA", seed=42)


def test_baseline_is_invariant_to_task_ordering():
    policy = _RandomAuditPolicy(3)
    task_ids = ["q0", "q1", "q2", "q3"]
    seed = 314159265 * 10000

    forward = {tid: _sequence(policy, tid, seed) for tid in task_ids}
    reverse = {tid: _sequence(policy, tid, seed) for tid in reversed(task_ids)}

    # Each task owns its rng, so a task's decisions never depend on when other
    # concurrently-gathered tasks ran.
    assert forward == reverse


def test_passed_rng_overrides_the_instance_rng():
    import random

    a = _RandomAuditPolicy(3, rng=random.Random(1))
    b = _RandomAuditPolicy(3, rng=random.Random(999))
    # Different instance rngs, but the per-call rng decides -> identical result.
    assert a.decide("", rng=random.Random(7)) == b.decide("", rng=random.Random(7))


def test_agent_and_role_are_in_range():
    from trinity.types import ROLE_ORDER

    policy = _RandomAuditPolicy(3)
    for _ in range(50):
        agent, role = policy.decide("", rng=task_rng(1, "q"))
        assert 0 <= agent < 3
        assert role in ROLE_ORDER
