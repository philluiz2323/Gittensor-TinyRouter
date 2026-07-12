"""Deterministic policy-sampling RNG for training trajectories."""
from __future__ import annotations

import hashlib


def trajectory_sampling_manual_seed(
    *,
    run_seed: int,
    generation: int,
    candidate_idx: int,
    task_id: str,
) -> int:
    """Derive the manual seed for one trajectory's policy sampling."""
    payload = f"{int(run_seed)}:{int(generation)}:{int(candidate_idx)}:{task_id}".encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "little",
    ) & 0xFFFFFFFF


def trajectory_sampling_rng(
    *,
    run_seed: int,
    generation: int,
    candidate_idx: int,
    task_id: str,
):
    """Return a ``torch.Generator`` for one trajectory's categorical sampling.

    Derived independently per task so ``asyncio.gather`` completion order does
    not affect reproducibility. The ``run_seed`` is the training ``--seed``.
    """
    import torch

    gen = torch.Generator()
    gen.manual_seed(
        trajectory_sampling_manual_seed(
            run_seed=run_seed,
            generation=generation,
            candidate_idx=candidate_idx,
            task_id=task_id,
        )
    )
    return gen
