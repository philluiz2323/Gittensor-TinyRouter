"""Separable CMA-ES trainer for the head parameter vector.

Wraps the `cma` library with the diagonal/separable option. The objective is
the mean task score of the coordinator over a minibatch of train tasks (each
candidate parameter vector = one head; fitness = run the coordination loop and
score). See docs/SPEC.md §5.

TODO(SPEC §5): search-space dim, lambda/mu, sigma0, generations, fitness
aggregation, and the RL/IL/random baselines for the comparison.
"""
from __future__ import annotations
# Placeholder — implemented once docs/SPEC.md pins the optimizer config.
