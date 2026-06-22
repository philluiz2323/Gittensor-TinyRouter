"""Lightweight ~10K-parameter head over the encoder feature.

The FLAT parameter vector of this head is exactly what separable CMA-ES
optimizes (src/trinity/optim/sep_cmaes.py). Outputs a distribution over
(model x role) actions plus an optional stop action.

TODO(SPEC §3): exact head architecture (linear vs MLP), input feature layout,
and confirmation that the parameter count is ~10K.
"""
from __future__ import annotations
# Placeholder — implemented once docs/SPEC.md pins the head shape.
