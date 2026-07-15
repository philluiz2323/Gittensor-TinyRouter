"""Frozen constants for routing-head submission validation."""

from __future__ import annotations

# Head W is 6 roles × 1024 hidden; SVF scales are 7 × 1024 singular values.
EXPECTED_HEAD_PARAMS: int = 6 * 1024
EXPECTED_SVF_PARAMS: int = 7 * 1024
EXPECTED_TOTAL_PARAMS: int = EXPECTED_HEAD_PARAMS + EXPECTED_SVF_PARAMS
EXPECTED_HEAD_SHAPE: tuple[int, int] = (6, 1024)

# Head rows [0:N_HEAD_MODELS) are agent logits, the rest are role logits. Each
# group is argmax/softmax'd independently, so routing is invariant to a per-group
# additive shift of the rows -- the duplicate gate must compare heads accordingly.
N_HEAD_MODELS: int = 3

MIN_TRAINING_COST_USD: float = 15.0
MAX_WEIGHT_MAGNITUDE: float = 1e6
DUPLICATE_HEAD_COSINE_THRESHOLD: float = 0.999

RATE_LIMIT_WINDOW_DAYS: int = 7
RATE_LIMIT_MAX_SUBMISSIONS: int = 1

# ---- Composite competition ----
# The competition evaluates ONE head across all three benchmarks.
# A new king must beat the previous king's composite by >= WIN_MARGIN.
COMPETITION_BENCHMARKS: tuple[str, ...] = ("math500", "mmlu", "livecodebench")
WIN_MARGIN: float = 0.02  # 2 percentage points — above the n=120 eval noise band

# Receipt cost vs verified ledger may differ by rounding across many rows.
LEDGER_RECEIPT_COST_TOLERANCE_USD: float = 0.05

DEFAULT_POOL_MODELS: tuple[str, ...] = (
    "qwen3.5-35b-a3b",
    "minimax-m3",
    "deepseek-v4-flash",
)
