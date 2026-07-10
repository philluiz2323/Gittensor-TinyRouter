# Glossary: Ceiling Metrics

Definitions for the oracle-ceiling diagnostic (`scripts/oracle_ceiling.py`; design in
[`ORACLE_CEILING_DIAGNOSTIC.md`](ORACLE_CEILING_DIAGNOSTIC.md)). Together these metrics answer one
question: *is there enough routable complementarity in the model pool that a better router could
beat the best single model, or is the ceiling so close to best-single that only the pool matters?*

Every metric derives from `p[q, m]` — the estimated probability that model `m` solves query `q`,
averaged over `K` independent samples (`p_hat` in code).

## The three headline metrics

### `routing_oracle` — the ceiling

`mean_q max_m p[q, m]`, winner's-curse-debiased by a split-half cross-fit. The best an ideal
query-conditional deterministic router could achieve: for each query it picks the single model most
likely to solve it. This is the honest, routing-*achievable* ceiling. It is floored at
`best_single_crossfit` — a perfect router can always fall back to the best fixed model, so the true
oracle is mathematically `>= best_single`.

### `routing_headroom` — the verdict number

`routing_oracle - best_single_crossfit`. How much accuracy a perfect router could add over the best
fixed model. **It is measured from the cross-fit baseline, not the full-K `best_single`.** Both
terms must share one estimation regime, or the difference is contaminated: an oracle built on the
held-out half compared against a full-K best-single yields a spurious negative headroom on
otherwise-identical pools. Read the verdict off this metric's bootstrap CI, never the point estimate.

### `router_gap_closed` — how good the trained router is

`(trinity - best_single_crossfit) / (routing_oracle - best_single_crossfit)`. The fraction of the
real, achievable headroom the trained router actually captures. It is **`NaN` when the denominator
is `<= 1e-9`** (no achievable headroom -> the ratio is undefined, not zero), so callers must guard
on the headroom CI before trusting it. Numerator and denominator both use `best_single_crossfit`;
mixing in the full-K `best_single` can make the denominator negative and silently flip the sign — a
router *below* the baseline would then report a large positive capture.

## Supporting terms

| Term | Definition | Meaning |
|---|---|---|
| `best_single` | `max_m mean_q p[q, m]` (full-K) | Best fixed single model; the per-model reporting baseline. |
| `best_single_crossfit` | `best_single` evaluated on the held-out cross-fit half | The routing baseline that `routing_headroom` and `router_gap_closed` are measured from; shares the oracle's estimation regime. |
| `routing_oracle_naive` | `mean_q max_m p[q, m]` (no cross-fit) | Upward-biased oracle (winner's curse on the `max`). Reported only to quantify the bias against `routing_oracle`. |
| `clairvoyant_any` | `mean_q (1 - prod_m (1 - p[q, m]))` | Probability that *some* model solves the query. An optimistic upper bound a single-pick router can **not** reach — it measures noise, not routable opportunity. |
| `unroutable_noise` | `clairvoyant_any - routing_oracle` | Apparent gain that exists only from luck / nondeterminism and is **not** capturable by routing. |

## Reproduce the numbers

Every metric above is emitted by the analyzer for a collected correctness matrix:

```bash
python scripts/oracle_ceiling.py --analyze experiments/final/oracle_matrix_math500.json
```

The report's `point_estimates` block carries `best_single`, `best_single_crossfit`,
`routing_oracle`, `routing_headroom`, and `unroutable_noise`. `router_gap_closed` appears under the
`trinity` block when a per-query TRINITY correctness file is supplied via `--trinity-per-query`. The
verdict (`pool-bound` / `router-bound` / `near-ceiling`) is read off the bootstrap CIs, per
[`ORACLE_CEILING_DIAGNOSTIC.md`](ORACLE_CEILING_DIAGNOSTIC.md) section 6.
