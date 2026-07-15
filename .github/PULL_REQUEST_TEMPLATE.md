## Type

**Only routing head submissions that improve accuracy earn TAO via Gittensor.**

- [ ] **Routing head submission** — I trained a coordinator head and want it evaluated for the leaderboard (earns TAO if merged)
- [ ] General improvement — bug fix, docs, refactor, or infrastructure (welcome but earns no TAO)

---

## Routing head submission

*Fill this section if you checked the box above. Otherwise delete it.*

**Benchmark:** composite (one head across math500 + mmlu + livecodebench)

**Miner name:** (your miner identity)

**Generation:** (submission number)

**Training method:** (CMA-ES or other)

**Training cost:** $XX.XX (from cost ledger)

**Notable techniques:** (anything unusual about your training approach)

### Submission files
<!-- The maintainer runs scripts/pr_eval.py on these files -->

- [ ] `head_weights.npy` — linear head W (6 × 1024) float32
- [ ] `svf_scales.npy` — SVF singular-value scales (7168,) float32
- [ ] `receipt.json` — training metadata (cost, generations, fitness curve)

### Rules acknowledgment

- [ ] I trained this head myself — it is not a copy of another miner's submission
- [ ] I understand the hidden benchmark is never revealed and the maintainer's decision is final
- [ ] I understand that if my score does not beat the current best, this PR will be closed without merge
- [ ] I understand that only merged routing head PRs earn TAO

---

## General improvement

*Fill this section if your PR is a general improvement. Delete the routing head section above.*

**What does this PR do?**

<!-- Brief description -->

**Why is it needed?**

<!-- What problem does it solve? -->

**Checklist**

- [ ] Tests pass: `pytest tests/`
- [ ] Lint passes: `ruff check src/`
- [ ] Type check passes: `mypy src/` (for changed files)
- [ ] New code follows existing style
- [ ] Public functions have docstrings and type annotations
