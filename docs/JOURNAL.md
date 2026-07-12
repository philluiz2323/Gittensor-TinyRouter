# JOURNAL — TRINITY replication lab notebook

This is the running log of **mistakes, findings, and decisions**. See `AGENTS.md` §6 for the
protocol. **Newest entries at the top.** Tag each entry with one or more of:
`#mistake` `#finding` `#decision` `#repro` `#gotcha` `#todo`.

### Entry template

```
## YYYY-MM-DD — short title  #tag #tag
**Context:** what we were doing.
**Expected:** what we thought would happen.
**Actual:** what happened (paste the error/number).
**Root cause:** why.
**Fix / decision:** what we changed and why.
**Follow-up:** anything left open.
```

---

## 2026-07-11 — dataset-quality audit missed a None prompt and mis-flagged it as a duplicate  #mistake #gotcha #repro

**Context:** reading the new `trinity.dataset_quality.audit_dataset` (the offline data-quality audit
of a built benchmark, #189).
**Expected:** an item with no usable prompt is counted as `missing_prompt` — the audit exists to
catch "an unscoreable item that still counts toward the denominator".
**Actual:** the prompt was read as `it.get("question_text", "")`, whose `""` default only applies
when the key is ABSENT. A present `question_text: None` returned `None`, and `str(None)` is the
truthy `"None"`: so `missing_prompt` stayed 0, and two None items both normalized to `"none"` and
were reported as a `duplicate_questions`. Reproduced offline: two `{"question_text": None}` items ->
`missing_prompt=0, duplicate_questions=1` (should be `2, 0`).
**Root cause:** `dict.get(k, default)` returns a stored `None` rather than the default. The sibling
`correct_answer` check on the same loop already guards `if ref is None or ...`, so the prompt path
was inconsistent with the answer path.
**Fix / decision:** read the prompt as `it.get("question_text") or ""` in all three spots, so `None`
is treated as blank exactly like `""` and like the answer path. Added
`tests/test_dataset_quality_none_prompt.py`. Fixes #225.
**Follow-up:** none.

## 2026-07-11 — verify_benchmark crashed on the missing-manifest case it exists to report  #mistake #gotcha #repro

**Context:** reading the new `scripts/verify_benchmark.py` CLI (the offline hidden-benchmark
integrity verifier, #174).
**Expected:** `verify_benchmark.py --dir <build>` reports every integrity problem — including the
most basic one, a build with no `meta.json` — as a clean `FAIL [...] — N problem(s)` report and
`exit 1`. `verify_dir` is written for exactly this: it returns `["missing meta.json"]` early.
**Actual:** `main()` re-read `meta.json` unconditionally right after `verify_dir` and BEFORE the
`if problems:` block (`meta = json.loads((Path(args.dir)/"meta.json").read_text())`). On a build
missing the manifest, that line raised an uncaught `FileNotFoundError` traceback; the intended
`FAIL — missing meta.json` report was never printed. Reproduced offline with an empty dir + dummy
`BENCHMARK_PASSWORD` (crashes before any AES-GCM decryption).
**Root cause:** `meta` is only consumed in the success branch (the `OK [...]` line and `--append`),
but it was read eagerly on the failure path too, defeating `verify_dir`'s clean missing-manifest
handling.
**Fix / decision:** track only `meta_path` per mode and defer the `meta` read to after the
`if problems:` block, where verification has passed so the file exists and is well-formed. Added
`tests/test_verify_benchmark_main.py` driving `main()` (the pure helpers were already covered but
`main()` never was): missing-meta → clean FAIL+exit 1 (regression guard), `--dir` without password
→ exit 2, no mode → exit 2, inconsistent `--meta` → FAIL+exit 1. Fixes #191.
**Follow-up:** none — the `--meta` branch is unchanged in practice (`verify_meta_file` reads that
path first anyway).

## 2026-07-11 — pack_submission generation auto-detect overwrote an existing generation  #mistake #gotcha #repro

**Context:** reading `scripts/pack_submission.py`'s auto-detect of the next submission generation
(the `--generation 0` default path).
**Expected:** packing a new head with auto-detect always lands in a fresh, unused
`submissions/<name>/<gen>/` directory.
**Actual:** `gen = len(existing) + 1` counts directory entries. With gens `1` and `3` present (gen 2
thrown out and deleted — a normal occurrence), the count is `2 + 1 = 3`, so it packed into the
existing `submissions/<name>/3/` and `mkdir(exist_ok=True)` silently **overwrote** that generation's
head_weights.npy / svf_scales.npy / receipt.json. `glob("*")` also counted stray files
(`README`, `.DS_Store`), inflating the count further.
**Root cause:** the auto-detect assumed contiguous numbering and counted entries instead of taking
the max generation. A gap or a non-generation entry shifts the detected number onto an existing one.
**Fix / decision:** extract a pure `next_generation(submissions_dir)` helper that returns
`max(numeric generation dirs) + 1` (or `1` when the dir is missing/empty), considering ONLY
numerically-named subdirectories — so gaps and stray files can never cause a collision. Added
`tests/test_pack_submission_generation.py` (pure pathlib, no torch): gap -> 4, contiguous -> 4,
stray file/non-numeric dir ignored, single high gen 7 -> 8. Fixes #187.
**Follow-up:** none — an explicit `--generation N` still overrides auto-detect unchanged.

## 2026-07-11 — Novelty scored identical heads as maximally novel after a JSON round-trip  #mistake #decision

**Context:** reading `novelty.normalize_decision` in the new novelty/routing-diversity analysis
(issue #164).
**Expected:** a routing decision persisted to JSON and reloaded compares equal to the live
`(agent, role)` tuple — the function's docstring says the key "round-trips through JSON."
**Actual:** it didn't. A normalized tuple key serialized to JSON reloads as a **list** (JSON has no
tuple type), and `normalize_decision` only branched on `tuple` — a list fell through to
`str([...])`, a different key. Two identical heads (one live tuples, one JSON-loaded lists) scored
0.0 agreement / **1.0 novelty**.
**Root cause:** `isinstance(decision, tuple)` misses the `list` shape that every pair takes after a
JSON round-trip, so the element-wise enum→name normalization never ran and the whole list was
stringified.
**Fix / decision:** branch on `(tuple, list)` and normalize element-wise to a tuple, so a
JSON-loaded `[0, "WORKER"]` becomes `(0, "WORKER")` and matches the live tuple. `[OUR CHOICE]`
always return a tuple key (not a list) so the key stays hashable and stable across the round-trip.
Novelty is 5% of the composite score and the reference is the king's persisted decisions, so a
head that routes identically to the king was being handed full novelty credit it did not earn.
**Follow-up:** the `reference is None` branch of `novelty_report` still returns `n_agree=0`
alongside `agreement_rate=0.5`, which is internally inconsistent for any JSON consumer that
recomputes agreement from `n_agree/n_questions`; harmless to the scalar novelty but worth
tidying separately.

## 2026-07-10 — Cross-fit oracle tie-break favoured the LAST model, not model 0  #mistake #decision

**Context:** reading `scripts/oracle_ceiling.crossfit_oracle_and_best` while checking the
routing-headroom diagnostic that backs the R1/R2 verdict (issue #126).
**Expected:** on an argmax tie over the selection half, the oracle picks model 0 — the
docstring says "Argmax ties are broken by a tiny deterministic per-model jitter (model 0
favoured)", and numpy's own `argmax` returns the lowest index on exact ties.
**Actual:** it picked the LAST model. `jitter = np.linspace(0.0, 1e-9, M)` is an *increasing*
ramp, so `argmax(sel + jitter)` adds the largest nudge to the highest index and overrides
numpy's natural lowest-index tie-break.
**Root cause:** wrong ramp direction. The jitter was meant to make near-ties deterministic
while preserving model-0 preference, but increasing it inverts that preference.
**Fix / decision:** `jitter = np.linspace(1e-9, 0.0, M)` (decreasing), so ties go to model 0 as
documented. `[OUR CHOICE]` fix the code, not the docstring: model-0 is both the stated intent and
numpy's convention. This is load-bearing, not cosmetic — selection-half ties are frequent at low
K (`n_a = K//2` makes `sel` land on a coarse grid like {0, 0.5, 1}), and every tie was silently
routed to the last-indexed pool model, injecting a pool-ordering bias into `routing_oracle` /
`routing_headroom`. On a constructed tie-heavy input the raw oracle read 0.506 vs the correct
1.0.
**Follow-up:** the tie-break still favours a fixed index, which is fine for determinism but means
the *raw* oracle is not pool-order-invariant on ties; `compute_stats` already floors the oracle at
the cross-fit best_single, which masks most of the residual. A fully order-invariant tie-break
(e.g. average over tied models) would be a larger change, deferred.

## 2026-07-10 — Rate-limit gate self-rejected a re-run of the same PR  #mistake #decision

**Context:** reading the anti-cheat Gate 1 (`submission/gates.check_rate_limit`, called by
`scripts/pr_eval.py`) after the submission-gate extraction (issue #144).
**Expected:** the maintainer can re-run `pr_eval` on a PR (a CI retry, or after a transient
GPU/API failure during live scoring) without it counting as a second submission.
**Actual:** `_record_attempt` consumes the weekly slot the moment Gate 1 passes ("so rejected
attempts still count — miners can't probe weekly"). On a re-run, `check_rate_limit` counted that
already-recorded attempt for the SAME PR (recent = 1 >= max 1) and rejected the PR as
`rate_limited`. A transient infra failure permanently burned a legitimate miner's slot AND
bounced their submission.
**Root cause:** `check_rate_limit` counted every attempt by the miner in the window with no
notion of "the PR currently under evaluation", even though each attempt already records its `pr`.
**Fix / decision:** thread the current PR number into `check_rate_limit` and skip attempts whose
`pr` equals it — a re-eval of one PR no longer counts against itself, while a DISTINCT PR still
does (anti-probe intent preserved). `_record_attempt` is now idempotent per PR so re-runs don't
bloat the ledger either. `[OUR CHOICE]` an attempt with no recorded `pr` still counts (it can't be
proven to be the same PR); `current_pr=None` (local preflight, no PR yet) preserves the old
count-everything behaviour so a miner sees their true status.
**Follow-up:** the slot is still consumed on the *first* Gate-1 pass regardless of whether live
eval ever completes; if an eval reliably dies before producing a score, that PR's slot is spent
until it either succeeds on re-run or the week rolls over. Acceptable given the anti-probe goal,
but worth revisiting if infra flakiness makes it common.

## 2026-07-10 — Preflight gates 1–5 did not catch receipt schema drift or theta layout corruption  #finding #decision
**Context:** extending ``trinity.submission`` after #104 merged the offline preflight CLI.
**Expected:** miners discover wrong ``receipt.json`` benchmark fields or hand-edited
weight packs before opening a PR.
**Actual:** gates 1–5 checked rate limits, weight magnitudes, duplicates, receipt
plausibility, and ledger/receipt cost — but not whether the receipt's
``benchmark`` / ``pool_models`` / ``n_total`` matched the submission context, or
whether ``head_weights.npy`` + ``svf_scales.npy`` round-tripped through the
canonical ``coordinator.params`` θ layout.
**Fix / decision:** add gate 6 (``pack_schema``) and gate 7 (``theta_integrity``)
in ``trinity.submission.schema``, wire them into ``OFFLINE_GATES``,
``scripts/preflight_submission.py``, and ``scripts/pr_eval.py`` before any GPU
work. Document the seven-gate flow in ``SUBMITTING.md``. Covered by
``tests/test_submission_schema.py``.
**Follow-up:** none.

## 2026-07-10 — The hidden-benchmark build accepted toy data, then died pointing elsewhere  #mistake #gotcha #repro
**Context:** #73 fixed MMLU's `train` split and wired `ToyFallbackWarning` into `load_split`. Checking what the *hidden-benchmark builder* does when that warning fires.
**Expected:** `build_benchmark.py` refuses to seal a benchmark whose questions came from the 2-item offline toy set.
**Actual:** it accepts them. `protocol.sample_pool` draws from the `"train"` split, so with `datasets` unavailable (offline box, or the gated `Idavidrein/gpqa`) the pool is the toy set. A warning is emitted — and warnings do not stop anything. The build then dies much later, in `select_splits`:
```
_sample_pool('mmlu')  -> 2 tasks  (ToyFallbackWarning emitted)
select_splits         -> ValueError: pool has 2 tasks but the protocol needs 220 (eval=150, audit=50, live=20)
```
That error names neither the toy fallback nor the split that failed to load. Someone reading it would go hunting in `benchmark_protocol.py`, which is entirely innocent.
**Root cause:** the loaders correctly *report* the substitution, but the only consumer for whom toy data is categorically unacceptable — the sealed, integrity-hashed hidden benchmark — never listened. `select_splits`' size check caught it by accident, one stage too late, because 2 < 220. A benchmark whose toy set happened to exceed the protocol's counts would have been sealed and hashed as real data with nothing but a warning on stderr.
**Fix / decision:** escalate `ToyFallbackWarning` to an error inside `_sample_pool` (`warnings.simplefilter("error", ToyFallbackWarning)`), and re-raise it as a `RuntimeError` that names the benchmark, quotes the original warning, and states the remedy. The original warning is preserved as `__cause__` rather than swallowed. Chose this over threading an `allow_toy_fallback=False` flag up through `BenchmarkAdapter.load_tasks`: escalating the warning reuses the signal #73 already added, changes no interface, and cannot be forgotten by the next adapter — any loader that warns is covered for free.
**Follow-up:** `trinity.train` has the same exposure — training on 2 toy questions produces a normal-looking receipt — but it goes through `load_tasks` rather than `_sample_pool`, so it needs its own decision about whether an offline smoke run should stay permitted. Deliberately out of scope here.
## 2026-07-11 — Audit random-routing baseline was non-reproducible (shared rng under asyncio.gather)  #mistake #finding #decision

**Context:** `scripts/audit_eval.py` is the SEALED, run-once "honest, ungameable"
number. Its random-routing baseline averages 100 seeds; each seed fans all tasks
out through `asyncio.gather`.
**Expected:** a fixed seed → a byte-reproducible `random_routing` score.
**Actual:** it wasn't. `_RandomAuditPolicy` held one shared `self.rng` and every
concurrently-gathered trajectory drew from it, so turn-2+ routing draws were
consumed in **network-completion order**, not seed order — the exact bug
`trinity.eval.RandomPolicy` was already fixed for (via `task_rng(seed, task_id)`),
but the audit script kept the old shared-rng form.
**Root cause:** `decide` used `self.rng` and ignored the per-trajectory `rng`
`run_trajectory` passes through; the audit loop never supplied one.
**Fix / decision:** mirror `trinity.eval` — `decide` draws from the passed `rng`
(instance rng only as a fallback), and the loop passes `rng=task_rng(seed_s,
t.task_id)` per trajectory. Covered by `tests/test_audit_random_routing_seed.py`.
**Follow-up:** the audit "held-out" guarantee is soft (samples train w/ a diff seed,
not a provably-disjoint partition) — a larger, separate change.

## 2026-07-10 — Miners had no offline path to run pr_eval gates before opening a PR  #finding #decision
**Context:** routing-head submissions were rejected only after opening a PR, when
``scripts/pr_eval.py`` ran gates 1–4 embedded in an 850-line maintainer script.
**Expected:** the same anti-cheat checks (rate limit, weights, duplicate head,
receipt plausibility) runnable locally with no GPU and no OpenRouter spend.
**Actual:** gate logic was not importable; miners discovered failures post-PR. A
fifth gap also existed: ``receipt.json`` ``total_cost_usd`` was never
cross-checked against a verified ``TRINITY_COST_LEDGER`` total, so fabricated
receipts could disagree with the append-only ledger.
**Fix / decision:** add ``trinity.submission`` (pack loader, gate classes,
``PreflightRunner``) plus ``scripts/preflight_submission.py`` for miners.
``pr_eval.py`` now imports the shared gates and runs gate 5
(ledger/receipt cost consistency) before any GPU work. Shared OpenRouter pricing
lives in ``trinity.llm.openrouter_pricing`` so ``cost_report.py``,
``pack_submission.py``, and the new gate agree on dollar totals. Covered by
``tests/test_submission_preflight.py``.
**Follow-up:** wire the preflight CLI into CONTRIBUTING/SUBMITTING docs when the
maintainer is ready to advertise it.

## 2026-07-10 — A null `usage` block crashed a successful inference call  #mistake #gotcha
**Context:** hardening `llm/openrouter_client.py` after #72 fixed the `content: null` case, to see whether the same present-but-null trap existed elsewhere in the response parsing.
**Expected:** a 200-OK response with `"usage": null` records zero tokens and returns the completion.
**Actual:** `usage = data.get("usage", {})` returns `None` (the `{}` default only applies to an *absent* key, not a present-null one), and the next line `usage.get("prompt_tokens", 0)` raises `AttributeError: 'NoneType' object has no attribute 'get'` — on an otherwise-successful call.
**Root cause:** the identical `dict.get(k, default)` misuse #72 fixed for `content`, one function away, in the token-accounting path — missed because `_message_text` and the usage parse were treated as separate concerns. OpenAI-compatible providers send `"usage": null` for some backends and for 200s with an empty completion.
**Why it is worse than the `content` case:** `_message_text` returned a wrong *value* ("None"); this *raises*. The `AttributeError` is not in the `_Retryable` set (only 429/5xx and network errors are), and `@retry(..., reraise=True)` re-raises it out of `chat`. With `eval`/`fitness` now gathering `return_exceptions=True`, that exception becomes the trajectory's result and is scored **0.0** — so a model that answered correctly is silently counted as wrong, in both eval and CMA-ES training reward.
**Fix / decision:** `usage = data.get("usage") or {}` — covers absent and null identically, exactly mirroring the `if content is None` guard #72 added in the same file. One line; no behaviour change for a populated, empty, or absent usage block.
**Follow-up:** `choice = data["choices"][0]` and `choice["message"]` are still index/`[]` access, so a response with no `choices` would `KeyError`. That is a genuinely malformed response (not a documented provider behaviour like `usage: null`), so I left it — a separate, weaker concern.

## 2026-07-10 — Added an offline view of efficiency and composite-score tradeoffs  #finding #decision
**Context:** contributors could see hidden/live accuracy, but the competition's 10% efficiency term still lived only inside `scripts/pr_eval.py::_compute_score`, making turn-efficiency tradeoffs hard to inspect offline.
**Expected:** a miner can estimate the composite score and inspect turns-per-correct-answer without opening a PR or touching the hidden evaluator.
**Actual:** there was no repo-local utility for that analysis; the formula existed only in the maintainer scorer.
**Fix / decision:** add `src/trinity/efficiency.py` plus `scripts/efficiency_report.py` as an offline mirror of the current score formula, with per-answer efficiency summaries (`turns_per_correct`, optional calls/cost per correct) and tests that pin the implementation to `pr_eval` when importable.
**Follow-up:** if the competition scoring formula changes, `pr_eval` and `trinity.efficiency` must be updated together so offline analysis stays aligned with the maintained scorer.

---

## 2026-07-10 — The head never read `<Head Input>`; the EOS trick was a no-op  #mistake #gotcha #repro
**Context:** verifying that `coordinator/slm.py::encode` matches the canonical extraction in SPEC §3.2 before trusting any head trained on it.
**Expected:** `encode` tokenizes `transcript + "\n<Head Input>"`, appends one EOS, and reads index `-2` — the suffix's final token, a fixed decision position.
**Actual:** the suffix was never appended. The string `<Head Input>` appeared nowhere in `src/`. `encode` tokenized the bare transcript, appended EOS, and read `-2` — which is the transcript's **last content token**. Its own comment asserted the opposite: *"The last real content token therefore sits at index -2 (the `<Head Input>` position)."* Both cannot be true. Demonstrated with a char-code tokenizer and an echo model:
```
before:  transcript ends '2' -> head reads '2'     ends 'x' -> reads 'x'   ends 's' -> reads 's'
after :  transcript ends '2' -> head reads '>'     ends 'x' -> reads '>'   ends 's' -> reads '>'
```
**Root cause:** two changes that each look right in isolation. Appending EOS *does* create a penultimate position — but **attention is causal**, so the hidden state at `-2` cannot attend to the EOS at `-1`. `h[-2]` with an EOS appended is bit-identical to the last content token's state with no EOS at all (pinned by `test_appending_eos_is_a_noop_under_causal_attention`, which builds a minimal causal block and asserts the equality). The EOS append was therefore a **no-op**: it renamed the last content token as "penultimate" without changing which vector the head sees. Only the suffix creates a position whose identity is independent of the transcript.
**Why it matters:** the head's sole input (SPEC §3.2: *"no pooling, no turn index, no role one-hots"*) was a token that varies with whatever the transcript happened to end on — a code brace, a digit, a period. SPEC §3.2 records the ablation for reading a content token rather than the intended one: **LiveCodeBench 61.46 → 50.85**.
**Fix / decision:** append `HEAD_INPUT_SUFFIX = "\n<Head Input>"` before tokenizing, and export it as a module constant so train and inference cannot drift. Keep the EOS append and the `-2` read exactly as SPEC prescribes — with the suffix present, `-2` is now the suffix's final token.
**Consequence to be explicit about:** this changes the coordinator's feature. `encode` is called by both training and evaluation, so the two stay consistent with each other — but any head trained *before* this fix was fitted to the old (last-content-token) feature. Those heads are not comparable to heads trained after it, and the leaderboard's archived `best_theta` files were fitted to the wrong feature. This wants a re-train, exactly as the 2026-06-23 extraction fix did.
**Follow-up:** the guard `input_ids.shape[1] < 2` can no longer trigger, since the suffix alone tokenizes to several tokens; left in place as a cheap invariant. Worth a follow-up: assert at load time that the tokenizer does not merge the suffix's final `>` into a preceding token for some other checkpoint.

---
## 2026-07-10 — R1/R2 gave TRINITY 5x the token budget of the baselines it beat  #mistake #gotcha
**Context:** auditing `trinity/eval.py` against SPEC §1.3 before trusting an R1/R2 verdict.
**Expected:** the single-model baselines are budget-matched to TRINITY, as SPEC §1.3.4 requires: *"run each single model at `max_tokens = 20,480` (5×) so the single-vs-TRINITY comparison is fair, matching the paper's 5× protocol."* The same 5× appears in the 2026-06-22 SPEC-decisions entry and in SPEC's own R1 row (*"budget-matched 5×"*).
**Actual:** the baselines got **1×**. `evaluate` passed `max_tokens=args.max_tokens` (default 4096) to `_score_single_model`, which spends it on **one** turn. TRINITY got `run_kwargs = dict(max_turns=5, max_tokens=4096)` — up to 5 turns at 4096 each, i.e. 20,480. So every "TRINITY > best single model" result was produced by a system with five times the token budget of the systems it was compared against.
**Root cause:** the budget rule lives in `--max-turns`, not in `--max-tokens`, so "give the baseline the same total" requires multiplying the two — and the one place that had to do the multiplication simply forwarded `args.max_tokens` to both paths. Both numbers look right in isolation. Nothing in the code referenced the 5×, so nothing enforced it; the constraint existed only in prose, in two documents.
**Fix / decision:** add `single_model_budget(max_tokens, max_turns) -> max_tokens * max_turns` and hand its result to the baselines. Derived from `max_turns` rather than hard-coding `5`, so the match survives someone passing `--max-turns 3`; at the defaults it is exactly the 20,480 SPEC names. The routed path is untouched — TRINITY still spends `max_tokens` *per turn*. `evaluate` now prints the matched budget, so a reader of the logs can see the comparison was fair rather than trusting that it was.
**Why this one stings:** R1/R2 is the paper's headline claim and the repo's reason to exist. The invariant check at the bottom of `evaluate` (`s_trinity > best_single`) was measuring, in part, a budget difference. Every R1/R2 number in this JOURNAL predating this fix was produced under the 1× baseline and should be read with that in mind — including the 2026-06-23 multi-task headline (*"best FIXED single model avg ≈ 0.65 … per-task TRINITY avg ≈ 0.75"*). Re-running eval is cheap (the JOURNAL prices it at ~$1.3); the numbers should be regenerated before any of them are quoted.
**Follow-up:** `scripts/audit_eval.py` builds its own `single::` baselines and needs the same treatment; `pr_eval.py`'s cached single-turn component is a separate contract (it caches one WORKER turn per question) and is deliberately not changed here.

---

## 2026-07-10 — Verifier verdicts missed when the model wraps them in Markdown  #mistake #finding #decision

**Context:** `roles/verifier.py::parse_verdict` extracts `VERDICT: ACCEPT|REVISE`
from the Verifier turn; an ACCEPT terminates the trajectory early (SPEC §0.3.5).
The regex was recently anchored with a trailing `\b` (good — stops `ACCEPTABLE`).
**Expected:** a verdict still parses when the model emphasises it.
**Actual:** it did not. Models routinely format the line as `**VERDICT:** ACCEPT`,
`VERDICT: **ACCEPT**`, or ``VERDICT: `REVISE` ``, and `VERDICT:\s*(ACCEPT|REVISE)`
requires the colon then only whitespace before the word — so all three returned
`None`, the loop fail-safed to REVISE, and a correct+complete answer never earned
the early ACCEPT. That needlessly runs the full turn budget, hurting the
efficiency term (10% of the competition score) and raising live-eval latency/cost.
**Root cause:** the separator between `VERDICT` and the verdict word only allowed
whitespace, not Markdown emphasis / code / dash markers.
**Fix / decision:** broaden the separator to `[\s:*_`~-]*` and replace the trailing
`\b` with `(?![A-Za-z])`. The lookahead is needed because `\b` treats an underscore
as a word char and would reject the italic wrapper `__REVISE__`; the lookahead
blocks only a trailing *letter*, so the `ACCEPTABLE`/`ACCEPTED`/`REVISED` guard is
preserved while `**`/`__`/`` ` ``/punctuation are fine. The character class matches
no letters, so prose ("the verdict is … accept") is still rejected. Covered by new
cases in `tests/test_verifier.py`.
**Follow-up:** none.

## 2026-07-10 — results_table multi-task summary crashed on a null system score  #mistake #gotcha

**Context:** `scripts/results_table.py` aggregates `experiments/**/eval*.json` into
the R1/R2/R4 summary. `load_rows` keeps a row when the `TRINITY` and `single::` keys
are PRESENT.
**Expected:** the summary renders for any row set `load_rows` accepts.
**Actual:** `trin_avg = sum(max(r["trinity"] for r in by_bench[b]) ...)` (and the
`random` twin) raise `TypeError: ... 'float' and 'NoneType'` when a row has
`"TRINITY": null` (or lacks `random_routing`) — key present but value `None`. Key
*presence* was checked; value *nullness* was not, so an older/partially-written
eval file crashes the whole summary.
**Root cause:** the per-benchmark reduction assumed every value non-null, unlike the
per-row table which already guards with `x or 0`.
**Fix / decision:** add `_bench_best(rows, key)` that maxes over non-null values
(0.0 if a benchmark has none), and route both `trin_avg`/`rand_avg` through it —
mirroring the per-row leniency. Covered by a null-score case in
`tests/test_results_table.py`; the existing reduce-the-same-way tests are unchanged.
**Follow-up:** none.

## 2026-07-10 — GPQA logical `test` resolved via a deterministic holdout  #finding #decision

**Context:** closing the follow-up left by the MMLU split fix below — "GPQA still has only an
upstream `train` split; deterministic holdout for logical `test` is separate work" (issue #95).
**Expected:** `python -m trinity.eval --benchmark gpqa` scores the router on real GPQA-Diamond
rows that training never saw.
**Actual:** it scored on **2 toy questions**. `eval.py` asks for split `"test"`; `Idavidrein/gpqa`
publishes only `train`; `_try_load_hf` swallowed the unknown-split error and `load_split`
substituted `_toy_tasks("gpqa")`. Training (`split="train"`) loaded the real 198 rows, so train
and eval were silently running on different data and the R1/R2 verdicts rested on 2 questions.
**Root cause:** `split_policy._SPLIT_ALIASES` had entries for `mmlu` (#35) and `mmlu_pro` (#50)
but none for `gpqa`, so the logical split was forwarded verbatim.
**Fix / decision:** alias both `train` and `test` onto upstream `train`, then partition those rows
with `split_policy.select_holdout` — a fixed-seed (`HOLDOUT_SEED = 20260710`), 25% holdout keyed on
upstream row position. `[OUR CHOICE]` a plain `test → train` alias was rejected: it would have
evaluated on exactly the rows training consumed. The partition is deliberately independent of
`load_split`'s shuffle `seed`, so the train/test boundary cannot drift when a caller changes
sampling. 198 rows → 148 train / 50 test, disjoint and covering. The toy fallback skips the
partition (a 2-item set cannot be divided) and still raises `ToyFallbackWarning`.
**Follow-up:** `eval.py` treats `ToyFallbackWarning` as non-fatal, so any *other* loader failure
still reports toy-set numbers as if real. Promoting that warning to an error under `--strict-data`
would close the class rather than this one instance. GPQA is also a gated HF repo — without auth
the fallback still fires, now loudly but not fatally.

## 2026-07-10 — Cost-ledger verifier hashed a different JSON string than the writer  #mistake #decision
**Context:** checking the token-cost ledger path used by training, `cost_report.py`,
and submission packing.
**Expected:** a ledger written by `OpenRouterPool._ledger_append` verifies in
`scripts/cost_report.py --ledger`, and the submission packer prices the same
entries from the same parsed rows.
**Actual:** legitimate ledgers failed verification. The writer hashed a fixed,
compact payload string (`{"m":"...","p":...,"c":...}` in that key order), while
the verifier rebuilt the payload with `json.dumps(..., sort_keys=True)`, which
changes the byte string and therefore the hash. `pack_submission.py` then read
the same file through a separate ad-hoc path, so the write, verify, and summarize
steps were not sharing one canonical implementation.
**Root cause:** the hash-chain format existed only implicitly inside the writer.
Verifiers reimplemented it differently, and nothing enforced one shared payload
encoding.
**Fix / decision:** add `trinity.llm.cost_ledger` as the single source of truth
for payload formatting, chained hashing, line parsing, ledger verification, and
append helpers. Route `openrouter_client`, `cost_report.py`, and
`pack_submission.py` through it, and cover the regression with
`tests/test_cost_ledger.py`.
**Follow-up:** the append helper writes text-only handles; keeping the test hook
typed as `TextIO` avoids implying binary support the implementation does not have.

## 2026-07-10 — Rate-limit gate counted wins, not attempts  #mistake #finding #decision

**Context:** follow-up from the UTC timestamp fix on Gate 1 (`_check_rate_limit`).
`SUBMITTING.md` says "1 submission per benchmark per week".
**Expected:** any evaluated submission consumes the weekly slot, win or lose.
**Actual:** `_update_leaderboard` (the only writer into `history`) ran only on
`score > best_score`. Score-rejections and later gate failures never touched the
log, so Gate 1 only saw prior *wins*. A miner could lose, immediately resubmit,
and probe the hidden benchmark without waiting 7 days.
**Root cause:** rate limit reused the win-only `history` log instead of an
attempt log.
**Fix / decision:** record an `attempts` entry as soon as Gate 1 passes (slot
consumed even if Gate 2+ fails or the score loses). Gate 1 reads `attempts`,
falling back to legacy `history` when `attempts` is absent. First write seeds
`attempts` from existing `history` so recent winners stay rate-limited after
rollout. Covered by `tests/test_pr_eval_rate_limit.py`.
**Follow-up:** none for this hole.

---

## 2026-07-10 — Passing a price table silently made the Conductor free  #mistake #gotcha
**Context:** checking the pre-launch projections in `fugu/cost.py` before trusting them to size a paid GRPO run. The module's stated job is to stop us launching a paid job blind.
**Expected:** `conductor_local=False` + `conductor_model="minimax-m3"` prices the Conductor's generation, whichever way the worker prices were supplied.
**Actual:** passing `prices=` billed the Conductor **$0**, while the returned `assumptions` still said `conductor_local: False` — an internally self-contradictory estimate.
```
prices=None (default table)   conductor_api_usd = 1.5
prices=dict(PRICES) passed    conductor_api_usd = 0.0     # same config
assumptions.conductor_local (passed table): False
```
**Root cause:** `conductor_local` and `conductor_model` were consumed **only inside `price_table`**. `estimate_grpo_cost` did `table = prices if prices is not None else price_table(...)`, so an explicit table skipped that branch and both knobs were ignored. A worker price table carries no `"<conductor>"` key, so `table.get(CONDUCTOR_KEY, (0.0, 0.0))` returned zeros and the Conductor cost nothing. The rule for *when the Conductor is billed* lived in one function while the *decision to bill it* was taken in another.
**Fix / decision:** extract that rule into `_conductor_price(lookup, conductor_model, conductor_local)` and call it from both `price_table` and the estimator. An explicit `prices` table stays authoritative for the **worker** models — no caller passes one today, so nothing depends on the old replace-everything semantics — but it no longer disables Conductor pricing: the entry is derived unless the caller supplied `CONDUCTOR_KEY` themselves, and the model's rate resolves against `PRICES` overlaid with the caller's table. The caller's dict is copied, not mutated.
**Why it mattered:** the prompted-Conductor baseline is exactly the `conductor_local=False` configuration, and it makes one Conductor call per rollout. At the module's own defaults (200 iterations × 4 questions × group size 64 = 51,200 rollouts) the estimate dropped an entire cost component. An under-stating projection is worse than none: this JOURNAL already records runs killed mid-flight on cost ($0.50, $1.59, ~$22 ledgered).
**Follow-up:** `run_cost` (the *exact*, post-hoc accounting) also takes `prices` and defaults to `price_table()` with a local Conductor. That is correct for its purpose — it prices observed per-model token totals and never looks up `CONDUCTOR_KEY` — but the two functions' `prices` parameters now mean subtly different things, which is worth a docstring note if a third caller appears.

## 2026-07-10 — Math grader false-negatives on LaTeX-grouped thousands (`1{,}000`)  #mistake #finding #decision

**Context:** the math grader (`orchestration/reward.py`) already strips *bare*
digit-grouping commas (`1,000` -> `1000`). Checking whether MATH-500's other common
thousands form is handled.
**Expected:** `\boxed{2{,}048}` grades equal to `2048`.
**Actual:** it graded **wrong** (score 0.0). MATH-500 frequently writes thousands
with LaTeX's `{,}` group (which renders as a comma): `\boxed{1{,}000}`,
`\boxed{2{,}048}`, `\boxed{1{,}234{,}567}`. The braces defeat both halves of the
grader — `extract_last_number` split `1{,}000` into `1` and `000` (returning
`000`), and `normalize_math_answer`'s comma-strip only matched a bare comma, so the
braced form survived and never equalled the plain reference.
**Root cause:** `{,}` was never normalised to a bare comma, so neither the
thousands-separator regex nor the comma-strip saw it.
**Fix / decision:** replace `{,}` -> `,` early in both `extract_last_number` and
`normalize_math_answer`, so the existing (already-tested) comma handling removes it.
Additive to the bare-comma logic; a comma-separated *list* (`1,2,3`) and genuinely
wrong answers stay wrong (no false positives). Covered by
`tests/test_reward_latex_thousands.py`.
**Follow-up:** none. (`1\,000` — the `\,` thin-space form — is already handled in
`normalize_math_answer`'s token strip.)

## 2026-07-10 — `main` went red: cache-prompt test not updated for the `_cache_answers` refactor  #mistake #gotcha

**Context:** two changes landed close together — the cache-prompt fix (which added
`tests/test_build_benchmark_cache_prompt.py`, asserting `_cache_answers` uses a
WORKER turn) and the #62 refactor that routes caching through the adapter registry.
**Expected:** green `main`.
**Actual:** `pytest` fails 2/2 in `test_build_benchmark_cache_prompt.py`. #62 changed
`_cache_answers(items, ...)` to `_cache_answers(task_item_pairs, ...)` — it now takes
`(task, item)` pairs and renders the prompt via `get_adapter(task.benchmark)
.build_prompt(task)` — but the test still called the old `(items, ...)` signature.
**Root cause:** the test encoded the *old* call shape; the refactor updated the
function and its caller but not this test, and with no CI gate the break merged.
**Fix / decision:** update the test to the `(task, item)` pair API and assert the
prompt equals `build_messages(Role.WORKER, adapter.build_prompt(task), [])` (still
pinning the WORKER-turn behaviour, now through the adapter). The `_cache_answers`
WORKER-turn fix itself is intact after #62; only the test was stale. Full suite
green again.
**Follow-up:** the offline PR CI in #52 would have caught this; worth landing.

## 2026-07-10 — govern job 403 on fork PR label write-back  #mistake #decision

**Context:** PR-bot governance (`pr-bot.yml`) from #51; every fork PR failed the
`govern` check on `ensure_labels.py` (issue #84, part 1). Routing-template false
positives were fixed separately in #86 (issue #85).
**Expected:** fork PRs run deterministic analysis even when `GITHUB_TOKEN` cannot
create labels or post comments.
**Actual:** `ensure_labels.py` raised on HTTP 403; `run_pr_bot.py` returned exit 1
when label write-back was rejected.
**Fix / decision:** treat 403 on label create / PR write-back as a warning (exit 0);
analysis output is still printed. Routing detection unchanged on `main` (#86).
**Follow-up:** if labels must be applied to fork PRs automatically, add a minimal
`pull_request_target` workflow that only calls the labels API.

## 2026-07-10 — Duplicate-detection gate (Gate 3) defeated by re-rolling SVF scales  #mistake #finding #decision

**Context:** auditing the anti-cheat gates in `scripts/pr_eval.py`. Gate 3
(`_check_duplicate`) enforces "original work" by cosine-comparing a submission
against every prior one, rejecting matches above `_COPY_THRESHOLD = 0.999`.
**Expected:** copying another miner's trained routing head should be rejected.
**Actual:** the gate concatenated the head block (6×1024 = 6144 values) and the
SVF block (7168 values) into ONE vector and compared that. The SVF singular-value
scales start at the identity (all 1.0) and move little, so every submission's SVF
block is near-identical to every other's — and it is the larger block, so it
dominates the cosine. Repro (numpy): copy a rival's head **verbatim** (head cosine
1.0) and re-roll only the SVF scales → concatenated cosine ≈ **0.9986 < 0.999**, so
the copied head **passes**. The SVF block both masks copied heads (false negatives)
and, when SVF is tight, can push honest distinct heads over the line (false
positives).
**Root cause:** mixing a near-constant, higher-dimensional block (SVF) with the
small discriminative block (the head) into a single cosine — the meaningful signal
(the head) is a minority of the norm.
**Fix / decision:** compare the HEAD blocks directly (the head is the trained
artifact "original work" refers to — it alone drives routing). SVF cosine is still
computed and reported for context but never masks a copied head. A shape-mismatch
guard skips non-comparable prior heads. Covered by
`tests/test_pr_eval_duplicate.py`: the copy-head/re-roll-SVF evasion is now caught,
exact copies are caught, distinct heads and self pass.
**Follow-up:** none for this bug. (Adjacent, out of scope: warm-started next-gen
heads from the same miner are compared against their own prior gens; if incremental
warm-starts should be allowed, the self-vs-prior-gen policy needs its own decision.)

## 2026-07-10 — The default seed (0) made every CMA-ES run irreproducible  #mistake #gotcha #repro
**Context:** `sep_cmaes.py` opens with "Thin, **deterministic** wrapper around the `cma` library" and documents `seed` as "RNG seed for reproducible sampling". Checking that claim before relying on it for receipt reproduction.
**Expected:** `SepCMAES(n, seed=0)` twice in a row samples the same first population.
**Actual:** it does not. Only *non-zero* seeds are honoured:
```
seed=0: identical first population? False   <-- the default everywhere
seed=1: identical first population? True
```
**Root cause:** pycma special-cases the value. `cma.CMAOptions.defaults()["seed"]` documents itself as *"random number seed for `numpy.random`; `None` and `0` equate to `time`, `np.nan` means 'do nothing'"*. So `opts["seed"] = 0` means **seed from the clock**. And `0` was the default at every level: `SepCMAES(seed=0)`, `run(seed=0)`, `trinity.train --seed default=0`, and the class's own usage example on line 72. What hid it is that the *other* consumers of `args.seed` really are deterministic — `load_tasks(seed=...)` and `gen_rng = random.Random(seed*100000 + gen)` — so a re-run draws the same tasks in the same order and only the CMA-ES trajectory silently diverges. It looks reproducible until the fitness curve differs.
**Fix / decision:** stop forwarding the seed to pycma. Pass `np.nan` (pycma's documented "do nothing") and call `np.random.seed(self.seed)` ourselves, since numpy treats `0` as an ordinary seed. This is behavior-preserving: pycma implements an honoured `seed=k` as exactly `np.random.seed(k)`, verified by a test that reconstructs the reference stream directly from `cma.CMAEvolutionStrategy` — so every previously-working seed keeps its byte-identical stream and archived fitness curves stay reproducible. `0` simply joins them. Rejected the tempting one-liner `seed or 1`, which would silently alias seeds 0 and 1 onto one stream (pinned by `test_zero_and_one_are_not_aliased`). Seeds outside `[0, 2**32-1]` now raise instead of reaching numpy.
**Follow-up:** the wrapper still seeds the **global** `numpy.random` state — that is unchanged from before (pycma did it too), but it means constructing a `SepCMAES` perturbs unrelated numpy randomness in the process. Isolating it behind a `np.random.Generator` / pycma's `randn` option is worth doing separately. Also relevant to the receipt gate in `pr_eval.py`: a "plausible CMA-ES fitness curve" is only re-derivable now that the default seed is honoured.

## 2026-07-10 — MMLU `train` split resolved via shared split_policy  #finding #decision

**Context:** `load_tasks("mmlu", "train")` silently fell back to the 2-item toy set
because `cais/mmlu` publishes `auxiliary_train`, not `train` (issue #35).
**Expected:** training and benchmark builds load real MMLU rows for logical `train`.
**Actual:** `_try_load_hf` swallowed the unknown-split error; `load_split` substituted
the toy set with no warning.
**Root cause:** split name forwarded verbatim; no alias table on the built-in loader path
(MMLU-Pro already fixed this in `split_policy.py` for its adapter).
**Fix / decision:** extend `split_policy._SPLIT_ALIASES` with `mmlu: train →
auxiliary_train`, resolve in `loaders.load_split`, and emit `ToyFallbackWarning` when
the toy set stands in. Keeps one split-resolution module for all benchmarks.
**Follow-up:** GPQA still has only an upstream `train` split; deterministic holdout for
logical `test` is separate work.

## 2026-07-10 — Hidden-benchmark cached answers used a bare prompt, not the WORKER turn  #mistake #finding #decision

**Context:** `scripts/build_benchmark.py::_cache_answers` pre-computes each pool
model's answer per question; those cached answers back the **70%-weighted**
single-turn score in `pr_eval._evaluate_cached`.
**Expected:** cached answers reflect how the pool is actually queried in the
pipeline the router is trained and live-evaluated on.
**Actual:** caching sent a bare `[{"role":"user","content": question_text}]`
message — no role contract. But the single-model baseline
(`trinity.eval._score_single_model`) and the live pipeline
(`orchestration.session.run_trajectory`) both query via
`build_messages(Role.WORKER, prompt, [])`, which prepends the WORKER system
prompt. So the 70% cached component measured model behaviour on a prompt the
pipeline never uses, and diverged from the 15% live component (which does use the
WORKER turn).
**Root cause:** the caching path bypassed `build_messages`, duplicating the
message construction with a different (role-less) shape.
**Fix / decision:** cache via `build_messages(Role.WORKER, question_text, [])`, so
the cached single-turn answers match the baseline and the live path exactly. The
integrity hash excludes `model_answers` (`benchmark_protocol._HASHED_ITEM_FIELDS`),
so this changes only the cached answers on a rebuild, not the sealed question set.
Covered by `tests/test_build_benchmark_cache_prompt.py` (a stub pool asserts the
WORKER-role message layout and the skip-already-cached behaviour).
**Follow-up:** none. Rebuilding a benchmark will refresh cached answers under the
corrected prompt.

## 2026-07-09 — Rate-limit gate parsed UTC timestamps as local time  #mistake #finding #decision

**Context:** Auditing the anti-cheat gates in `scripts/pr_eval.py`. Gate 1
(`_check_rate_limit`) enforces "1 submission per benchmark per week" by comparing
each leaderboard `history` entry's `timestamp` against a 7-day cutoff.
**Expected:** the 7-day window behaves identically wherever `pr_eval.py` runs.
**Actual:** timestamps are *written* in UTC (`time.strftime("...Z", time.gmtime())`
in `_update_leaderboard`) but were *read back* with
`time.mktime(time.strptime(ts, "...Z"))`. `time.mktime` interprets the struct as
**local** time, so on a non-UTC host the parsed epoch is skewed by the host's UTC
offset. Repro on a UTC+9 box: `2026-01-01T00:00:00Z` parsed to an epoch **9 h
earlier** than the true instant. Near the window boundary this lets a miner east
of UTC evade the rate limit (their prior submission reads as older than it is).
**Root cause:** `time.mktime` is the inverse of `time.localtime`, not
`time.gmtime`. The correct UTC inverse is `calendar.timegm`.
**Fix / decision:** added `_parse_utc_timestamp()` (uses `calendar.timegm`, returns
`None` on empty/malformed input) and routed the gate through it, so the window is
timezone-independent. Covered by `tests/test_pr_eval_rate_limit.py` (asserts the
true UTC epoch via `datetime(..., tzinfo=utc)`, and — where `time.tzset` exists —
re-checks under forced non-UTC zones).
**Follow-up:** none for this bug. (Superseded 2026-07-10: rate limit now
counts attempts via the `attempts` log, not only approved wins.)

---

## 2026-06-25 — Constrained decoding fixes parse_rate (1.0), but GRPO has a dead gradient (samples=0)  #repro #finding #decision #gotcha

**Context:** Phase-0 plateaued at parse_rate ~0.047 (format-bound). Added flag-gated constrained
decoding to `HFPolicyBackend` (`--constrained-decoding`): a masked-logits decode samples the per-step
worker index and step count from the policy itself, restricted to legal worker ids and the list
continue/close tokens; `subtasks`/`access_list` are assembled canonically (`_canonical_workflow`), so
every proposal passes the parse-gate by construction. Validated free first, then paid on GPU 3.
**Expected:** parse_rate -> ~1.0 would give GRPO a dense reward signal and let routing accuracy move
toward / past the 0.808 best-single baseline.
**Actual (three runs, all metered to `cost_ledger.jsonl`):**
- Free stub (`--stub-pool --constrained-decoding`): **parse_rate 1.0, $0.00**. Fix confirmed offline.
- Paid g16x5x8 (64-task pool): healthy but **~5 worker calls/min**, projected ~3-4 h. `collect_group`
  runs a group's rollouts sequentially and constrained decoding makes the base policy emit ~3-step
  workflows, so every rollout now dispatches ~3 real worker calls (Phase-0 was fast only because ~95%
  failed parse and skipped the worker). Killed at **$0.50** / 184 calls to relaunch smaller.
- Paid g8x3x4 (32-task pool): iter0 `parse 1.0 acc 1.0 reward 1.0`, iter1 `parse 1.0 acc 0.75
  reward 0.875`. **Every iteration: `samples=0`, `mean_abs_advantage=0.0` -> zero GRPO updates.**
  iter2 dragged on Fireworks retries (~6+ ledger calls/rollout, > the 5-step cap); killed it since a
  third `samples=0` line adds nothing. **$1.59** / 465 calls.
**Root cause (the real finding):** GRPO advantage is computed *within* each question's group of 8
rollouts. On math500 the strong workers solve (or fail) a given question consistently regardless of
which one routing picks, so within-group reward variance is ~0 -> std~0 -> all advantages 0 -> the
update skips every sample. This is the **NEAR_CEILING / 0.29-disagreement oracle result showing up as a
dead training gradient**: variance lives *between* questions, but GRPO only uses *within*-group
variance. acc 1.0 at iter0 was a 4-easy-question artifact, not a real beat of 0.808.
**Fix / decision:** constrained decoding stays (parse_rate 1.0 is a permanent win). Stop spending on
GRPO over math500 until the gradient problem is addressed.
**Follow-up:** (1) a clean accuracy-vs-0.808 number needs a held-out eval over ~120 questions, not
training reward on 4; (2) to get a non-zero gradient: train on the **disagreement subset** (contested
questions), move to a harder benchmark (AIME/GPQA) where routing flips correctness, or use a
**cost-aware reward** so a cheaper-but-correct route beats an expensive-but-correct one even when both
are right; (3) throughput: make `collect_group` run a group's rollouts concurrently (~10x); (4) bias
the policy toward shorter workflows (multi-step inflated cost/latency ~3x for no accuracy gain here);
(5) still open: `SyntaxWarning: invalid escape sequence` from a non-raw regex on the worker/grader path.
Constrained-decoding code committed on branch `fugu-grpo-trainable-backend` (local, not pushed).

---

## 2026-06-25 — Phase-0 paid GRPO: pipeline works, but parse_rate is the bottleneck (not routing)  #repro #finding #decision

**Context:** first *paid* GRPO run of the trainable HF Conductor (Qwen3-0.6B) on `trinity-gpu`
GPU 3. Config g16 × 5 iters × 8 q over a 64-task math500 train pool, 30-step format warmup,
hard cap `--max-cost-usd 25`, warnings at $5/$10/$20, exact spend appended to `cost_ledger.jsonl`.
**Expected:** the format warmup + GRPO would lift parse_rate enough for routing reward to climb.
**Actual:** clean finish (`aborted: false`), **$0.151 total** (704 calls / 640 runs), final parse_rate
**0.047** and final accuracy **0.047**. parse_rate per iter: 0.008 → 0.039 → 0.047 → 0.031 → 0.047 —
fluctuating in a 0.03–0.05 band, no breakout. (Lifetime ledger reads ~$21; that is the cumulative
June-23 eval spend, NOT this run — `cost_report.py` sums the whole file. This run = lines 6994–7053.)
**Root cause:** base 0.6B cannot reliably emit the three-list workflow grammar; a 30-step warmup is far
too weak. With ~95% of rollouts failing the parse-gate, GRPO advantages are computed over degenerate
~0-reward groups, so there is almost no signal to climb.
**Key diagnostic:** at the final iteration **accuracy == parse_rate** — i.e. *when* the policy emits a
valid workflow, the routed worker solves it essentially every time. The bottleneck is **format, not
routing quality.** Routing is already sound; the model just can't produce the schema.
**Fix / decision:** stop trying to fix format probabilistically. Add **constrained/canonical decoding**
to `HFPolicyBackend` (flag-gated): structurally guarantee a schema-valid proposal so parse_rate → ~1.0
by construction and GRPO gets dense reward from iter 0. Validate free (`--stub-pool`, $0) that parse_rate
hits ~1.0 before any further paid spend.
**Follow-up:** (1) prove parse_rate ~1.0 offline; (2) short paid GRPO to see routing accuracy clear the
0.808 best-single baseline; (3) cleanup: `SyntaxWarning: invalid escape sequence` spam from a non-raw
regex string on the worker-output/grader path.

---

## 2026-06-25 — Fugu GRPO GPU-3 free smoke completed; schema and stub-cost gotchas fixed  #repro #finding #mistake #decision

**Context:** ran the new HF Conductor backend on `trinity-gpu` with `CUDA_VISIBLE_DEVICES=3` and
`--stub-pool` to validate model load, sampling, reward grouping, and optimizer update before any Fireworks
spend.
**Expected:** Qwen3-0.6B would emit at least some parseable 3-list workflows and the stub run would report
zero spend.
**Actual:** first smokes loaded the model but had parse_rate 0.0. Raw proposals showed Qwen3 thinking/scratch
leakage and common schema slips (`model_id = 0, 1, 2`, `access_list = []` for one-step workflows, `"none"`
or string indices in access entries). A later smoke hit the GRPO update path but falsely reported `$0.0005`
because stub worker tokens were priced with the real Fireworks table.
**Root cause:** the local HF chat template needed thinking disabled for Qwen3-style models; the parse gate was
too brittle for unambiguous access shorthands; and `scripts/fugu_grpo_train.py --stub-pool` reused real worker
prices even though no API calls happen.
**Fix / decision:** `HFPolicyBackend` now calls chat templates with `enable_thinking=False` when supported and
prefills `model_id = [` to keep generation inside the schema. The parser still requires literal workflow
lists, but now normalizes unambiguous access shorthands (`[]` for one-step no-context, `"none"`, numeric
strings). Stub mode uses an all-zero price table.
**Result:** final free GPU-3 smoke
`summary_gpu3_stub_group32.json` completed with **spend_usd 0.0**, group_size 32, parse/accuracy **3/32
= 0.09375**, and a nonzero GRPO update (`samples=32`, `tokens=1860`, mean_abs_advantage 0.583). No Fireworks
key was sourced and no paid worker calls were made.
**Follow-up:** paid Phase 0 remains gated on explicit spend approval. The trainable backend is now ready for
that run, but base Qwen3-0.6B is still format-weak enough that a short format warmup may be worth testing
before spending on larger GRPO sweeps.

---

## 2026-06-25, Fugu GRPO HF backend built; GPU 3 chosen for the free smoke  #decision #finding #todo

**Context:** next step after the prompted Conductor baseline is the actual GRPO-trained Conductor: test
whether learned workflow/routing can match or beat the strong prompted multi-step scaffold. User reported
GPUs 5 and 6 are in use and asked to check GPU 0 or GPU 3.
**Finding:** `nvidia-smi` on `trinity-gpu` showed GPU 3 essentially idle (5 MiB used), GPU 0 with ~36 GB
allocated, GPU 5 with ~121 GB used, and GPU 6 active. Remote `/mnt/data/harshal/trinity` is not a git
checkout but has the project files and a working `.venv` with torch 2.12.1+cu130, transformers, datasets,
httpx, pyyaml, and accelerate.
**Decision:** use **GPU 3** for this GRPO smoke as an explicit user-approved exception to the standing
`AGENTS.md` GPU-5 rule; do not edit the global GPU-5 default. Record the exception here and pass
`CUDA_VISIBLE_DEVICES=3` only for this run.
**Implementation:** added `src/trinity/fugu/hf_backend.py`, a lazy-import HF `PolicyBackend` that samples
3-list workflows with `generate` and applies no-KL GRPO directly in torch by recomputing token NLL for each
emitted workflow weighted by group-normalized advantages. Added `scripts/fugu_grpo_train.py` with
`--stub-pool` (free CUDA/model/optimizer smoke) and paid Fireworks mode behind `--max-cost-usd`.
**Verification so far:** local CPU Fugu tests passed via `PYTHONPATH=src .venv-lite/bin/python
tests/test_fugu_grpo.py` and `tests/test_fugu_reward.py`; new files compile; remote deps are present.
**Follow-up:** sync the branch files to the remote project directory, run the **free** `--stub-pool` smoke on
GPU 3 first, then only run paid Phase 0 if explicitly continuing with spend.

---

## 2026-06-25, Fugu Conductor prompted-baseline on math500: 0.917 (multi-step lift, not routing); grader dollar FN fixed  #repro #finding #mistake

**Context:** ran the zero-training prompted Conductor (deepseek-v4-pro emits the 3-list workflow; pool
executes; FIXED grader) on the SAME 120 held-out math500 tasks as the oracle matrix. max_depth 0, reps 1,
cost-capped, ledgered. Cost **$1.10** (445 calls).
**Result:** accuracy **0.917**, parse rate 0.975. vs best-single glm 0.808, deepseek single-shot 0.783,
TinyRouter router 0.792, random 0.792, single-pick ceiling 0.855. Paired McNemar vs best-single: b=15, c=3,
**p_exact=0.0075**; router_gap_closed=2.3 (exceeds the single-pick ceiling).
**Read (honest):** the lift is **multi-step test-time compute, NOT routing**. Token share: deepseek 203k
completion toks vs glm 271, kimi 64, so the conductor sent ~all work to deepseek and ran it through a 2-3
step decompose/solve/verify scaffold (+13 pts over deepseek single-shot). It beats the single-pick ceiling
because multi-step solve/verify is a stronger computation than picking one model once. Learned routing is a
separate lever GRPO would add. Cost ~3.6x a single-shot pass (the fanout tax).
**FP/FN audit (the user's explicit ask):** 18-task spot-check printed gold vs grader-extracted vs verdict.
**Zero false positives** (every grader-correct row had extracted==gold, incl. 0.09==9/100, \dfrac{2}{21}==
\frac{2}{21}). **One false negative, FIXED:** math500-459 gold "$18.90" answer "18.90" graded wrong because
`normalize_math_answer` stripped bare "$" before "\$", leaving "\18.90". Fixed in orchestration/reward.py
(strip "\$" first) + regression test. This is a SHARED-grader bug, so it slightly understated the oracle
matrix too; 0.917 is therefore a conservative lower bound (correcting 459 alone -> ~0.925). Grader errs only
in the safe direction (correct marked wrong), never the dangerous one (wrong marked correct).
**Decision / follow-up:** baseline is a genuine, verified result but single-rep; a 3-rep rerun (~$4) would
tighten it. The headline open question for GRPO is whether learned routing + a learned workflow beats this
strong-model multi-step scaffold, and at what cost. Full writeup: docs/fugu/BASELINE_RESULTS.md.

---

## 2026-06-25, OpenFugu replication scaffold: Conductor (Fugu-Ultra) over our pool, offline-tested  #decision #finding #todo

**Context:** new branch `openfugu-replication`. Built the Conductor / Fugu-Ultra tier the repo lacks,
over the existing Fireworks pool (deepseek-v4-pro / glm-5p2 / kimi-k2p6), replicating OpenFugu's design
natively rather than vendoring it (so the grader stays our FIXED one). New package `src/trinity/fugu/`:
`workflow.py` (3-list schema + strict parse-gate + executor with access-list topology and bounded
recursive self-call), `reward.py` (two-stage training reward + PURE-binary `is_correct`), `conductor.py`
(prompted baseline + stub + trained-LM seam), `grpo.py` (group-normalized advantages, no KL, rollout/loop,
cost cap), `eval.py` (pure-binary multi-rep eval, emits per-query 0/1 for the oracle diagnostic),
`cost.py` (per-run pricing, running CostMeter with a spend cap, pre-run estimators).
**Finding (FP/FN discipline):** correctness flows through `orchestration.reward.score_text` only, so the
prose-"A" false positive and LiveCodeBench-reward-0 false negative cannot recur; training reward (parse-gate
+ 0.5 partial) is kept strictly separate from the reported pure-binary metric. 21 offline tests pass with
zero network/GPU/spend. One real bug caught + fixed: `train()` `final_accuracy` KeyError'd when the cost cap
aborted iteration 0 (the trailing abort record has no accuracy key); now reads the last record that has one.
**Cost (user asked to track API spend):** every run carries exact per-model token totals (incl. recursion
and the conductor's own generation); CostMeter aborts at `max_cost_usd`. Projected Fireworks spend
(conductor served locally, ~2.5 steps/workflow): GRPO Phase-0 smoke ~$1.5, Phase-1 small ~$31, paper-scale
(G64 x 200it) ~$615; eval 120x3 reps ~$4.3. The bottleneck is paid worker rollouts, not GPU.
**Decision / follow-up:** the implementation is complete and offline-verified; the only pending piece is the
trainable HF `PolicyBackend` (GRPO on the H200) and the paid run itself, which is GATED on a budget choice
(see docs/fugu/REPLICATION_PLAN.md). Phase 0 (~$1.5) validates the loop before any larger spend.

---

## 2026-06-25, Fugu replication research: the gap is the Conductor, not more router tuning  #finding #decision #todo

**Context:** the user asked for a 2026-only literature sweep on Sakana **Fugu** (released 2026-06-22)
and how to replicate it with open-source models, as the next effort beyond TinyRouter's TRINITY
routing. Ran a 6-angle multi-agent web workflow with per-angle adversarial recency + measured-vs-marketed
auditing (14 agents, ~683K tokens). The synthesis agent stubbed its output (`PLACEHOLDER_MAIN`), so the
dossier was authored by hand from the recovered, verified findings (journal cache in the workflow run dir).
**Finding:** Fugu = TRINITY (which we already replicate) **plus the Conductor** (arXiv:2512.04388, Nielsen
et al., ICLR 2026), productized behind one OpenAI-compatible API. The Conductor is a separate ~7B model
(Qwen2.5-7B) RL-trained with **GRPO (no KL penalty, G=64, two-stage parse+1.0/0.5/0 reward)** that emits a
natural-language workflow (3 lists: model_id / subtasks / access_list, max 5 steps) and can call itself
recursively. So closing the Fugu gap is a **second model to build (RL), not a tune of the CMA-ES head**.
Fugu's *base* tier recipe (SFT on soft per-model performance distributions, then sep-CMA-ES) independently
**validates** the IMPROVEMENTS.md #2 warm-start + #3 shaped-fitness direction (the 2026-06-24 inconclusive
retrain); the likely missing ingredient there was soft performance targets, not more shaping.
**Decision:** wrote the dossier + dedup index to `docs/fugu/` (`FUGU_REPLICATION_RESEARCH.md`,
`REFERENCE_INDEX.md`). All Fugu/Conductor benchmark numbers are first-party with provider-reported
baselines; **no independent third-party reproduction exists as of 2026-06-25**, so they are recorded as
claims, never facts. Closest open prior art is **OpenFugu** (trotsky1997, Apache-2.0: Qwen3-0.6B CMA-ES
router + a GRPO Llama-3.2-3B Conductor with published HF weights); closest open Conductor *method* template
is the non-Sakana **Uno-Orchestra** (arXiv:2605.05007).
**Follow-up:** suggested first experiment (Phase 0 in the dossier §7.4): wrap the existing binary oracle as
a prime-rl/verifiers RLVR environment and train a Qwen3-0.6B "mini-Conductor" to emit parseable 3-list
workflows over the current pool, to prove the loop on one H200 before scaling to Qwen3.5-4B. Main open risk
is rollout economics (GRPO fans out to many paid Fireworks calls). Before locking the recipe, re-read
`arxiv.org/html/2512.04388v5` to pin GRPO hyperparameters, and audit OpenFugu's `train/`.

---

## 2026-06-24 — Warm-start + shaped-fitness retrain on math: inconclusive (within noise)  #finding #decision

**Context:** ran the end-to-end #2+#3 pipeline on the box (`scratchpad/run_warm_shaped.sh`): collect
train-split oracle labels (K=3, n=200) → encode on GPU → numpy fit of the agent head → pack as CMA-ES x0
→ retrain (popsize 8, m_cma 8, gen 12, seed 0) with shaped training fitness → held-out eval on the n=120
test split (same set as the 0.856 oracle ceiling).
**Expected:** if warm-start + shaping helped, the new router beats the prior 0.792 and closes some of the
+4.9pt headroom toward 0.856.
**Actual:** TRINITY 0.808 vs prior 0.792 (+1.6pt). Best single (glm) 0.817, random 0.733. Training best
shaped-fitness 1.0145 at gen 11 (full 12 gens). Spend $27.22 total.
**Root cause / read:** the +1.6pt is inside eval noise. Random routing scored 0.733 here vs 0.792 in the
§1 rigorous eval — same baseline, ~6pt swing — because eval uses `--single-reps 1` (one sample/query).
A moving baseline of that size swamps a 1.6pt router delta.
**Fix / decision:** documented as RESULTS §9 with **no causal claim**. The clean control (zero-init +
pure-binary at the same config) was offered and the user chose to skip it (~$11 to settle borderline
noise), so we do not attribute the change to warm-start or shaping. Result still below best-single (0.817)
and the oracle ceiling (0.856); the achievable headroom from §8 remains uncaptured.
**Follow-up:** if revisited, run the control + raise eval `--single-reps` to ≥3 to shrink the baseline
noise band before claiming any lift. Interventions remain implemented + tested (54 offline tests pass).

---

## 2026-06-23 — Oracle-ceiling diagnostic: math is ROUTER_BOUND (overturns the "math null" read)  #finding #repro #mistake

**Context:** built `scripts/oracle_ceiling.py` (recommendation #1, branch `oracle-ceiling-diagnostic`) to
answer whether routing can help at all on our 3-model pool, FP/FN-proof per the plan. Collected a
per-(query,model) solve matrix (math K=5, MMLU K=3) on the box, $14 (separate `oracle_cost_ledger.jsonl`).

**Result:**
- **math500: ROUTER_BOUND.** Perfect router could reach **0.856** vs best-single 0.808 = **+0.049
  headroom, 95% CI [0.005, 0.085]** (excludes 0). Naive oracle 0.900; cross-fit stripped 0.044 of
  winner's-curse inflation. Models disagree on 29% of queries.
- **mmlu: INCONCLUSIVE at K=3.** Threshold headroom +0.025, CI [0, 0.058]; deepseek dominates (0.94).
  Near-ceiling in practice (TRINITY already ≈ best-single).

**Finding that matters (#finding):** this **overturns** the earlier "math routing gives no benefit"
read. That conclusion came from the trained router tying random/best-single — but the oracle shows there
IS ~+4.9pt of *achievable* headroom on math; our router just captures none of it. So math is limited by
the **router, not the pool** → the warm-start (#2) + shaped-fitness (#3) upgrades are justified, with a
concrete target (math oracle 0.856). The diagnostic earned its keep by catching this false-negative.

**#mistake (caught by the diagnostic's own guard):** MMLU at K=3 first produced an *impossible* negative
headroom (routing_oracle 0.750 < best_single 0.939). Cause: cross-fit splits K in half, so K=3 leaves 1
selection sample/query and the argmax misroutes. Fix: floor the oracle at best_single (a perfect router
can always fall back to the best fixed model), flag `crossfit_reliable=false` when K<5, and base the
verdict on the split-free threshold headroom. Added selftest (e). Cross-fit needs **K>=5** (n_a>=2).

**#gotcha:** the K=6 MMLU re-collect (for a valid cross-fit) was abandoned after Fireworks latency
spiked (throughput fell ~38→4 calls/min, ~5h ETA, no errors). MMLU verdict stands on the threshold
estimate + rigorous-eval evidence; re-collect at K>=5 when the API is healthy for the exact number.

---

## 2026-06-23 — RIGOROUS final eval (n=120, baselines ×3 reps): math null, MMLU win, thin multi-task win  #repro #finding

**Context:** the n=40 per-coordinator numbers were too noisy (same baseline swung 0.45–0.79 across runs).
Ran the definitive eval: n=120 held-out items, each single-model baseline averaged over 3 reps to kill
reasoning-model nondeterminism. Raw: `experiments/final/{math_rigorous,mmlu_rigorous}.json`.

**Result:**

| system | math500 | MMLU | **avg** |
|---|---|---|---|
| **TRINITY** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro | 0.747 ± 0.014 | 0.922 ± 0.010 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 | **0.794** ± 0.017 | 0.783 ± 0.007 | 0.789 |
| kimi-k2p6 | 0.742 ± 0.018 | 0.539 ± 0.004 | 0.640 |

**Findings (honest):**
- **Multi-task avg: R1/R2 ✅ and R4 ✅, but THIN.** TRINITY 0.858 > best fixed single 0.835 (deepseek)
  > random 0.833. Margins ~0.02.
- **math500: routing gives NO benefit.** TRINITY 0.792 = random 0.792 *exactly*, and ties best single
  (glm 0.794, inside noise). R1/R2 ❌, R4 ❌ for math as a standalone task. Root cause: all three models
  cluster at ~0.74–0.79, so there is no complementarity to exploit — any routing (incl. random) lands at
  ~0.79. This is the SAME thin-complementarity pattern an independent sibling project (project_harness)
  measured on math/proofs (see entry below).
- **MMLU: routing helps.** TRINITY 0.925 > random 0.875 (R4 ✅), edges best single (deepseek 0.922).
  Models are spread (0.54–0.92) → real headroom, captured.
- **The win is CROSS-task, not within-task.** No single model is good at both (deepseek=knowledge,
  glm=math); TRINITY picks the right specialist per benchmark, so its *average* beats any fixed model.
  Within a benchmark of similar models it only matches random.

**Correction to earlier claims (#mistake):** the n=40 math story (TRINITY 0.55 > glm 0.50) that read as a
routing win was small-sample noise; at n=120 it is a tie. RESULTS.md §3 marks the n=40 table superseded.

**Cost (final):** $20.89 exact (deepseek $6.56, glm $6.70, kimi $7.64), well under the ~$65 projection.

**Follow-up:** routing value lives where models genuinely differ. Next levers documented in the entry
below (project_harness) and the classifier-improvement research note.

---

## 2026-06-23 — Assessed sibling repo `project_harness` for reuse (12-agent workflow, every claim verified)  #finding #decision

**Context:** user pointed at `~/Desktop/2026/experiments/project_harness` (a sibling multi-LLM
orchestration project on IMO-ProofBench + SWE-bench Pro) and asked, honestly, whether its data is usable
for TRINITY. Ran a survey→assess→adversarial-verify workflow; all key claims re-checked against files.

**Findings:**
- **Direct data reuse: NO.** Two independent blockers (verified): (1) **model mismatch** — locally-graded
  generator labels cover kimi-k2p6 + glm-5**p1** + gpt-oss-120b; deepseek-v4-pro has zero generator
  grades (it sits on the grading jury), and glm is 5p1 not our 5p2; (2) **domain mismatch** — everything
  is IMO proofs (0–7 LLM-jury) or SWE-bench coding (binary Docker), with **no math500 and no MMLU
  anywhere**.
- **Independent corroboration (valuable):** project_harness measured the SAME thin complementarity on
  math/proofs that we just saw — oracle-union 8/30 vs best-single 7/30 (+1, inside jury MAE), and "no
  live selector beat best-single." Matches our math null result exactly.
- **Real complementarity exists on CODING:** their SWE-bench arena (kimi/deepseek/glm) shows oracle 0.600
  vs best-single 0.480 (+12pt); 6/15 solvable instances solved by exactly one model. Coding is where
  routing would actually pay off.
- **Groundwork is on OUR box:** `/mnt/data/harshal/evo_study/` (same machine) has a frozen 25-instance
  3-model candidate pool (incl. glm-5p2 capture) + gold pass/fail labels + Docker + the SWE-bench eval
  harness. The expensive part (generate + grade candidates) is already done.

**Decision:** a separate coding/SWE-bench TRINITY run was scoped (offline "selection arena": train the
coordinator to route over the frozen, gold-labelled pool for ~$0). **User chose to abandon it** for now
and keep the math/MMLU result clean. Reusable take-aways imported as a prior: routing helps only where
models genuinely differ; report 1-instance margins as noise; value may live in the Verifier role.

**Follow-up:** if we revisit, the offline arena is the cheapest high-value experiment available.

---

## 2026-06-23 — MMLU eval: models SPLIT across tasks → routing headroom confirmed  #repro #finding #decision

**Eval (40 held-out MMLU items; math-trained θ used → TRINITY here is zero-shot transfer):**

| condition | math500 | MMLU |
|---|---|---|
| deepseek-v4-pro | 0.325 | **0.975** ← best on MMLU |
| glm-5p2 | **0.550** ← best on math | 0.725 |
| kimi-k2p6 | 0.275 | 0.600 |
| random routing | 0.350 | 0.675 |
| TRINITY | 0.550 (trained) | 0.850 (zero-shot transfer) |

**Headline finding (#finding):** the pool **splits** — glm-5p2 is the math specialist, deepseek-v4-pro
the knowledge specialist. **No single model wins both.** This is precisely the regime where the paper's
claim lives: a per-task router that picks the right specialist beats any *fixed* single model on the
average. Quick arithmetic with best-per-task routing:
- best fixed single model avg: deepseek (0.325+0.975)/2 = **0.65**; glm (0.55+0.725)/2 = 0.6375.
- per-task-trained TRINITY ≈ per-task best (math→0.55, mmlu→~0.97) → avg ≈ **0.76** > 0.65. R1/R2 holds
  **on the multi-task average** (to be made concrete by training an MMLU coordinator).
- Zero-shot transfer bonus: math-trained θ on MMLU still beat random (0.85 vs 0.675) ✅ R4 again, though
  below deepseek (it was trained to favor glm, the wrong pick for MMLU) — sensible.

**#mistake (minor):** `eval.py --out` crashed writing `experiments/mmlu/...` because the parent dir
didn't exist (results were already printed, so no data lost). Fixed with `parent.mkdir(parents=True)`.

**Decision:** train an MMLU-specific coordinator (paper protocol = one coordinator per task) so we have
a real per-task TRINITY on MMLU, then report the concrete multi-task average vs best single model.

---

## 2026-06-23 — Held-out eval: R4 holds, R1/R2 ties on single task → need multi-task  #repro #finding #decision

**Eval (40 held-out math500 items, trained `full_pilot` θ):**

| condition | acc |
|---|---|
| single deepseek-v4-pro | 0.325 |
| single **glm-5p2** | **0.550** (best single) |
| single kimi-k2p6 | 0.275 |
| **TRINITY (trained)** | **0.550** |
| random routing | 0.350 |

- **R4 ✅** TRINITY (0.55) > random routing (0.35). The learned coordinator is meaningfully better
  than random — it discovered that **glm-5p2 is the math specialist** and routes to it.
- **R1/R2 ❌ (tie, not a win):** TRINITY (0.55) = best single model glm-5p2 (0.55).

**Why this is the EXPECTED outcome, not a failure (#finding):** on a SINGLE benchmark, pure routing
can at best *match* the best model (by always picking it) — there's no headroom to *exceed* it unless
multi-turn Thinker→Worker→Verifier collaboration adds value, which 3 turns / 640 tokens barely
exercises. The paper's "TRINITY > best single model" headline is on the **average across 4 diverse
tasks**, where *no single model wins them all* — that's where per-task routing beats any fixed model.
We replicated the routing mechanism on one task (it found the specialist + beat random); to replicate
the *headline* we need the multi-task setting. Eval is also noisy at n=40 (±~8%), so 0.55 vs 0.55 is a
statistical tie.

**Decision / next:** map single-model strengths across ≥2 benchmarks (math500 + a second where a
*different* model wins, e.g. gpqa/livecodebench). If models split, a per-task router beats best-single
on the average — the real R1/R2 test. Train a coordinator per task (paper protocol) and report the
multi-task average.

---

## 2026-06-23 — Pilot #3 (full_pilot): 12 generations complete, no crashes  #repro #finding

**Result:** First end-to-end training run to **complete all 12 generations** (math500, λ=8, m_cma=8,
3 turns, 64 train tasks) — the timeout-retry + degrade-to-0 fixes held the entire run, zero crashes.

```
gen0 0.094  gen1 0.266  gen2 0.469  gen3 0.500  gen4 0.422  gen5 0.406
gen6 0.297  gen7 0.391  gen8 0.359  gen9 0.469  gen10 0.469  gen11 0.250    (per-gen mean fitness)
```

**Read:** strong early learning (mean 0.094→0.50 over gens 0–3), then the population mean plateaus
and oscillates ~0.25–0.47. `best_fitness` (es.best) = 0.75. `best_theta.npy` saved.

**Caveat (honest):** the cross-generation mean is confounded — common random numbers re-samples a
*different* minibatch each generation, so a harder draw lowers that gen's mean independent of policy
quality (e.g. gen11's 0.250 is likely a hard draw, not regression). So this curve shows "it learned"
but is NOT a clean convergence curve. The decisive measurement is the held-out eval (running now):
TRINITY vs each single model vs random routing on fresh math500 test items (R1/R2/R4).

**Possible next improvements (if eval is inconclusive):** (a) a fixed validation minibatch to make
gen means comparable, (b) larger m_cma to cut reward variance, (c) more generations / σ tuning.

---

## 2026-06-22 — Pilot #2 (CRN): J climbs, then crashes on httpx.ReadTimeout  #repro #mistake #finding

**Result (the headline so far):** with common random numbers, **sep-CMA-ES learns** — clean upward J:

```
gen0 mean=0.141 best=0.375    gen1 mean=0.203 best=0.375    gen2 mean=0.438 best=0.625
```

Mean fitness tripled and best improved 0.375→0.625 in 3 generations on math500 — the core
replication claim (the trained coordinator improves over evolution) is demonstrated in principle.
`best_theta.npy` + `history.json` saved each generation, so gen-2's θ (fitness 0.625) survived.

**#mistake — run crashed at gen 3 on `httpx.ReadTimeout`.** My Fireworks client retried only on HTTP
status codes (429/5xx), not on network timeouts / transport errors, so one slow reasoning call (>120s)
raised an uncaught `ReadTimeout` that propagated through `asyncio.gather` and killed the whole run.
(The `python | tee` pipeline masked it as exit 0 — `tee` succeeded even though python died.)

**Fixes (#decision):**
1. `FireworksPool.chat` now catches `httpx.TimeoutException`/`TransportError` and retries them
   (transient blips are normal over thousands of calls). Timeout 120→180s, retries 4→6.
2. `evaluate_candidate` uses `gather(return_exceptions=True)` — a trajectory that exhausts retries
   degrades to **reward 0** and logs a warning, instead of crashing the generation/run.
3. Next run uses `nohup` (survives an ssh drop). **Detach gotcha (root-caused):** `nohup cmd > log &`
   inside a long `a && b && read KEY && nohup ... &` chain silently failed because `&` binds the
   *whole* `&&` list — so the chain (including `read KEY`) ran in a backgrounded subshell whose stdin
   is `/dev/null`, the `read` hit EOF, the chain aborted before launching python, and `$!` was just
   the dead subshell. Fix: keep the key-`read` in the foreground and background only the launch with a
   brace group: `... && export KEY && { nohup python ... > log 2>&1 </dev/null & }`. Verified the log
   is created and training proceeds.

**Lesson:** a long API-bound training loop must treat transient network errors as expected, not fatal.
CPU smoke ladder still 4/4 green after the fixes.

---

## 2026-06-22 — Pilot #1: flat J → root cause = per-candidate minibatch noise → CRN fix  #mistake #finding #decision

**Context:** First real pilot (math500, λ=6, m_cma=4, 12 gens). Ran clean across 5 generations on
GPU 5 (no crashes, per-candidate fitness varied 0.0–1.0), but **mean fitness did not climb**:

```
gen0 mean=0.375  gen1 0.542  gen2 0.375  gen3 0.375  gen4 0.250   (max stuck at 1.0 = lucky minibatch)
```

**Root cause (#mistake in my training loop):** `minibatch_fn(i)` drew a *fresh random* minibatch for
*each candidate* within a generation. With `m_cma=4` the per-candidate fitness is a mean of 4
Bernoulli draws (std ≈ 0.25) AND each candidate saw *different tasks* — so sep-CMA-ES was ranking
candidates largely by task-luck, not policy quality. The "best=1.0" was one candidate that drew 4
easy problems, not a good coordinator. This is precisely the binary-reward variance risk flagged in
SPEC §0.4 and the review (#5).

**Fix / decision (#decision):** **Common Random Numbers (CRN)** — score *all* candidates in a
generation on the *same* minibatch (re-sampled across generations). Standard ES variance-reduction:
fitness *differences* now reflect policy differences, not which tasks were drawn. Also raised
`m_cma 4 → 8` to halve the reward-estimate std. Re-launched as `pilot_crn`. This is a deliberate
[OUR CHOICE] deviation from the paper's "re-sample per replication" phrasing; documented because it
materially changes the optimizer's signal.

**Lesson:** for noisy-reward ES, *how* you sample the fitness minibatch matters as much as the
optimizer. Watch whether `pilot_crn` shows a cleaner upward J trend.

---

## 2026-06-22 — GPU env up; full smoke ladder S1-S8 PASS; pilot training launched  #repro #finding #gotcha

**Context:** Provisioned the H200 box and ran the GPU/network smoke rungs, then launched training.

**Env:** `uv` venv on the box, **torch 2.12.1+cu130**, transformers 5.12.1, numpy 2.5.0, cma. With
`CUDA_VISIBLE_DEVICES=5`, `torch.cuda.device_count()==1` (correctly pinned to our H200).

**Smoke ladder (GPU/net rungs) — ALL PASS on GPU 5:**
- **S1**: Qwen3-0.6B loads, `hidden_size=1024`, `layers=28`, encode deterministic, `‖h‖=1.0000`.
- **S2**: SVF `num_scales=7168` — **exactly the predicted 7×1024, confirmed on the real
  checkpoint** (resolves the paper's 9,216 discrepancy for our model). Identity round-trips
  (`max|Δ|=1.2e-3` bf16), perturb changes weights, `reset()` exact.
- **S6**: all 3 Fireworks models answer live with `reasoning_effort=minimal`.
- **S8**: end-to-end fitness produced within the call budget.

**#gotcha — detached launch quoting.** First `nohup`/`setsid` launches failed silently (log file
never created) due to nested single/double-quote + `\$HOME` escaping across the ssh boundary, plus
SIGHUP timing. **Fix:** run training via a local-background `ssh ... | tee` (key fed on stdin →
remote env, never in argv/disk); foreground sanity run first confirmed the loop works
(`gen0 best=0.500`, n=13312, 85s for pop3×m2).

**#finding — uniform-policy argmax degenerates to THINKER.** With `W=0`, argmax over uniform role
logits always picks role index 0 = THINKER (ROLE_ORDER[0]), so a never-trained coordinator under
argmax produces no Worker answer → reward 0. Training uses `sample=True` so it explores; this is
expected, not a bug. Eval uses argmax (post-training, when W is non-trivial).

**Pilot config (running):** math500, λ=8, m_cma=6, T=8, max_items=64, max_turns=3, max_tokens=1024
→ budget ≈ 384 atomic evals. Watching whether `J` rises before committing to a full-budget run.

---

## 2026-06-22 — Core implemented; CPU smoke rungs green; review bugs fixed  #repro #mistake #finding

**Context:** Implemented all modules (coordinator, roles, orchestration, sep-CMA-ES, train/eval)
via a parallel build + adversarial integration review, then validated the CPU smoke ladder.

**Integration review caught 3 real bugs (fixed before any GPU/LLM spend):**
- **#mistake P0 — LiveCodeBench reward was identically 0.** `reward._run_one_test` read stdin from
  `test["stdin"]` but `dataset.py` emits `{"input":..., "output":...}`, so every code test ran on
  empty stdin → reward 0 for every candidate → CMA would optimize against a dead signal. Fixed to
  read `test.get("stdin", test.get("input",""))` and trigger on `input`/`output` keys.
- **#mistake P1 — `trinity.optim` couldn't import without pycma.** `sep_cmaes.py` re-raised the
  `cma` ImportError at module top. Deferred it into `_import_cma()` (cma is only needed to build
  the optimizer at train time).
- **#mistake P2 — choice-letter extractor matched prose.** `"A nice approach"` → `"A"`. Tightened
  the regexes + restricted the fallback to a final standalone-letter line. Also fixed plain-text
  fraction extraction (`"1/2"` was read as `"2"`).

**Smoke ladder (SPEC §11), CPU rungs run locally — ALL PASS:**
- S3 params pack/unpack round-trip, `n_total=13312`, head `(6,1024)`, `n_svf=7168`.
- S4 multi-turn termination + worker-guarded Verifier-ACCEPT + fail-safe REVISE.
- S5 reward checkers (math incl. fractions, MMLU/GPQA letters, code pass@1 with stdin).
- S7 sep-CMA-ES maximizes a synthetic objective; `popsize(13312)=33` confirmed.

**Follow-up:** provision GPU box env, run GPU rungs S1 (encoder/penultimate), S2 (SVF identity +
real scale count), S6 (live pool), S8 (end-to-end fitness), then launch sep-CMA-ES training on GPU 5.

---

## 2026-06-22 — Fireworks reasoning-effort mapping resolved  #finding #decision

**Context:** SPEC left "minimal reasoning effort" → API param unspecified (open item #12/#16).
**Finding:** all 3 models accept `reasoning_effort` ∈ {none, low, medium, high} (HTTP 200).
**Decision:** map "minimal" → `reasoning_effort: "low"` in `FireworksPool.chat`; configurable.

---

## 2026-06-22 — Paper → SPEC, with verified facts & review corrections  #finding #decision #repro

**Context:** Ran a 9-agent deep read of the paper → `docs/SPEC.md` (+ `PAPER_NOTES.md`,
`SPEC_REVIEW.md`). Then grounded the risky numbers against ground truth.

**Findings / corrections (SPEC §0 is authoritative):**
- **Qwen3-0.6B real config:** `hidden_size=1024` (d_h CONFIRMED), **28 layers** (2nd-to-last =
  index 26), GQA (16 q / 8 kv heads, head_dim 128 → `q_proj` is 1024×2048, `o_proj` 2048×1024),
  SwiGLU `intermediate_size=3072`, tied embeddings, bf16. Every linear matrix has min-dim 1024 →
  **1024 SVs each**.
- **SVF count mismatch (#mistake-averted):** paper states 9,216 SVF scales (=9×1024), but a Qwen3
  layer has only **7** linear matrices → 7×1024 = **7,168**. The paper's 9,216 does not map onto
  Qwen3's matrix set. **Decision:** SVF all 7 matrices of layer 26 → 7,168 scales (init 1.0),
  documented delta. Smoke test **S2 must print the real count** and assert θ matches.
- **CMA λ arithmetic error caught by review:** spec body said λ=34; correct is
  `⌈4+3·ln(13312)⌉ = ⌈32.49⌉ = 33`. Budget `B_env = 16·33·60 = 31,680` (body's "34,560" was wrong).
- **Totals (ours):** head `6×1024 = 6,144` + SVF `7,168` = **n = 13,312** trainable (CMA dim).
- **Decisions:** L2-normalize `h` before the head (σ₀ stability); Verifier can't ACCEPT before a
  Worker output exists; MT-Bench is report-only (never binarized into reward); single-model
  baselines run at 20,480 tokens (5×) for fair R1/R2; disk-cache LLM calls.
- **Open risk:** block-ε-separability was shown on the paper's 7-agent pool; it may NOT transfer
  to our 3-model pool, so **R8 (CMA > SFT > RS > REINFORCE) is a hypothesis to test**, not assumed.

**Follow-up:** implement M0–M4, pass the S1–S8 smoke ladder, then run head + sep-CMA-ES on GPU 5.

---

## 2026-06-22 — Remote H200 box inventory  #finding

**Context:** Inventoried `trinity-gpu` (read-only) before writing the remote setup path.

**Findings:**
- **Ubuntu 24.04**, **192 vCPU**, **~2 TB RAM**. `$HOME` = `/mnt/data/harshal` on a 12 TB array
  with **3.2 TB free** — ample for HF model caches.
- **8× NVIDIA H200 NVL (143 GB each)**, driver `595.71.05`. **We use index 5 only.**
- Python **3.12.3** present; **no `uv`, no `conda`, no `torch`** system-wide → `setup_remote.sh`
  installs `uv` and builds a project `.venv`.
- Network: `huggingface.co` → 200 (model downloads OK). `api.fireworks.ai` root → 404, which is
  expected (the API lives under `/inference/v1`; root has no handler). Not a problem.
- No pre-existing `~/trinity`; we sync there fresh.

**Decision:** default `TRINITY_REMOTE_DIR=$HOME/trinity` (= `/mnt/data/harshal/trinity`),
`HF_HOME` under the project dir to avoid polluting shared `$HOME`.

---

## 2026-06-22 — Project bootstrap & environment verification  #finding #decision

**Context:** Kicking off the replication. Verified the full toolchain before writing code.

**Findings:**
- **GPU box reachable.** `ssh trinity-gpu` works. `nvidia-smi -i 5` reports **NVIDIA H200 NVL,
  143771 MiB, idle**. We are allocated **GPU 5 only** — all CUDA work pins
  `CUDA_VISIBLE_DEVICES=5`.
- **All three Fireworks models answer** a chat completion (HTTP 200):
  `accounts/fireworks/models/deepseek-v4-pro`, `.../glm-5p2`, `.../kimi-k2p6`.
- **GitHub:** authed as `harrrshall` with `repo` scope; repo will be created **private**.

**Decisions:**
- Secrets live **outside the repo**: SSH key at `~/.ssh/trinity_gpu` (600) behind the
  `trinity-gpu` host alias; Fireworks key at `~/.config/trinity/secrets.env` (600), read via
  `FIREWORKS_API_KEY`. `.gitignore` blocks all secret patterns + the 11 MB paper PDF.
- Replication target is the paper's **relative** claims (TRINITY > best single model > random
  routing; sep-CMA-ES > RL/IL/random), since our open-source model pool differs from the
  paper's. See `AGENTS.md` §1.

## 2026-06-22 — Fireworks account model-list endpoint 500s  #gotcha

**Context:** Tried `GET /inference/v1/models` to enumerate available models.
**Actual:** `HTTP 500 — "Error listing deployed models"`.
**Root cause:** That account-level listing endpoint is flaky / not enabled for this key; it is
not needed.
**Fix / decision:** Probe model IDs directly with a tiny `chat/completions` call instead. All
three target IDs returned 200, so we hardcode them in `configs/models.yaml`. Do **not** depend
on the list endpoint.

---

## 2026-06-23 — Parallel multi-seed training across free GPUs  #decision #repro

**Context:** User cleared use of all free GPUs (1,4,6,7); GPU 5 keeps its run. Launched stronger
coordinators with seed replicates for an error-barred multi-task result.

| GPU | run | task | seed | config |
|---|---|---|---|---|
| 1 | math_s0 | math500 | 0 | λ8, m_cma10, T14, 4 turns, 768 tok, 80 items |
| 4 | math_s1 | math500 | 1 | same |
| 5 | mmlu_pilot | mmlu | — | basic (earlier run, still finishing) |
| 6 | mmlu_s0 | mmlu | 0 | strong |
| 7 | mmlu_s1 | mmlu | 1 | strong |

**Caveat (#finding):** workload is **API-bound, not GPU-bound** (GPUs idle ~0% util waiting on
Fireworks). 5 parallel processes share the Fireworks rate limit → ~40 concurrent calls; 429s are
retried (robustness fix), so throughput is shared, not 5×. Real benchmarks limited to math500 + mmlu
(gpqa gated, livecodebench loader → toy fallback; both need extra work to enable real data).

**Plan:** when all finish, eval each coordinator on its held-out test set, then report
TRINITY-per-task (mean±std over seeds) vs best single fixed model on the math+mmlu average = the R1/R2
headline test.

---

## 2026-06-23 — Cost tracking added  #decision #finding

**Context:** User asked to track Fireworks cost. No usable billing API (probed: `/v1/usage`→404,
`/v1/accounts/usage`→403), but every chat response carries exact token counts, so we price from tokens.

**Built:**
- `scripts/cost_report.py` — `--ledger` (exact, from recorded tokens) or `--estimate` (from run configs).
- `FireworksPool` now appends `{model, prompt_tok, completion_tok}` to `$TRINITY_COST_LEDGER` per call
  (best-effort JSONL) → future runs/evals are tracked exactly. The 5 in-flight runs predate this, so
  they're estimated.

**Empirical fact:** reasoning models fill ~all of `max_tokens` on completion (glm used 400/400),
so completion_tokens ≈ max_tokens; prompt grows with the multi-turn transcript (~650 avg).

**Estimate (ASSUMED prices, blended ~$0.67/1M in, $2.10/1M out — NOT confirmed):**
- Spent so far (pilots + 2 evals): **≈ $5**.
- Projected when the 5 current runs finish: **≈ $34 total** (~24M tokens, ~17k calls).
- The 4 strong parallel runs dominate (~$6.4 each).

**TODO:** get real per-model Fireworks rates from the dashboard to convert estimate → exact. Report a
live cost line at each monitoring checkpoint (scale per-run cost by generations completed).

---

## 2026-06-23 — Real Fireworks prices → cost ~doubled  #finding

Web search gave real serverless rates ($/1M, in/out): deepseek-v4-pro **1.74 / 3.48**,
kimi-k2p6 **0.95 / 4.00**, glm-5p2 **1.40 / 4.40** (GLM-5.1 proxy; 5.2 not separately listed).
Output is ~$4/M, so the projection rose from ~$34 (assumed) to **~$65 total** (~$10 spent so far;
the 4 strong parallel runs ~$12 each). All 5 runs confirmed RUNNING. Caching (~50% off cached input)
ignored, so this slightly over-estimates.

---

## 2026-06-23 — MMLU underperformance root-caused: brittle extraction + bad routing  #mistake #finding #decision

**Diagnostic** (dumped argmax trajectories of the mmlu_pilot coordinator) revealed two compounding bugs:
1. **Extraction loses correct answers.** A worker derives "degree 2" (correct, =B) but emits no clean
   letter; `_final_answer` returns that verbose worker output → `extract_choice_letter`→None → reward 0,
   even though the system found the answer. This penalizes multi-turn answers; the single-model baseline
   (one clean direct call) extracts fine → unfair 0.55-vs-0.95 gap.
2. **Coordinator routes to kimi/glm, NOT deepseek** (the MMLU champion). It never learned deepseek is best
   for MMLU — because bug #1 made the *training* reward noisy (correct answers scored 0), corrupting the
   signal the optimizer needed to distinguish deepseek > kimi.

**Fix #1 (done):** `score()` now scores the MOST RECENT turn with an *extractable* answer (falling back to
final_answer), applied equally to all conditions. Re-evaluating to quantify how much of the gap was
extraction vs routing.

**Implication:** the extraction bug hurt BOTH eval scoring AND training quality. Cleanly fixing it likely
needs a re-train (so the optimizer sees an honest reward and learns to route to the right specialist).
Caveat for choice tasks: models often answer with the VALUE ("degree 2") not the LETTER ("B"); fully
fixing that needs answer-format prompting (task-aware), noted as a follow-up.

---

## 2026-06-23 — Extraction fix → MMLU TRINITY 0.55→0.95 (ties best single); multi-task R1/R2 holds  #repro #finding

**mmlu_s1 re-eval with fixed extraction (40 held-out items):**
deepseek 0.975 | glm 0.750 | kimi 0.600 | random 0.850 | **TRINITY 0.950**.
TRINITY jumped 0.55→**0.95**, now a statistical tie with the best single model (deepseek 0.975) — the
0.40 gap was almost entirely the extraction bug, confirming the diagnosis. R4 holds (0.95>0.85).

**Multi-task headline (the paper's core claim) — reproduced on our open-source pool:**
- best FIXED single model avg ≈ **0.65** (deepseek 0.33/0.975; glm 0.55/0.75).
- per-task TRINITY avg ≈ **0.75** (math 0.55, mmlu 0.95) → **TRINITY > any single fixed model on the
  multi-task average.** No single model wins both tasks; routing to the per-task specialist does.
- Per-task, TRINITY ties the best specialist (math≈glm, mmlu≈deepseek) and beats random on both — the
  expected single-task ceiling for routing.

**Status:** strong math coordinators at 13/14; re-evaluating math with the fixed extraction to finalize
the table. Exact ledger cost so far ~$1.3 (evals); total spend ~$13.

---

## 2026-06-23 — Structured results + rigorous eval launched  #decision #repro

Per user request (document everything + structured output):
- `scripts/results_table.py` aggregates all `experiments/**/eval*.json` → structured Markdown table +
  `experiments/results.json` (machine-readable). `docs/RESULTS.md` is the human report (linked from README).
- Current aggregated verdict (40-item evals): **R1/R2 ✅ 0.750>0.639, R4 ✅ 0.750>0.558**, with caveats:
  math seed variance (math_s0 failed at 0.325), n=40 eval noise (single baselines swing 0.45–0.70).
- **Rigorous eval running** (GPU5): n=120, single baselines ×3 reps (kills reasoning-model
  nondeterminism), best math (full_pilot) + best MMLU (mmlu_s1) coordinators → definitive numbers.
- Cost ~$22 (ledger-tracked). No GPU was empty (other tenants), but evals are light (~4 GB) so they
  coexist on a shared H200.
