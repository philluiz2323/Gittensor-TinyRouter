# CI Policy

TinyRouter uses GitHub Actions to enforce repository policy without GPU,
OpenRouter, or hidden-benchmark access.

## Fast PR lane

Workflow: `.github/workflows/ci.yml`

Runs on every pull request and on pushes to `main`:

1. `pip install -e ".[dev]"`
2. `ruff check src/ scripts/repo_governance/`
3. `mypy src/ scripts/repo_governance/`
4. `pytest tests/ -q`

This lane is offline and deterministic. It is the required check for normal
contributor PRs.

## Repository bots

Workflows:

- `.github/workflows/issue-bot.yml`
- `.github/workflows/pr-bot.yml`

Scripts live in `scripts/repo_governance/`:

- `issue_bot.py` — auto-labels issues and comments when required details are missing
- `pr_bot.py` — auto-labels PRs by changed paths, checks template completion, and flags sensitive scoring/protocol files
- `ensure_labels.py` — creates governance labels if they do not exist yet

Sensitive paths include:

- `scripts/pr_eval.py`
- `scripts/build_benchmark.py`
- `scripts/benchmark_protocol.py`
- `leaderboard.json`
- `configs/`
- `docs/BENCHMARK_PROTOCOL.md`
- `submissions/`

## Manual evaluation lane

Workflow: `.github/workflows/eval-manual.yml`

This is a `workflow_dispatch` lane for offline maintainer commands such as:

```bash
python scripts/audit_eval.py --help
```

It must not be wired into the default PR lane. Paid API calls, GPU work, and
hidden-benchmark decryption stay manual.

## Local reproduction

```bash
pip install -e ".[dev]"
ruff check src/ scripts/repo_governance/
mypy src/ scripts/repo_governance/
pytest tests/ -q
```

Dry-run the bots locally:

```bash
cd scripts/repo_governance
python issue_bot.py --title "[bug] example" --body "..." --json
python pr_bot.py --title "fix: example" --body "..." --files '["README.md"]' --json
```
