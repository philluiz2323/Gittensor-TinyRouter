# Contributing to TinyRouter

TinyRouter is a **routing accuracy competition** on the
[Gittensor](https://github.com/entrius/gittensor) network (Bittensor Subnet 74).

**Only PRs that demonstrably improve routing accuracy are merged for TAO rewards.**
General improvements (docs, fixes, refactors) are welcome but earn no TAO.

## Two ways to contribute

### 1. Submit a trained routing head (earns TAO)

Compete to train the best coordinator. If your head beats the current best
accuracy on the hidden benchmark, your PR gets merged and you earn TAO.

→ Read [SUBMITTING.md](SUBMITTING.md) for the full guide.

Quick start:
```bash
# 1. Train
python -m trinity.train --benchmark math500 --run-name my-run

# 2. Pack
python scripts/pack_submission.py --run-dir experiments/math500/my-run \
    --miner-name your-name --benchmark math500

# 3. Submit as PR
# See SUBMITTING.md for PR format
```

### 2. Improve the infrastructure (no TAO, but appreciated)

Bug fixes, documentation, test coverage, and code quality improvements are
always welcome. These PRs follow normal open-source workflow.

## Setup

```bash
git clone https://github.com/<org>/tinyrouter.git
cd tinyrouter
pip install -e ".[dev]"
source ~/.config/trinity/secrets.env   # exports OPENROUTER_API_KEY
```

See [AGENTS.md](AGENTS.md) for the full compute environment description.

## Code style

- Match the existing code. Look at adjacent files before writing.
- Line length: 100 characters
- Docstrings: Google style. Every public function gets one.
- Type annotations: required for all public functions.
- Lazy imports: torch, transformers, datasets are imported inside functions
  (the dev box has no GPU).

## Testing

```bash
pytest tests/
ruff check src/
mypy src/
```

Pull requests also run the offline GitHub Actions lane described in
[docs/CI.md](docs/CI.md).

## Journal

Every non-obvious decision, mistake, or finding goes in
[docs/JOURNAL.md](docs/JOURNAL.md). Read it before starting significant work.

## Scoring rules for routing heads

| Rule | Detail |
|---|---|
| **Rate limit** | 1 submission per benchmark per week (enforced in code) |
| **Original work** | Every submission is checked against all previous heads via cosine similarity. Copies are rejected |
| **Hidden benchmark** | Encrypted, never revealed — don't ask for it |
| **Score feedback** | Winners get full results. Losers get only composite score + delta — no component breakdown |
| **Receipt** | Must show cost ≥ $15 and a plausible CMA-ES fitness curve. Fabricated receipts fail validation |
| **Evaluation** | 70% cached single-turn + 15% live multi-turn + 10% efficiency + 5% novelty (computed vs current king) |

See [SUBMITTING.md](SUBMITTING.md) for full competition rules.
