# TinyRouter Baselines

Reference baselines that every submission is compared against.
Run these before training to see what you're beating.

## Quick start

```bash
# Always pick one model (the floor for each individual model)
python baselines/always_model.py --model qwen3.5 --benchmark math500
python baselines/always_model.py --model gemini-flash-lite --benchmark math500
python baselines/always_model.py --model deepseek-v4-flash --benchmark math500

# Random routing (the floor — your head MUST beat this)
python baselines/random_router.py --benchmark math500 --seeds 100

# Perfect-router oracle ceiling (the ceiling — how much headroom exists)
python scripts/oracle_ceiling.py --collect --benchmark math500
python scripts/oracle_ceiling.py --analyze experiments/final/oracle_matrix_math500.json
```

## What each baseline means

| Baseline | What it tells you |
|---|---|
| **always_model** | The accuracy if you pick ONE model for every question. The best of these is "best-single" — the simplest strategy to beat. |
| **random_router** | The accuracy if you pick a random model + random role each turn. Your head must beat this to demonstrate any routing intelligence. |
| **oracle_ceiling** | The theoretical maximum if you always picked the RIGHT model per question (winner's-curse-debiased). The gap between best-single and oracle is the **routing headroom** — the maximum your head can gain. |
