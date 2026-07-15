#!/usr/bin/env python3
"""One-time launch preparation for the TinyRouter competition.

Run this ONCE on the GPU box (where numpy, datasets, torch, and the API key
are available). It performs every step of the launch checklist:

  1. Verify pool models are reachable
  2. Verify each benchmark loads real data (not toy fallback)
  3. Build hidden benchmarks for any missing benchmark
  4. Collect oracle matrices for missing baselines
  5. Update leaderboard.json with collected baselines
  6. Report readiness

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    export BENCHMARK_PASSWORD=...
    python scripts/prepare_launch.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

BENCHMARKS = ("math500", "mmlu", "livecodebench")


def _check_env() -> bool:
    """Verify required env vars are set."""
    ok = True
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        ok = False
    else:
        print("✅ OPENROUTER_API_KEY set")
    if not os.environ.get("BENCHMARK_PASSWORD"):
        print("❌ BENCHMARK_PASSWORD not set")
        ok = False
    else:
        print("✅ BENCHMARK_PASSWORD set")
    return ok


def _bench_dir() -> Path:
    d = os.environ.get("TINYROUTER_BENCHMARK_DIR",
                       str(_REPO.parent / "tinyrouter-benchmark"))
    return Path(d)


def step1_verify_pool() -> bool:
    """Step 1: Verify all pool models are reachable."""
    print("\n" + "=" * 60)
    print("STEP 1: Verify pool models")
    print("=" * 60)
    try:
        from trinity.llm.openrouter_client import OpenRouterPool
        pool = OpenRouterPool(str(_REPO / "configs" / "models.yaml"))
        models = list(pool.models)
        print(f"  Pool models: {models}")
        if len(models) >= 3:
            print("✅ Pool has 3+ models")
            return True
        else:
            print(f"❌ Pool has only {len(models)} models (expected 3)")
            return False
    except Exception as e:
        print(f"❌ Pool verification failed: {e}")
        return False


def step2_verify_data() -> dict[str, bool]:
    """Step 2: Verify each benchmark loads real data (not toy fallback)."""
    print("\n" + "=" * 60)
    print("STEP 2: Verify benchmark data loading")
    print("=" * 60)
    results = {}
    try:
        from trinity.adapters import get_adapter
    except ImportError as e:
        print(f"❌ Cannot import adapters: {e}")
        return {b: False for b in BENCHMARKS}

    for bench in BENCHMARKS:
        try:
            adapter = get_adapter(bench)
            tasks = adapter.load_tasks("test", max_items=5, seed=0)
            n = len(tasks)
            if n <= 3:
                print(f"  ⚠️  {bench}: only {n} tasks — likely toy fallback")
                results[bench] = False
            else:
                print(f"✅ {bench}: {n} tasks loaded (real data)")
                results[bench] = True
        except Exception as e:
            print(f"❌ {bench}: load failed: {e}")
            results[bench] = False
    return results


def step3_build_benchmarks(data_ok: dict[str, bool]) -> None:
    """Step 3: Build hidden benchmarks for benchmarks that pass data check."""
    print("\n" + "=" * 60)
    print("STEP 3: Build hidden benchmarks")
    print("=" * 60)
    bench_dir = _bench_dir()

    for bench in BENCHMARKS:
        if not data_ok.get(bench):
            print(f"  ⏭️  Skipping {bench} (data not verified)")
            continue

        eval_path = bench_dir / bench / "eval.json"
        if eval_path.exists():
            print(f"✅ {bench}: hidden benchmark already exists")
            continue

        print(f"  🔨 Building hidden benchmark for {bench}...")
        print(f"     (This pre-caches model answers — costs API $$)")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "build_benchmark.py"),
             "--benchmark", bench],
            cwd=str(_REPO),
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            print(f"✅ {bench}: hidden benchmark built")
        else:
            print(f"❌ {bench}: build failed (exit {result.returncode})")


def step4_collect_oracle() -> None:
    """Step 4: Collect oracle matrices for missing baselines."""
    print("\n" + "=" * 60)
    print("STEP 4: Collect oracle matrices")
    print("=" * 60)

    lb_path = _REPO / "leaderboard.json"
    lb = json.loads(lb_path.read_text()) if lb_path.exists() else {}
    benches = lb.get("benchmarks", {})

    for bench in BENCHMARKS:
        entry = benches.get(bench, {})
        has_baselines = (entry.get("baseline_random") is not None
                         and entry.get("best_single_model") is not None)
        if has_baselines:
            print(f"✅ {bench}: baselines already set")
            continue

        print(f"  📊 Collecting oracle matrix for {bench}...")
        print(f"     (This calls each model K times — costs API $$)")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "oracle_ceiling.py"),
             "--collect", "--benchmark", bench, "--split", "test"],
            cwd=str(_REPO),
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            print(f"✅ {bench}: oracle matrix collected")
        else:
            print(f"❌ {bench}: oracle collection failed (exit {result.returncode})")


def step5_report() -> None:
    """Step 5: Report launch readiness."""
    print("\n" + "=" * 60)
    print("LAUNCH READINESS REPORT")
    print("=" * 60)

    bench_dir = _bench_dir()
    lb = json.loads((_REPO / "leaderboard.json").read_text())
    benches = lb.get("benchmarks", {})

    all_ready = True
    for bench in BENCHMARKS:
        eval_exists = (bench_dir / bench / "eval.json").exists()
        audit_exists = (bench_dir / bench / "audit.json").exists()
        live_exists = (bench_dir / bench / "live.json").exists()
        entry = benches.get(bench, {})
        has_baselines = (entry.get("baseline_random") is not None
                         and entry.get("best_single_model") is not None)

        hidden_ok = eval_exists and audit_exists and live_exists
        status = "✅" if (hidden_ok and has_baselines) else "❌"
        print(f"  {status} {bench}:")
        print(f"     hidden benchmark: eval={eval_exists}, audit={audit_exists}, live={live_exists}")
        print(f"     baselines: random={entry.get('baseline_random')}, "
              f"best_single={entry.get('best_single_model')}, "
              f"oracle={entry.get('oracle_ceiling')}")

        if not hidden_ok or not has_baselines:
            all_ready = False

    print()
    if all_ready:
        print("🚀 READY FOR LAUNCH — all 3 benchmarks have hidden sets + baselines.")
        print("   Miners can now: clone → train --benchmarks → pack → submit.")
    else:
        print("⚠️  NOT READY — some benchmarks are missing hidden sets or baselines.")
        print("   Fix the ❌ items above before announcing the competition.")


def main() -> None:
    print("TinyRouter Competition — Launch Preparation")
    print("=" * 60)

    if not _check_env():
        print("\n❌ Missing environment variables. Set them and re-run:")
        print("   export OPENROUTER_API_KEY=sk-or-v1-...")
        print("   export BENCHMARK_PASSWORD=...")
        sys.exit(1)

    data_ok = step2_verify_data()
    step1_verify_pool()
    step3_build_benchmarks(data_ok)
    step4_collect_oracle()
    step5_report()


if __name__ == "__main__":
    main()
