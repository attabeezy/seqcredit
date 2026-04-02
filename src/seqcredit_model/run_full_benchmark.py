"""Reproducibility script for full CV benchmark.

This script runs the complete 5-fold stratified CV benchmark with:
- Data loading and validation
- CV for all 6 models on both targets (y_default, y_bad)
- Statistical significance tests
- Output to data/ directory

Usage:
    python -m seqcredit_model.run_full_benchmark
"""

import sys
from pathlib import Path

RANDOM_SEED = 42


def clear_caches():
    """Clear any stale caches that might affect reproducibility."""
    print("[CACHE] Checking for stale caches...")

    cache_files = [
        "data/lstm_sequences.npz",
        "data/cv_results_y_default.csv",
        "data/cv_results_bad.csv",
        "data/significance_tests.csv",
        "data/cv_manifest.json",
    ]

    for cache_file in cache_files:
        p = Path(cache_file)
        if p.exists():
            print(f"  - Removing stale {cache_file}")
            p.unlink()

    print("[CACHE] Done\n")


def set_global_seed():
    """Set all random seeds for reproducibility."""
    import numpy as np
    import random
    import tensorflow as tf

    print(f"[SEED] Setting global random seed to {RANDOM_SEED}...")
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    print("[SEED] Done\n")


def main():
    print("=" * 60)
    print("FULL CV BENCHMARK - REPRODUCIBILITY RUN")
    print("=" * 60)
    print()

    clear_caches()
    set_global_seed()

    print("[RUN] Starting CV benchmark...")
    print()

    from seqcredit_model.run_cv_benchmark import main as run_cv

    try:
        run_cv()
    except Exception as e:
        print(f"\n[ERROR] CV benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("REPRODUCIBILITY RUN COMPLETE")
    print("=" * 60)
    print()
    print("Output files:")
    print("  - data/cv_results_y_default.csv")
    print("  - data/cv_results_bad.csv")
    print("  - data/significance_tests.csv")
    print("  - data/cv_manifest.json")


if __name__ == "__main__":
    main()
