"""
reset.py — Clean up generated artifacts for a fresh run.

Flags (combinable):
  --models      Trained models (models/)
  --data        Cached/generated data files:
                  data/lstm_sequences.npz
                  data/lstm_test_arrays.npz
                  data/user_features.csv
                  data/model_comparison.csv
  --data-full   Everything in --data PLUS preserved source data:
                  data/user_transactions/   (10,000 per-user CSVs)
                  data/user_labels.csv
                  data/synthetic_params.json
  --yes         Skip confirmation prompt

No flags = --models --data (default, does NOT include --data-full).

Usage:
  python reset.py                        # models + cached data (default)
  python reset.py --yes                  # same, skip confirmation
  python reset.py --models               # only trained models
  python reset.py --data                 # only cached/generated data
  python reset.py --models --data        # same as default
  python reset.py --data-full            # cached data + source data
  python reset.py --models --data-full   # full wipe except nothing
  python reset.py --models --data-full --yes  # full wipe, no prompt
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

MODELS = [
    ROOT / "models" / "lr_model.pkl",
    ROOT / "models" / "xgb_model.pkl",
    ROOT / "models" / "rf_model.pkl",
    ROOT / "models" / "lgbm_model.pkl",
    ROOT / "models" / "lstm_model.keras",
    ROOT / "models" / "lstm_model.keras.json",
    ROOT / "models" / "hybrid_model.keras",
    ROOT / "models" / "hybrid_model.keras.json",
]

DATA_FILES = [
    ROOT / "data" / "lstm_sequences.npz",
    ROOT / "data" / "lstm_test_arrays.npz",
    ROOT / "data" / "user_features.csv",
    ROOT / "data" / "model_comparison.csv",
]

# Removed only with --data-full
DATA_FULL_FILES = [
    ROOT / "data" / "user_labels.csv",
    ROOT / "data" / "synthetic_params.json",
]

DATA_FULL_DIRS = [
    ROOT / "data" / "user_transactions",
]


def remove_files(files: list[Path]) -> None:
    for path in files:
        if path.exists():
            path.unlink()
            print(f"  deleted  {path.relative_to(ROOT)}")
        else:
            print(f"  skipped  {path.relative_to(ROOT)}  (not found)")


def remove_dirs(dirs: list[Path]) -> None:
    for path in dirs:
        if path.exists():
            shutil.rmtree(path)
            print(f"  deleted  {path.relative_to(ROOT)}/  ({len(list(path.glob('*')))} files)" if path.exists() else f"  deleted  {path.relative_to(ROOT)}/")
        else:
            print(f"  skipped  {path.relative_to(ROOT)}/  (not found)")


def collect_existing(files: list[Path], dirs: list[Path]) -> list[str]:
    labels = []
    for p in files:
        if p.exists():
            labels.append(str(p.relative_to(ROOT)))
    for p in dirs:
        if p.exists():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            labels.append(f"{p.relative_to(ROOT)}/  ({count} files)")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset seqcredit-model for a fresh run.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--models", action="store_true", help="Remove trained models")
    parser.add_argument("--data", action="store_true", help="Remove cached/generated data files")
    parser.add_argument("--data-full", dest="data_full", action="store_true",
                        help="Remove --data files AND source data (transactions, labels, params)")
    args = parser.parse_args()

    # Default (no flags): equivalent to --models --data
    default = not any([args.models, args.data, args.data_full])
    do_models = args.models or default
    do_data = args.data or args.data_full or default
    do_data_full = args.data_full

    target_files = []
    target_dirs = []
    if do_models:
        target_files += MODELS
    if do_data:
        target_files += DATA_FILES
    if do_data_full:
        target_files += DATA_FULL_FILES
        target_dirs += DATA_FULL_DIRS

    existing = collect_existing(target_files, target_dirs)
    if not existing:
        print("Nothing to clean — already fresh.")
        return

    print("The following will be deleted:")
    for label in existing:
        print(f"  {label}")

    if do_data_full:
        print("\n  WARNING: --data-full will remove source data that requires")
        print("  re-running synthetic_data.py to rebuild.")

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    if do_models:
        print("Removing trained models...")
        remove_files(MODELS)
    if do_data:
        print("Removing cached/generated data files...")
        remove_files(DATA_FILES)
    if do_data_full:
        print("Removing source data...")
        remove_files(DATA_FULL_FILES)
        remove_dirs(DATA_FULL_DIRS)

    print("\nDone.")
    if do_data_full:
        print("Rebuild source data:  python -m seqcredit_model.synthetic_data")
    print("Start fresh:          jupyter notebook notebooks/credit_risk_model.ipynb")


if __name__ == "__main__":
    main()
