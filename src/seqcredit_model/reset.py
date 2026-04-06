"""Clean up generated artifacts for a fresh run."""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = [
    MODELS_DIR / "lr_model.pkl",
    MODELS_DIR / "xgb_model.pkl",
    MODELS_DIR / "rf_model.pkl",
    MODELS_DIR / "lgbm_model.pkl",
    MODELS_DIR / "lstm_model.keras",
    MODELS_DIR / "lstm_model.json",
    MODELS_DIR / "hybrid_model.keras",
    MODELS_DIR / "hybrid_model.json",
]

DATA_FILES = [
    DATA_DIR / "lstm_sequences.npz",
    DATA_DIR / "lstm_test_arrays.npz",
    DATA_DIR / "user_features.csv",
    DATA_DIR / "model_comparison.csv",
    DATA_DIR / "tuning_results.csv",
    DATA_DIR / "significance_tests.csv",
    DATA_DIR / "cv_manifest.json",
    DATA_DIR / "ablation_features.csv",
    DATA_DIR / "cv_results_y_default.csv",
    DATA_DIR / "cv_results_y_bad.csv",
    DATA_DIR / "cv_results_intermediate_y_default.csv",
    DATA_DIR / "cv_results_intermediate_y_bad.csv",
]

DATA_FULL_FILES = [
    DATA_DIR / "user_labels.csv",
]

DATA_FULL_DIRS = [
    DATA_DIR / "user_transactions",
]


def remove_files(files: list[Path]) -> None:
    for path in files:
        if path.exists():
            path.unlink()
            print(f"  deleted  {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  skipped  {path.relative_to(PROJECT_ROOT)}  (not found)")


def remove_dirs(dirs: list[Path]) -> None:
    for path in dirs:
        if path.exists():
            count = sum(1 for _ in path.glob("*") if _.is_file())
            shutil.rmtree(path)
            print(f"  deleted  {path.relative_to(PROJECT_ROOT)}/  ({count} files)")
        else:
            print(f"  skipped  {path.relative_to(PROJECT_ROOT)}/  (not found)")


def collect_existing(files: list[Path], dirs: list[Path]) -> list[str]:
    labels = []
    for p in files:
        if p.exists():
            labels.append(str(p.relative_to(PROJECT_ROOT)))
    for p in dirs:
        if p.exists():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            labels.append(f"{p.relative_to(PROJECT_ROOT)}/  ({count} files)")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset seqcredit-model for a fresh run."
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--models", action="store_true", help="Remove trained models")
    parser.add_argument(
        "--data", action="store_true", help="Remove cached/generated data files"
    )
    parser.add_argument(
        "--data-full",
        dest="data_full",
        action="store_true",
        help="Remove --data files AND source data (transactions, labels)",
    )
    args = parser.parse_args()

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
        print("  re-running python -m seqcredit_model.synthesize to rebuild.")

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
        print("Rebuild source data:  python -m seqcredit_model.synthesize")
    print("Start fresh:          jupyter notebook notebooks/model.ipynb")


if __name__ == "__main__":
    main()
