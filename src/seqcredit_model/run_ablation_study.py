"""Feature group ablation study for Paper A.

Quantifies the contribution of each feature group by running 5-fold CV on
RandomForest with one group dropped at a time, and with each group alone.
Answers the key Paper A question: which feature groups drive the 0.83 AUC?

Usage:
    python src/seqcredit_model/run_ablation_study.py

Outputs:
    data/ablation_features.csv  — AUC-ROC / AUC-PR per ablation condition
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from seqcredit_model.config import RANDOM_SEED
    from seqcredit_model.credit_model import CreditRiskDataLoader, RandomForestModel, set_random_seeds
    from seqcredit_model.run_cv_benchmark import run_static_model_cv
except ModuleNotFoundError:
    from config import RANDOM_SEED
    from credit_model import CreditRiskDataLoader, RandomForestModel, set_random_seeds
    from run_cv_benchmark import run_static_model_cv

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------
# Keys map to human-readable names; values list the column names that belong
# to that group (as they appear in user_features.csv / after load_static_data).
FEATURE_GROUPS = {
    "amount_stats": [
        "total_volume",
        "avg_transaction_amount",
        "median_transaction_amount",
        "std_transaction_amount",
        "max_transaction_amount",
        "min_transaction_amount",
        "cv_transaction_amount",
    ],
    "txn_type_mix": [
        "pct_transfers",
        "pct_debits",
        "pct_cashouts",
        "pct_payments",
    ],
    "temporal_patterns": [
        "avg_hours_between_txns",
        "pct_weekend_txns",
        "pct_night_txns",
        "pct_early_morning_txns",
    ],
    "balance_dynamics": [
        "avg_balance",
        "min_balance",
        "max_balance",
        "balance_volatility",
        "pct_low_balance_txns",
    ],
    "fee_behaviour": [
        "total_fees_paid",
        "avg_fees_per_txn",
        "pct_txns_with_fees",
    ],
    "behavioural_diversity": [
        "unique_recipients",
        "recipient_concentration",
        "unique_txn_types",
    ],
    "activity_intensity": [
        "obs_txn_count",
        "account_age_days",
        "transactions_per_day",
    ],
    "loan_history": [
        "total_loan_volume",
        "avg_loan_amount",
        "max_loan_amount",
        "loan_to_total_volume_ratio",
        "pct_credit_transactions",
        "loan_timing_in_sequence",
        "avg_balance_at_loan",
        "min_balance_at_loan",
        "balance_to_loan_ratio_at_disbursement",
    ],
}

RF_PARAMS = {"class_weight": "balanced", "random_state": RANDOM_SEED}
N_SPLITS = 5


def run_condition(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """Run 5-fold CV on RandomForest for one ablation condition."""
    t0 = time.time()
    _, summary, _ = run_static_model_cv(RandomForestModel, RF_PARAMS, X, y, n_splits=N_SPLITS)
    elapsed = time.time() - t0
    return {
        "condition": name,
        "n_features": X.shape[1],
        "auc_roc": round(summary["auc_roc"], 4),
        "auc_roc_std": round(summary["auc_roc_std"], 4),
        "auc_pr": round(summary["auc_pr"], 4),
        "auc_pr_std": round(summary["auc_pr_std"], 4),
        "f1": round(summary["f1"], 4),
        "brier": round(summary["brier"], 4),
        "ece": round(summary["ece"], 4),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("Feature Group Ablation Study")
    print("=" * 60)

    set_random_seeds(RANDOM_SEED)

    # Load data
    print("\nLoading data...")
    loader = CreditRiskDataLoader()
    static_data = loader.prepare_static_splits()
    X_df = static_data["X_train"]          # DataFrame with column names
    y = static_data["y_train"]
    feature_names = static_data["feature_names"]

    print(f"  Train users: {len(y)}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Positive rate: {y.mean():.4f}")
    print(f"  Feature names: {feature_names[:5]} ... ({len(feature_names)} total)")

    # Warn about any group columns not found in the actual feature set
    all_group_cols = [c for cols in FEATURE_GROUPS.values() for c in cols]
    missing = [c for c in all_group_cols if c not in X_df.columns]
    if missing:
        print(f"\n  [WARNING] {len(missing)} group columns not found in feature set: {missing}")

    results = []

    # --- Baseline: all features ---
    print("\n[Baseline] All features...")
    results.append(run_condition("ALL_FEATURES", X_df.values, y))
    print(f"  AUC-ROC = {results[-1]['auc_roc']:.4f} ({results[-1]['n_features']} features)")

    # --- Drop-one-group ablations ---
    print("\n[Drop-one-group ablations]")
    for group_name, cols in FEATURE_GROUPS.items():
        drop_cols = [c for c in cols if c in X_df.columns]
        if not drop_cols:
            print(f"  {group_name}: skipped (no matching columns in feature set)")
            continue
        X_ablated = X_df.drop(columns=drop_cols).values
        label = f"DROP_{group_name}"
        print(f"  {label} ({len(drop_cols)} cols dropped, {X_ablated.shape[1]} remaining)...", end=" ", flush=True)
        row = run_condition(label, X_ablated, y)
        baseline_auc = results[0]["auc_roc"]
        delta = row["auc_roc"] - baseline_auc
        print(f"AUC-ROC = {row['auc_roc']:.4f}  (Δ = {delta:+.4f})")
        results.append(row)

    # --- Single-group-only ablations ---
    print("\n[Single-group-only ablations]")
    for group_name, cols in FEATURE_GROUPS.items():
        keep_cols = [c for c in cols if c in X_df.columns]
        if not keep_cols:
            continue
        X_only = X_df[keep_cols].values
        label = f"ONLY_{group_name}"
        print(f"  {label} ({len(keep_cols)} features)...", end=" ", flush=True)
        row = run_condition(label, X_only, y)
        print(f"AUC-ROC = {row['auc_roc']:.4f}")
        results.append(row)

    # --- Save ---
    out_df = pd.DataFrame(results)
    out_path = Path(__file__).parent.parent.parent / "data" / "ablation_features.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (drop-one, sorted by AUC-ROC drop)")
    print("=" * 60)
    baseline_auc = results[0]["auc_roc"]
    drop_rows = [r for r in results if r["condition"].startswith("DROP_")]
    drop_rows.sort(key=lambda r: r["auc_roc"])  # lowest AUC = most important group

    print(f"{'Condition':<35} {'AUC-ROC':>8} {'Delta':>8}")
    print(f"  {'ALL_FEATURES':<33} {baseline_auc:>8.4f} {'—':>8}")
    for r in drop_rows:
        delta = r["auc_roc"] - baseline_auc
        print(f"  {r['condition']:<33} {r['auc_roc']:>8.4f} {delta:>+8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
