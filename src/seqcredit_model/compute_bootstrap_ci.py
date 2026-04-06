"""
Compute bootstrap confidence intervals for all trained models.

This script loads saved models and test data, then computes 95% bootstrap CIs
for all evaluation metrics. The results are saved to data/model_comparison.csv
with additional CI columns.

Usage:
    python src/seqcredit_model/compute_bootstrap_ci.py
"""

import sys
from pathlib import Path

# Ensure seqcredit_model is importable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np  # noqa: E402

from seqcredit_model.config import DATA_DIR, MODELS_DIR  # noqa: E402
from seqcredit_model.credit_model import (  # noqa: E402
    CreditRiskDataLoader,
    HybridLSTMModel,
    LightGBMModel,
    LogisticRegressionModel,
    LSTMModel,
    ModelEvaluator,
    RandomForestModel,
    set_random_seeds,
    XGBoostModel,
)

# Configuration
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RANDOM_SEED = 42


def main():
    """Compute bootstrap CIs for all models and save results."""
    print("=" * 60)
    print("Computing Bootstrap Confidence Intervals")
    print("=" * 60)
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Confidence level: {CI_LEVEL * 100:.0f}%")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    set_random_seeds(RANDOM_SEED)

    # Load data
    print("Loading data...")
    loader = CreditRiskDataLoader()
    static_data = loader.prepare_static_splits()

    # Load LSTM test arrays
    lstm_arrays_path = DATA_DIR / "lstm_test_arrays.npz"
    if not lstm_arrays_path.exists():
        print(f"[ERROR] LSTM test arrays not found at {lstm_arrays_path}")
        print("  Run the model.ipynb notebook first to generate them.")
        return 1

    lstm_data = np.load(lstm_arrays_path)
    X_test_seq = lstm_data["X_test_seq"]
    X_test_static = lstm_data["X_test_static"]
    y_test = lstm_data["y_test"]

    print(f"  Test set size: {len(y_test)}")
    print(f"  Default rate: {y_test.mean():.4f}")
    print()

    # Load models
    print("Loading trained models...")
    try:
        lr_model = LogisticRegressionModel.load(str(MODELS_DIR / "lr_model.pkl"))
        xgb_model = XGBoostModel.load(str(MODELS_DIR / "xgb_model.pkl"))
        rf_model = RandomForestModel.load(str(MODELS_DIR / "rf_model.pkl"))
        lgbm_model = LightGBMModel.load(str(MODELS_DIR / "lgbm_model.pkl"))
        lstm_model = LSTMModel.load(str(MODELS_DIR / "lstm_model"))
        hybrid_model = HybridLSTMModel.load(str(MODELS_DIR / "hybrid_model"))
        print("  All models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print(
            "  Run the model.ipynb notebook first to train and save models."
        )
        return 1
    print()

    # Generate predictions
    print("Generating predictions...")
    lr_proba = lr_model.predict_proba(static_data["X_test_scaled"])
    xgb_proba = xgb_model.predict_proba(static_data["X_test"])
    rf_proba = rf_model.predict_proba(static_data["X_test"])
    lgbm_proba = lgbm_model.predict_proba(static_data["X_test"])
    lstm_proba = lstm_model.predict_proba(X_test_seq)
    hybrid_proba = hybrid_model.predict_proba(X_test_seq, X_test_static)
    print("  Predictions generated.")
    print()

    # Create evaluator and add models with bootstrap CIs
    print("Computing bootstrap confidence intervals...")
    evaluator = ModelEvaluator(y_test)

    models = [
        ("Logistic Regression", lr_proba),
        ("XGBoost", xgb_proba),
        ("Random Forest", rf_proba),
        ("LightGBM", lgbm_proba),
        ("Sequential LSTM", lstm_proba),
        ("Hybrid LSTM", hybrid_proba),
    ]

    for name, proba in models:
        evaluator.add_model(
            name, proba, compute_ci=True, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL
        )
    print()

    # Get results
    print("Results:")
    print("-" * 60)
    comparison = evaluator.get_comparison_table()
    print(comparison.round(4).to_string())
    print()

    # Display formatted results with CIs
    print("\nResults with 95% Confidence Intervals:")
    print("-" * 60)
    formatted = evaluator.format_results_with_ci(decimals=3)
    print(formatted.to_string())
    print()

    # Save CSV with CI columns
    csv_output = evaluator.get_comparison_csv_with_ci()
    output_path = DATA_DIR / "model_comparison.csv"
    csv_output.to_csv(output_path)
    print(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
