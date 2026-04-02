"""Bounded-budget hyperparameter tuning for top static models.

Uses RandomizedSearchCV with 3-fold CV for fast search, then re-evaluates
the best params on 5-fold CV to get final publishable numbers.

Models tuned: RandomForest, XGBoost, LightGBM
(HybridLSTM excluded — too slow for random search; tune manually if needed)

Usage:
    python src/seqcredit_model/run_hyperparameter_tuning.py

Outputs:
    data/tuning_results.csv  — best params + CV scores per model
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from seqcredit_model.config import DATA_DIR, RANDOM_SEED
    from seqcredit_model.credit_model import (
        CreditRiskDataLoader,
        LightGBMModel,
        RandomForestModel,
        XGBoostModel,
        set_random_seeds,
    )
    from seqcredit_model.run_cv_benchmark import run_static_model_cv
except ModuleNotFoundError:
    from config import DATA_DIR, RANDOM_SEED
    from credit_model import (
        CreditRiskDataLoader,
        LightGBMModel,
        RandomForestModel,
        XGBoostModel,
        set_random_seeds,
    )
    from run_cv_benchmark import run_static_model_cv

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

N_ITER = 40        # random search trials per model
CV_SEARCH = 3      # folds during search (fast)
CV_FINAL = 5       # folds for final eval (publishable)
RANDOM_SEED = RANDOM_SEED

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
SEARCH_SPACES = {
    "RandomForest": {
        "model_cls": RandomForestClassifier,
        "param_dist": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 7, 10, 15, None],
            "min_samples_leaf": [1, 2, 4, 8],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": ["balanced"],
        },
        "wrapper_cls": RandomForestModel,
        "fixed_params": {"random_state": RANDOM_SEED},
    },
    "XGBoost": {
        "model_cls": xgb.XGBClassifier,
        "param_dist": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.3],
        },
        "wrapper_cls": XGBoostModel,
        "fixed_params": {"random_state": RANDOM_SEED},
    },
    "LightGBM": {
        "model_cls": lgb.LGBMClassifier,
        "param_dist": {
            "n_estimators": [100, 200, 300, 500],
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "min_child_samples": [10, 20, 30, 50],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "class_weight": ["balanced"],
        },
        "wrapper_cls": LightGBMModel,
        "fixed_params": {"random_state": RANDOM_SEED},
    },
}


def search_best_params(
    model_name: str,
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Run RandomizedSearchCV and return best params + search score."""
    print(f"\n  Searching {model_name} ({N_ITER} trials, {CV_SEARCH}-fold)...")

    scale_pos = (y == 0).sum() / (y == 1).sum()

    # XGBoost needs scale_pos_weight, not class_weight
    param_dist = dict(config["param_dist"])
    if model_name == "XGBoost":
        param_dist["scale_pos_weight"] = [scale_pos]

    skf = StratifiedKFold(n_splits=CV_SEARCH, shuffle=True, random_state=RANDOM_SEED)

    # Scale features for the search (StandardScaler fit inside each fold not
    # practical here; scale once on full train — acceptable for search phase)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = config["model_cls"](random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    search.fit(X_scaled, y)

    print(f"    Best search AUC-ROC: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")

    return search.best_params_, search.best_score_


def eval_best_params(
    model_name: str,
    config: dict,
    best_params: dict,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Re-evaluate best params with proper 5-fold CV (fold-level scaler)."""
    print(f"  Final 5-fold CV for {model_name}...")

    # Map raw sklearn params back to wrapper params
    scale_pos = (y == 0).sum() / (y == 1).sum()
    wrapper_params = dict(config["fixed_params"])

    if model_name == "RandomForest":
        # RandomForestModel wrapper only accepts these params (no max_features)
        for k in ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "class_weight"]:
            if k in best_params:
                wrapper_params[k] = best_params[k]
    elif model_name == "XGBoost":
        wrapper_params["scale_pos_weight"] = scale_pos
        # XGBoostModel wrapper accepts: n_estimators, max_depth, learning_rate,
        # min_child_weight, subsample, colsample_bytree (no gamma)
        for k in ["n_estimators", "max_depth", "learning_rate", "subsample",
                  "colsample_bytree", "min_child_weight"]:
            if k in best_params:
                wrapper_params[k] = best_params[k]
    elif model_name == "LightGBM":
        # LightGBMModel wrapper accepts: n_estimators, max_depth, learning_rate,
        # num_leaves, class_weight (no subsample/colsample/min_child_samples/reg_*)
        for k in ["n_estimators", "num_leaves", "learning_rate", "class_weight"]:
            if k in best_params:
                wrapper_params[k] = best_params[k]

    _, summary, _ = run_static_model_cv(
        config["wrapper_cls"], wrapper_params, X, y, n_splits=CV_FINAL
    )
    print(f"    Final AUC-ROC: {summary['auc_roc']:.4f} ± {summary['auc_roc_std']:.4f}")
    return summary, wrapper_params


def main():
    print("=" * 60)
    print("Hyperparameter Tuning")
    print(f"  {N_ITER} trials per model, {CV_SEARCH}-fold search, {CV_FINAL}-fold final eval")
    print("=" * 60)

    set_random_seeds(RANDOM_SEED)

    print("\nLoading data...")
    loader = CreditRiskDataLoader()
    static_data = loader.prepare_static_splits()
    X_df = static_data["X_train"]
    y = static_data["y_train"]
    X = X_df.values

    print(f"  Train users: {len(y)}, features: {X.shape[1]}, positive rate: {y.mean():.4f}")

    all_results = []
    best_params_all = {}

    for model_name, config in SEARCH_SPACES.items():
        t0 = time.time()
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")

        best_params, search_auc = search_best_params(model_name, config, X, y)
        final_summary, final_params = eval_best_params(model_name, config, best_params, X, y)

        elapsed = time.time() - t0
        best_params_all[model_name] = final_params

        all_results.append({
            "model": model_name,
            "search_auc_roc": round(search_auc, 4),
            "final_auc_roc": round(final_summary["auc_roc"], 4),
            "final_auc_roc_std": round(final_summary["auc_roc_std"], 4),
            "final_auc_pr": round(final_summary["auc_pr"], 4),
            "final_f1": round(final_summary["f1"], 4),
            "final_brier": round(final_summary["brier"], 4),
            "final_ece": round(final_summary["ece"], 4),
            "best_params": json.dumps(final_params),
            "elapsed_s": round(elapsed, 1),
        })

    # Save results
    out_df = pd.DataFrame(all_results)
    out_path = Path(__file__).parent.parent.parent / "data" / "tuning_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Search AUC':>10} {'Final AUC':>10} {'±':>6} {'AUC-PR':>8}")
    for r in all_results:
        print(
            f"  {r['model']:<13} {r['search_auc_roc']:>10.4f} "
            f"{r['final_auc_roc']:>10.4f} {r['final_auc_roc_std']:>6.4f} "
            f"{r['final_auc_pr']:>8.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
