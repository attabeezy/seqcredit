"""5-fold stratified cross-validation benchmark with statistical significance tests."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from seqcredit_model.config import (
        DATA_DIR,
        LSTM_CACHE_FILE,
        RAW_SEQ_FILE,
        RANDOM_SEED,
        TRANSACTIONS_DIR,
        USER_FEATURES_FILE,
        USER_LABELS_FILE,
        get_runtime_lstm_cache_file,
        get_runtime_raw_seq_file,
        get_runtime_user_features_file,
        get_runtime_user_labels_file,
    )
    from seqcredit_model.credit_model import (
        CreditRiskDataLoader,
        GRUModel,
        HybridGRUModel,
        HybridLSTMModel,
        LightGBMModel,
        LogisticRegressionModel,
        LSTMModel,
        RandomForestModel,
        XGBoostModel,
        set_random_seeds,
    )
except ModuleNotFoundError:
    from config import (
        DATA_DIR,
        LSTM_CACHE_FILE,
        RAW_SEQ_FILE,
        RANDOM_SEED,
        TRANSACTIONS_DIR,
        USER_FEATURES_FILE,
        USER_LABELS_FILE,
        get_runtime_lstm_cache_file,
        get_runtime_raw_seq_file,
        get_runtime_user_features_file,
        get_runtime_user_labels_file,
    )
    from credit_model import (
        CreditRiskDataLoader,
        GRUModel,
        HybridGRUModel,
        HybridLSTMModel,
        LightGBMModel,
        LogisticRegressionModel,
        LSTMModel,
        RandomForestModel,
        XGBoostModel,
        set_random_seeds,
    )

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95


def is_ephemeral_run() -> bool:
    """Return whether the current run should avoid persistent artifacts."""
    return os.environ.get("SEQCREDIT_EPHEMERAL") == "1"


def get_runtime_data_dir() -> Path:
    """Return a writable directory for CV outputs in the current runtime.

    In ephemeral Databricks runs the features file lives under a temp dir
    (e.g. /tmp/seqcredit_model/).  CV results are written there so that
    cell 24 can read them in the same session.  Outside ephemeral mode the
    committed data/ directory is used as usual.
    """
    runtime_features = get_runtime_user_features_file()
    if runtime_features != USER_FEATURES_FILE:
        return runtime_features.parent
    return DATA_DIR

def has_sequences() -> bool:
    """Return whether sequential inputs are available in the current runtime."""
    return (
        (TRANSACTIONS_DIR.exists() and any(TRANSACTIONS_DIR.glob("*.csv")))
        or get_runtime_raw_seq_file().exists()
    )


def cleanup_ephemeral_sequence_artifacts() -> None:
    """Remove temporary sequence artifacts created for a single notebook run."""
    if os.environ.get("SEQCREDIT_EPHEMERAL") != "1":
        return

    for path in [
        get_runtime_lstm_cache_file(),
        get_runtime_raw_seq_file(),
        get_runtime_user_features_file(),
        get_runtime_user_labels_file(),
    ]:
        if path.exists():
            path.unlink()
            print(f"  Removed temporary artifact: {path}")
    for env_var in [
        "SEQCREDIT_EPHEMERAL",
        "SEQCREDIT_RAW_SEQ_FILE",
        "SEQCREDIT_LSTM_CACHE_FILE",
        "SEQCREDIT_USER_FEATURES_FILE",
        "SEQCREDIT_USER_LABELS_FILE",
    ]:
        os.environ.pop(env_var, None)


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute Brier score for probabilistic predictions."""
    return brier_score_loss(y_true, y_proba)


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error using histogram binning."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        in_bin = (y_proba > bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        count = in_bin.sum()
        if count > 0:
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_proba[in_bin].mean()
            ece += (count / total) * abs(bin_accuracy - bin_confidence)

    return ece


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for a single fold."""
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": (y_pred == y_true).mean(),
        "brier": compute_brier_score(y_true, y_proba),
        "ece": compute_ece(y_true, y_proba),
    }


def run_static_model_cv(
    model_class,
    model_params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
    """Run 5-fold CV for a static model and return fold results + aggregates + OOF predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        if model_class == LogisticRegressionModel:
            model = model_class(**model_params)
            model.fit(X_tr_scaled, y_tr)
            y_proba = model.predict_proba(X_val_scaled)
        elif model_class in (XGBoostModel, RandomForestModel, LightGBMModel):
            model = model_class(**model_params)
            model.fit(X_tr_scaled, y_tr)
            y_proba = model.predict_proba(X_val_scaled)
        else:
            raise ValueError(f"Unknown static model class: {model_class}")

        metrics = compute_metrics(y_val, y_proba)
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        oof_predictions[val_idx] = y_proba

    fold_df = pd.DataFrame(fold_results)
    summary = {col: fold_df[col].mean() for col in fold_df.columns if col != "fold"}

    std_cols = {}
    for col in ["auc_roc", "auc_pr", "f1"]:
        std_cols[f"{col}_std"] = fold_df[col].std()
    summary.update(std_cols)

    return fold_df, summary, oof_predictions


def run_lstm_cv(
    model_params: Dict,
    X_seq: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
    model_class=None,
) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
    """Run 5-fold CV for LSTM-compatible sequential model with OOF predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    oof_predictions = np.zeros(len(y))

    input_shape = (X_seq.shape[1], X_seq.shape[2])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        print(f"      Fold {fold + 1}/{n_splits}...")
        X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        _cls = model_class if model_class is not None else LSTMModel
        model = _cls(**model_params)
        model.build_model(input_shape)
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, epochs=50, batch_size=256)

        y_proba = model.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_proba)
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        oof_predictions[val_idx] = y_proba

        del model
        import tensorflow as tf

        tf.keras.backend.clear_session()

    fold_df = pd.DataFrame(fold_results)
    summary = {}
    for col in fold_df.columns:
        if col != "fold":
            summary[col] = fold_df[col].mean()
            if col in ["auc_roc", "auc_pr", "f1"]:
                summary[f"{col}_std"] = fold_df[col].std()

    return fold_df, summary, oof_predictions


def run_hybrid_cv(
    model_params: Dict,
    X_seq: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
    model_class=None,
) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
    """Run 5-fold CV for a hybrid sequential+static model with OOF predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    oof_predictions = np.zeros(len(y))

    seq_shape = (X_seq.shape[1], X_seq.shape[2])
    static_dim = X_static.shape[1]

    _cls = model_class if model_class is not None else HybridLSTMModel

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        print(f"      Fold {fold + 1}/{n_splits}...")
        X_seq_tr, X_seq_val = X_seq[train_idx], X_seq[val_idx]
        X_static_tr, X_static_val = X_static[train_idx], X_static[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_static_tr_scaled = scaler.fit_transform(X_static_tr)
        X_static_val_scaled = scaler.transform(X_static_val)

        model = _cls(**model_params)
        model.build_model(seq_shape, static_dim)
        model.fit(
            X_seq_tr,
            X_static_tr_scaled,
            y_tr,
            X_val_seq=X_seq_val,
            X_val_static=X_static_val_scaled,
            y_val=y_val,
            epochs=50,
            batch_size=256,
        )

        y_proba = model.predict_proba(X_seq_val, X_static_val_scaled)
        metrics = compute_metrics(y_val, y_proba)
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        oof_predictions[val_idx] = y_proba

        del model
        import tensorflow as tf

        tf.keras.backend.clear_session()

    fold_df = pd.DataFrame(fold_results)
    summary = {}
    for col in fold_df.columns:
        if col != "fold":
            summary[col] = fold_df[col].mean()
            if col in ["auc_roc", "auc_pr", "f1"]:
                summary[f"{col}_std"] = fold_df[col].std()

    return fold_df, summary, oof_predictions


def bootstrap_delta_test(
    y_true: np.ndarray,
    y_proba_1: np.ndarray,
    y_proba_2: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Bootstrap test for difference between two model predictions on same data."""
    np.random.seed(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p1 = y_proba_1[idx]
        y_p2 = y_proba_2[idx]

        if len(np.unique(y_t)) < 2:
            continue

        m1 = metric_fn(y_t, y_p1)
        m2 = metric_fn(y_t, y_p2)
        deltas.append(m1 - m2)

    deltas = np.array(deltas)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas)
    alpha = 1 - CI_LEVEL
    ci_lower = np.percentile(deltas, alpha / 2 * 100)
    ci_upper = np.percentile(deltas, (1 - alpha / 2) * 100)
    # Shift bootstrap distribution to null (delta=0) before computing p-value.
    # Without centering, p_value ≈ 0.5 always (mean is always near median of
    # its own distribution — a tautology, not a test).
    deltas_centered = deltas - delta_mean
    p_value = (np.abs(deltas_centered) >= np.abs(delta_mean)).mean()

    return {
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def run_significance_tests(
    y_true: np.ndarray,
    oof_preds: Dict[str, np.ndarray],
    model_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Run pairwise bootstrap significance tests between models using OOF predictions."""
    comparisons = []

    for model_1, model_2 in model_pairs:
        if model_1 not in oof_preds or model_2 not in oof_preds:
            continue

        y_proba_1 = oof_preds[model_1]
        y_proba_2 = oof_preds[model_2]

        delta_roc = bootstrap_delta_test(y_true, y_proba_1, y_proba_2, roc_auc_score)
        delta_pr = bootstrap_delta_test(
            y_true, y_proba_1, y_proba_2, average_precision_score
        )

        comparisons.append(
            {
                "comparison": f"{model_1} vs {model_2}",
                "metric": "AUC-ROC",
                "delta_mean": delta_roc["delta_mean"],
                "p_value": delta_roc["p_value"],
                "ci_lower": delta_roc["ci_lower"],
                "ci_upper": delta_roc["ci_upper"],
                "significant": delta_roc["p_value"] < 0.05,
            }
        )
        comparisons.append(
            {
                "comparison": f"{model_1} vs {model_2}",
                "metric": "AUC-PR",
                "delta_mean": delta_pr["delta_mean"],
                "p_value": delta_pr["p_value"],
                "ci_lower": delta_pr["ci_lower"],
                "ci_upper": delta_pr["ci_upper"],
                "significant": delta_pr["p_value"] < 0.05,
            }
        )

    return pd.DataFrame(comparisons)


def main():
    print("=" * 60)
    print("5-Fold Stratified CV Benchmark")
    print("=" * 60)
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  CV folds: {N_SPLITS}")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")

    set_random_seeds(RANDOM_SEED)
    has_seq_inputs = has_sequences()

    print("\n[1/6] Loading data...")
    loader = CreditRiskDataLoader()
    static_data = loader.prepare_static_splits()
    X = static_data["X_train_scaled"]
    X_unscaled = static_data["X_train"]
    y_default = static_data["y_train"]
    feature_names = static_data["feature_names"]

    # Load labels and create y_bad target
    df_labels = pd.read_csv(get_runtime_user_labels_file())
    df_labels = df_labels[df_labels["credit_risk_label"] != -1]
    df_labels["y_bad"] = df_labels["credit_risk_label"].isin([1, 2]).astype(int)

    # Use ordered train user IDs from the loader (matches X_train row order)
    train_user_ids_ordered = loader._train_user_ids_ordered

    # Build y_bad aligned with X_train row order
    uid_to_ybad = dict(zip(df_labels["user_id"], df_labels["y_bad"]))
    y_bad = np.array([uid_to_ybad[uid] for uid in train_user_ids_ordered])

    print(f"  Train users: {len(train_user_ids_ordered)}")
    print(f"  y_default positive rate: {y_default.mean():.4f}")
    print(f"  y_bad positive rate: {y_bad.mean():.4f}")

    # Validate alignment
    assert len(y_default) == len(y_bad), "y_default and y_bad length mismatch!"
    assert len(X) == len(y_bad), "X and y_bad length mismatch!"

    if has_seq_inputs:
        print("\n[2/6] Loading LSTM sequences...")
        seq_data = loader.load_sequences()
        X_seq = seq_data["X_train_seq"]
        y_seq_default = seq_data["y_train"]
        y_seq_bad = y_bad.copy()
        print(f"  Sequence shape: {X_seq.shape}")
        assert len(X_seq) == len(y_seq_bad), "X_seq and y_seq_bad length mismatch!"
        assert np.array_equal(y_default, y_seq_default), (
            "y_default and y_seq_default mismatch!"
        )
    else:
        print("\n[2/6] Skipping LSTM sequences (no per-user transaction files found).")
        X_seq = y_seq_default = y_seq_bad = None

    print("\n[3/6] Running CV for all models...")

    static_model_params = {
        "LogisticRegression": {
            "class_weight": "balanced",
            "C": 1.0,
            "random_state": RANDOM_SEED,
        },
        "XGBoost": {
            "scale_pos_weight": loader.get_scale_pos_weight(),
            "random_state": RANDOM_SEED,
        },
        "RandomForest": {"class_weight": "balanced", "random_state": RANDOM_SEED},
        "LightGBM": {"class_weight": "balanced", "random_state": RANDOM_SEED},
    }

    lstm_params = {"learning_rate": 0.001}

    all_results = []
    oof_preds_default = {}
    oof_preds_bad = {}
    model_timings = {}

    # Prepare static features for hybrid model (unscaled — per-fold scaler inside run_hybrid_cv)
    static_X = static_data["X_train"]

    total_start = time.time()

    for target_name, y, X_seq_curr in [
        ("y_default", y_default, X_seq if has_seq_inputs else None),
        ("y_bad",     y_bad,     X_seq if has_seq_inputs else None),
    ]:
        print(f"\n  === Target: {target_name} ===")

        # --- Static models ---
        for model_name, model_class in [
            ("LogisticRegression", LogisticRegressionModel),
            ("XGBoost", XGBoostModel),
            ("RandomForest", RandomForestModel),
            ("LightGBM", LightGBMModel),
        ]:
            model_start = time.time()
            print(f"    {model_name}...", end=" ", flush=True)
            params = static_model_params[model_name]
            fold_df, summary, oof = run_static_model_cv(
                model_class, params, X_unscaled.values, y
            )
            elapsed = time.time() - model_start
            model_timings[f"{model_name}_{target_name}"] = elapsed
            print(f"done ({elapsed:.1f}s, AUC-ROC={summary['auc_roc']:.4f})")

            for _, row in fold_df.iterrows():
                all_results.append(
                    {
                        "model": model_name,
                        "target": target_name,
                        "fold": row["fold"],
                        **{
                            k: v
                            for k, v in row.items()
                            if k not in ["model", "target", "fold"]
                        },
                    }
                )

            # Store OOF predictions with consistent keys (just model name)
            if target_name == "y_default":
                oof_preds_default[model_name] = oof
            else:
                oof_preds_bad[model_name] = oof

        # --- LSTM + HybridLSTM (skipped when no per-user transaction files) ---
        if has_seq_inputs:
            model_start = time.time()
            print(f"    LSTM...")
            fold_df, summary, oof = run_lstm_cv(lstm_params, X_seq_curr, y)
            elapsed = time.time() - model_start
            model_timings[f"LSTM_{target_name}"] = elapsed
            print(f"      done ({elapsed:.1f}s, AUC-ROC={summary['auc_roc']:.4f})")

            for _, row in fold_df.iterrows():
                all_results.append(
                    {
                        "model": "LSTM",
                        "target": target_name,
                        "fold": row["fold"],
                        **{k: v for k, v in row.items() if k not in ["model", "target", "fold"]},
                    }
                )

            if target_name == "y_default":
                oof_preds_default["LSTM"] = oof
            else:
                oof_preds_bad["LSTM"] = oof

            model_start = time.time()
            print(f"    HybridLSTM...")
            fold_df, summary, oof = run_hybrid_cv(
                lstm_params, X_seq_curr, static_X.values, y
            )
            elapsed = time.time() - model_start
            model_timings[f"HybridLSTM_{target_name}"] = elapsed
            print(f"      done ({elapsed:.1f}s, AUC-ROC={summary['auc_roc']:.4f})")

            for _, row in fold_df.iterrows():
                all_results.append(
                    {
                        "model": "HybridLSTM",
                        "target": target_name,
                        "fold": row["fold"],
                        **{k: v for k, v in row.items() if k not in ["model", "target", "fold"]},
                    }
                )

            if target_name == "y_default":
                oof_preds_default["HybridLSTM"] = oof
            else:
                oof_preds_bad["HybridLSTM"] = oof

            model_start = time.time()
            print(f"    GRU...")
            fold_df, summary, oof = run_lstm_cv(
                lstm_params, X_seq_curr, y, model_class=GRUModel
            )
            elapsed = time.time() - model_start
            model_timings[f"GRU_{target_name}"] = elapsed
            print(f"      done ({elapsed:.1f}s, AUC-ROC={summary['auc_roc']:.4f})")

            for _, row in fold_df.iterrows():
                all_results.append(
                    {
                        "model": "GRU",
                        "target": target_name,
                        "fold": row["fold"],
                        **{k: v for k, v in row.items() if k not in ["model", "target", "fold"]},
                    }
                )

            if target_name == "y_default":
                oof_preds_default["GRU"] = oof
            else:
                oof_preds_bad["GRU"] = oof

            model_start = time.time()
            print(f"    HybridGRU...")
            fold_df, summary, oof = run_hybrid_cv(
                lstm_params, X_seq_curr, static_X.values, y, model_class=HybridGRUModel
            )
            elapsed = time.time() - model_start
            model_timings[f"HybridGRU_{target_name}"] = elapsed
            print(f"      done ({elapsed:.1f}s, AUC-ROC={summary['auc_roc']:.4f})")

            for _, row in fold_df.iterrows():
                all_results.append(
                    {
                        "model": "HybridGRU",
                        "target": target_name,
                        "fold": row["fold"],
                        **{k: v for k, v in row.items() if k not in ["model", "target", "fold"]},
                    }
                )

            if target_name == "y_default":
                oof_preds_default["HybridGRU"] = oof
            else:
                oof_preds_bad["HybridGRU"] = oof
        else:
            print("    LSTM / HybridLSTM / GRU / HybridGRU — skipped (no sequence files)")

        # Intermediate checkpoint — always save so progress survives a kill
        _runtime_dir = get_runtime_data_dir()
        _runtime_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(
            _runtime_dir / f"cv_results_intermediate_{target_name}.csv", index=False
        )
        print(f"    [Checkpoint] Saved intermediate results for {target_name}")

    total_elapsed = time.time() - total_start
    print(f"\n  Total CV time: {total_elapsed / 60:.1f} minutes")

    results_df = pd.DataFrame(all_results)

    runtime_dir = get_runtime_data_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    print("\n[4/6] Saving CV results...")
    for target_name in ["y_default", "y_bad"]:
        target_df = results_df[results_df["target"] == target_name].copy()
        target_df.to_csv(runtime_dir / f"cv_results_{target_name}.csv", index=False)
    print("  Saved cv_results_y_default.csv")
    print("  Saved cv_results_y_bad.csv")

    # Save OOF predictions for downstream ROC curve generation
    oof_arrays = {"y_default": y_default, "y_bad": y_bad}
    for model_name, oof in oof_preds_default.items():
        oof_arrays[f"default_{model_name}"] = oof
    for model_name, oof in oof_preds_bad.items():
        oof_arrays[f"bad_{model_name}"] = oof
    np.savez(runtime_dir / "cv_oof_preds.npz", **oof_arrays)
    print("  Saved cv_oof_preds.npz")

    print("\n[5/6] Running significance tests...")

    static_models = ["LogisticRegression", "XGBoost", "RandomForest", "LightGBM"]
    seq_models = ["LSTM", "HybridLSTM", "GRU", "HybridGRU"] if has_seq_inputs else []
    all_models = static_models + seq_models

    best_static = None
    best_static_auc = -1
    for m in static_models:
        row = results_df[
            (results_df["model"] == m) & (results_df["target"] == "y_default")
        ]
        if not row.empty:
            auc = row["auc_roc"].mean()
            if auc > best_static_auc:
                best_static = m
                best_static_auc = auc

    if seq_models:
        # Compare each sequence model against the best static model and all others
        model_pairs = []
        for seq_m in seq_models:
            if best_static:
                model_pairs.append((seq_m, best_static))
            for static_m in static_models:
                model_pairs.append((seq_m, static_m))
    else:
        # No sequence models available — compare static models pairwise instead
        from itertools import combinations
        model_pairs = list(combinations(static_models, 2))

    sig_results = []

    for target_name, y, oof_dict in [
        ("y_default", y_default, oof_preds_default),
        ("y_bad", y_bad, oof_preds_bad),
    ]:
        print(f"  === {target_name} ===")
        oof_preds = {m: oof_dict.get(m) for m in all_models}
        valid_oof = {
            k: v for k, v in oof_preds.items() if v is not None and len(v) == len(y)
        }

        if len(valid_oof) >= 2:
            sig_df = run_significance_tests(y, valid_oof, model_pairs)
            sig_df["target"] = target_name
            sig_results.append(sig_df)

    if sig_results:
        sig_results_df = pd.concat(sig_results, ignore_index=True)
        sig_results_df.to_csv(runtime_dir / "significance_tests.csv", index=False)
        print("  Saved significance_tests.csv")

        print("\n  Significant comparisons (p < 0.05):")
        if "significant" in sig_results_df.columns:
            sig_only = sig_results_df[sig_results_df["significant"] == True]
            print(
                sig_only[
                    ["comparison", "target", "metric", "delta_mean", "p_value"]
                ].to_string()
            )
        else:
            print("  (no comparisons available)")
    else:
        print("  No valid OOF predictions available for significance tests")

    print("\n[6/6] Finalizing run...")

    manifest = {
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "models": all_models,
        "targets": ["y_default", "y_bad"],
        "timings_seconds": model_timings,
        "total_time_minutes": total_elapsed / 60,
    }

    with open(runtime_dir / "cv_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("  Saved cv_manifest.json")

    cleanup_ephemeral_sequence_artifacts()

    print("\n" + "=" * 60)
    print("CV Benchmark Complete")
    print("=" * 60)

    print("\n=== Summary (y_default) ===")
    default_summary = (
        results_df[results_df["target"] == "y_default"]
        .groupby("model")[["auc_roc", "auc_pr", "f1", "brier", "ece"]]
        .mean()
    )
    print(default_summary.round(4).to_string())

    print("\n=== Summary (y_bad) ===")
    bad_summary = (
        results_df[results_df["target"] == "y_bad"]
        .groupby("model")[["auc_roc", "auc_pr", "f1", "brier", "ece"]]
        .mean()
    )
    print(bad_summary.round(4).to_string())

    return {
        "results_df": results_df,
        "significance_tests": sig_results_df if sig_results else None,
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
