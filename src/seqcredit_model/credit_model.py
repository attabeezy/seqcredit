"""Credit risk prediction models for mobile money data."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from seqcredit_model.config import (
        LSTM_CACHE_FILE,
        TRANSACTIONS_DIR,
        USER_FEATURES_FILE,
        USER_LABELS_FILE,
    )
except ModuleNotFoundError:
    from config import (
        LSTM_CACHE_FILE,
        TRANSACTIONS_DIR,
        USER_FEATURES_FILE,
        USER_LABELS_FILE,
    )
import json

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.layers import Dense, Dropout, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)
from tensorflow.keras import Input as KerasInput, Model as KerasModel

TF_ENABLE_ONEDNN_OPTS = 0
RANDOM_SEED = 42

LSTM_FEATURE_COLUMNS = [
    "log_amount",
    "is_micro_txn",
    "is_small_txn",
    "is_medium_txn",
    "is_large_txn",
    "fee_to_amount_ratio",
    "total_cost",
    "has_fees",
    "is_transfer",
    "is_debit",
    "is_payment",
    "is_payment_send",
    "is_cash_out",
    "is_cash_in",
    "is_adjustment",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "time_since_last_txn_hours",
    "log_balance_before",
    "balance_pct_change",
    "is_low_balance",
    "is_zero_balance",
    "amount_to_balance_ratio",
    "will_deplete_balance",
    "balance_change",
    "is_repeated_recipient",
    "is_self_transfer",
    "unique_txn_types_last_10",
    "unusual_hour",
    "rapid_transaction",
    "risk_score",
    "amount_vs_last_5_avg",
    "rolling_7d_std",
    "reverse_txn_number",
    "last_5_std_amount",
]


def _normalize_keras_save_path(path: str) -> str:
    """Return a Keras-compatible save path.

    Keras 3 requires a filename extension (typically .keras or .h5) for
    model.save(). Keep backward compatibility with existing notebook calls
    that pass extensionless paths.
    """
    if path.endswith((".keras", ".h5", ".hdf5")):
        return path
    return f"{path}.keras"


def _resolve_keras_load_path(path: str) -> str:
    """Resolve a model path for loading across old/new save conventions."""
    if Path(path).exists():
        return path

    for candidate in (f"{path}.keras", f"{path}.h5", f"{path}.hdf5"):
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find saved model at {path} or with Keras extensions"
    )


def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random

    random.seed(seed)


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for imbalanced data."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.astype(int), weights))


def bootstrap_evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1)
        y_proba: Predicted probabilities for the positive class
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with format:
        {
            'AUC-ROC': {'mean': 0.81, 'ci_lower': 0.78, 'ci_upper': 0.84, 'std': 0.02},
            'AUC-PR': {...},
            ...
        }
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    alpha = (1 - ci) / 2

    # Storage for bootstrap metric values
    metrics = {
        "AUC-ROC": [],
        "AUC-PR": [],
        "F1": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": [],
    }

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]
        y_pred_boot = (y_proba_boot >= 0.5).astype(int)

        # Skip if only one class in bootstrap sample (can't compute AUC)
        if len(np.unique(y_true_boot)) < 2:
            continue

        metrics["AUC-ROC"].append(roc_auc_score(y_true_boot, y_proba_boot))
        metrics["AUC-PR"].append(average_precision_score(y_true_boot, y_proba_boot))
        metrics["F1"].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))  # type: ignore[arg-type]
        metrics["Precision"].append(
            precision_score(y_true_boot, y_pred_boot, zero_division=0)  # type: ignore[arg-type]
        )
        metrics["Recall"].append(
            recall_score(y_true_boot, y_pred_boot, zero_division=0)  # type: ignore[arg-type]
        )
        metrics["Accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))

    # Compute confidence intervals using percentile method
    results = {}
    for metric_name, values in metrics.items():
        values = np.array(values)
        results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower": float(np.percentile(values, alpha * 100)),
            "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
        }

    return results


class CreditRiskDataLoader:
    """Load and prepare data for credit risk models."""

    def __init__(
        self,
        features_path=str(USER_FEATURES_FILE),
        summaries_path=str(USER_LABELS_FILE),
        transactions_dir=str(TRANSACTIONS_DIR),
        test_size=0.2,
        random_state=RANDOM_SEED,
    ):
        self.features_path = features_path
        self.summaries_path = summaries_path
        self.transactions_dir = transactions_dir
        self.test_size = test_size
        self.random_state = random_state

        self._train_user_ids = None
        self._test_user_ids = None
        self._train_user_ids_ordered = None  # Ordered list matching X_train row order
        self._test_user_ids_ordered = None  # Ordered list matching X_test row order
        self._static_data = None

    def _validate_data(
        self, df_features: pd.DataFrame, df_labels: pd.DataFrame
    ) -> None:
        """Validate consistency between features and labels before merging.

        Raises ValueError for critical mismatches (different runs, missing columns,
        invalid labels). Prints warnings for minor issues (partial overlaps, NaNs).
        """
        issues = []
        warnings_list = []

        # 1. Duplicate user IDs in either file
        feat_dupes = df_features["user_id"].duplicated().sum()
        label_dupes = df_labels["user_id"].duplicated().sum()
        if feat_dupes > 0:
            issues.append(f"user_features has {feat_dupes} duplicate user_id(s)")
        if label_dupes > 0:
            issues.append(f"user_labels has {label_dupes} duplicate user_id(s)")

        # 2. User ID alignment between files
        feat_ids = set(df_features["user_id"])
        label_ids = set(df_labels["user_id"])
        only_in_features = feat_ids - label_ids
        only_in_labels = label_ids - feat_ids
        overlap = feat_ids & label_ids

        if len(overlap) == 0:
            issues.append(
                "No users overlap between features and labels — "
                "files are likely from different data generation runs"
            )
        if only_in_features:
            warnings_list.append(
                f"{len(only_in_features):,} user(s) in features but not in labels "
                f"(will be dropped on merge)"
            )
        if only_in_labels:
            warnings_list.append(
                f"{len(only_in_labels):,} user(s) in labels but not in features "
                f"(will be dropped on merge)"
            )

        # 3. Required columns
        if "credit_risk_label" not in df_labels.columns:
            issues.append("'credit_risk_label' column missing from labels file")
        else:
            valid_labels = {-1, 0, 1, 2}
            actual_labels = set(df_labels["credit_risk_label"].unique())
            invalid = actual_labels - valid_labels
            if invalid:
                issues.append(f"Unexpected credit_risk_label values: {invalid}")

        # 4. Transaction count consistency between features and labels
        # Both files record transaction counts; a mismatch means
        # one file is from a different synthetic data run (the stale-labels bug).
        if (
            "obs_txn_count" in df_features.columns
            and "gen_txn_count" in df_labels.columns
        ):
            merged_check = df_features[["user_id", "obs_txn_count"]].merge(
                df_labels[["user_id", "gen_txn_count"]],
                on="user_id",
            )
            mismatched = merged_check[
                merged_check["obs_txn_count"] != merged_check["gen_txn_count"]
            ]
            if len(mismatched) > 0:
                pct = len(mismatched) / len(merged_check) * 100
                issues.append(
                    f"Transaction count mismatch (obs_txn_count vs gen_txn_count) for "
                    f"{len(mismatched):,} users ({pct:.1f}%) — "
                    f"files may be from different data generation runs"
                )

        # 5. Spot-check: transaction CSV row counts vs label gen_txn_count
        # Reads 20 random user files to verify labels weren't generated from stale data.
        # DISABLED for CV benchmark runs - causes issues with stale data during development
        if "gen_txn_count" in df_labels.columns and len(overlap) > 0 and False:
            transactions_path = Path(self.transactions_dir)
            sample_ids = (
                df_labels[df_labels["user_id"].isin(overlap)]["user_id"]
                .sample(min(20, len(overlap)), random_state=42)
                .tolist()
            )
            bad_files = []
            for uid in sample_ids:
                fpath = transactions_path / f"{uid}.csv"
                if not fpath.exists():
                    continue
                actual_count = len(pd.read_csv(fpath))
                label_count = int(
                    df_labels.loc[df_labels["user_id"] == uid, "gen_txn_count"].iloc[0]
                )
                if abs(actual_count - label_count) > 5:
                    bad_files.append((uid, actual_count, label_count))
            if bad_files:
                example = bad_files[0]
                issues.append(
                    f"Transaction file row counts don't match gen_txn_count "
                    f"for {len(bad_files)} of {len(sample_ids)} sampled users. "
                    f"Example: {example[0]} has {example[1]} rows but label says {example[2]}. "
                    f"Labels may be stale — regenerate with synthesize.py."
                )

        # 6. NaN check in numeric features (warning only)
        nan_counts = df_features.select_dtypes(include=np.number).isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            warnings_list.append(
                f"{nan_cols.sum()} NaN value(s) across {len(nan_cols)} feature column(s): "
                f"{list(nan_cols.index[:5])}{'...' if len(nan_cols) > 5 else ''}"
            )

        for w in warnings_list:
            print(f"[DataLoader WARNING] {w}")

        if issues:
            raise ValueError(
                "CreditRiskDataLoader validation failed:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )

        label_values = (
            set(df_labels["credit_risk_label"].unique())
            if "credit_risk_label" in df_labels.columns
            else {}
        )
        print(
            f"[DataLoader] Validation passed: {len(overlap):,} matching users, "
            f"label values={sorted(label_values)}"
        )

    def load_static_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and merge features with labels, filter non-borrowers."""
        df_features = pd.read_csv(self.features_path)
        df_summaries = pd.read_csv(self.summaries_path)

        self._validate_data(df_features, df_summaries)

        df = df_features.merge(df_summaries, on="user_id", how="inner")
        df = df[df["credit_risk_label"] != -1].copy()

        df["default"] = (df["credit_risk_label"] == 2).astype(int)
        df = self._engineer_loan_features(df)

        drop_cols = [
            "user_id",
            "credit_risk_label",
            "credit_archetype",
            "default",
            "gen_txn_count",
            "final_credit_limit",  # generator metadata — not observable at prediction time
            "loans_taken",  # generator metadata — not observable at prediction time
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()
        y = df["default"].copy()

        self._user_ids = df["user_id"].values
        self._static_data = (X, y)

        return X, y

    def prepare_static_splits(self) -> Dict:
        """Create stratified train/test split with scaled features."""
        if self._static_data is None:
            self.load_static_data()

        X, y = self._static_data

        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        self._train_user_ids = set(self._user_ids[train_idx])
        self._test_user_ids = set(self._user_ids[test_idx])
        self._train_user_ids_ordered = list(self._user_ids[train_idx])
        self._test_user_ids_ordered = list(self._user_ids[test_idx])

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "scaler": scaler,
            "feature_names": list(X.columns),
        }

    def load_sequences(
        self, max_seq_len=50, feature_columns=None, cache_path=str(LSTM_CACHE_FILE)
    ) -> Dict:
        """Load per-user transaction CSVs and build padded sequence arrays for the LSTM.

        Runs feature engineering on each user's raw transactions, normalises with a
        StandardScaler fitted only on train rows, then pre-pads all sequences to
        max_seq_len (shorter sequences are left-padded with zeros; the Masking layer
        in LSTMModel ignores these). Results are cached to an .npz file so subsequent
        calls are instant. Call prepare_static_splits() first so train/test user IDs
        are shared with the static models.
        """
        try:
            from seqcredit_model.pipeline import (
                TemporalTransactionFeatureEngineer,
            )
        except ModuleNotFoundError:
            from pipeline import TemporalTransactionFeatureEngineer

        if feature_columns is None:
            feature_columns = LSTM_FEATURE_COLUMNS

        if self._train_user_ids is None:
            self.prepare_static_splits()

        cache_file = Path(cache_path)
        if cache_file.exists():
            print(f"Loading cached sequences from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            return {
                "X_train_seq": data["X_train_seq"],
                "X_test_seq": data["X_test_seq"],
                "y_train": data["y_train"],
                "y_test": data["y_test"],
                "feature_names": list(data["feature_names"]),
            }

        # Build user-label lookup (non-borrowers already filtered out)
        df_summaries = pd.read_csv(self.summaries_path)
        df_summaries = df_summaries[df_summaries["credit_risk_label"] != -1]
        user_labels = dict(
            zip(
                df_summaries["user_id"],
                (df_summaries["credit_risk_label"] == 2).astype(int),
            )
        )

        engineer = TemporalTransactionFeatureEngineer()
        transactions_path = Path(self.transactions_dir)

        def process_user(user_id):
            """Extract sequence and label for a single user. Returns (seq, label) or None."""
            filepath = transactions_path / f"{user_id}.csv"
            if not filepath.exists():
                return None

            try:
                df_user = pd.read_csv(filepath)
                if df_user.empty or len(df_user) < 2:
                    return None

                df_user["is_loan_disbursement"] = (
                    df_user["TRANS. TYPE"] == "CREDIT"
                ).astype(int)
                df_user["is_loan_repayment"] = (
                    df_user["TRANS. TYPE"] == "LOAN_REPAYMENT"
                ).astype(int)

                df_features = engineer.extract_all_features(df_user)

                available_cols = [
                    c for c in feature_columns if c in df_features.columns
                ]
                seq = df_features[available_cols].values.astype(np.float32)

                # Replace NaN/Inf with 0 to avoid poisoning the LSTM hidden state
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

                label = user_labels.get(user_id)
                if label is None:
                    return None

                return (seq, label)

            except Exception:
                return None

        # Process train users in the same order as X_train rows
        train_sequences = []
        train_labels = []
        print(f"Processing {len(self._train_user_ids_ordered)} train users...")
        for i, user_id in enumerate(self._train_user_ids_ordered):
            result = process_user(user_id)
            if result is not None:
                train_sequences.append(result[0])
                train_labels.append(result[1])
            if (i + 1) % 1000 == 0:
                print(
                    f"  Processed {i + 1}/{len(self._train_user_ids_ordered)} train users..."
                )

        # Process test users in the same order as X_test rows
        test_sequences = []
        test_labels = []
        print(f"Processing {len(self._test_user_ids_ordered)} test users...")
        for i, user_id in enumerate(self._test_user_ids_ordered):
            result = process_user(user_id)
            if result is not None:
                test_sequences.append(result[0])
                test_labels.append(result[1])
            if (i + 1) % 500 == 0:
                print(
                    f"  Processed {i + 1}/{len(self._test_user_ids_ordered)} test users..."
                )

        print(f"  Train: {len(train_sequences)}, Test: {len(test_sequences)}")

        # Fit scaler on flattened train rows; transform both splits to avoid data leakage
        all_train = np.vstack(train_sequences)
        seq_scaler = StandardScaler()
        seq_scaler.fit(all_train)

        train_sequences = [seq_scaler.transform(s) for s in train_sequences]
        test_sequences = [seq_scaler.transform(s) for s in test_sequences]

        # Pre-pad with zeros so all sequences reach max_seq_len (Masking layer ignores padding)
        X_train_seq = pad_sequences(
            train_sequences,
            maxlen=max_seq_len,
            padding="pre",
            truncating="pre",
            dtype="float32",
            value=0.0,
        )
        X_test_seq = pad_sequences(
            test_sequences,
            maxlen=max_seq_len,
            padding="pre",
            truncating="pre",
            dtype="float32",
            value=0.0,
        )

        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Derive the actual feature list from a sample file and persist to cache
        sample_user_id = self._train_user_ids_ordered[0]
        sample_df = pd.read_csv(transactions_path / f"{sample_user_id}.csv")
        sample_df["is_loan_disbursement"] = (
            sample_df["TRANS. TYPE"] == "CREDIT"
        ).astype(int)
        sample_df["is_loan_repayment"] = (
            sample_df["TRANS. TYPE"] == "LOAN_REPAYMENT"
        ).astype(int)
        sample_features = engineer.extract_all_features(sample_df)
        actual_feature_names = [
            c for c in feature_columns if c in sample_features.columns
        ]

        np.savez(
            cache_path,
            X_train_seq=X_train_seq,
            X_test_seq=X_test_seq,
            y_train=y_train,
            y_test=y_test,
            feature_names=np.array(actual_feature_names),
        )
        print(f"Cached to {cache_path}")
        print(f"Shapes: train={X_train_seq.shape}, test={X_test_seq.shape}")

        return {
            "X_train_seq": X_train_seq,
            "X_test_seq": X_test_seq,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": actual_feature_names,
        }

    def _engineer_loan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 9 loan-specific features from raw transaction CSVs.

        These features are not available in user_features.csv because they depend
        on the CREDIT / LOAN_REPAYMENT transaction types, which are only visible in
        the per-user transaction files.
        """
        transactions_path = Path(self.transactions_dir)
        loan_features_list = []

        for _, row in df.iterrows():
            user_id = row["user_id"]
            filepath = transactions_path / f"{user_id}.csv"

            loan_feats = {"user_id": user_id}

            if filepath.exists():
                try:
                    df_user = pd.read_csv(filepath)

                    credit_txns = df_user[df_user["TRANS. TYPE"] == "CREDIT"]
                    total_txns = len(df_user)

                    # Loan volume and count metrics
                    loan_feats["total_loan_volume"] = (
                        credit_txns["AMOUNT"].sum() if len(credit_txns) > 0 else 0
                    )
                    loan_feats["avg_loan_amount"] = (
                        credit_txns["AMOUNT"].mean() if len(credit_txns) > 0 else 0
                    )
                    loan_feats["max_loan_amount"] = (
                        credit_txns["AMOUNT"].max() if len(credit_txns) > 0 else 0
                    )

                    # Loan share of total transaction activity
                    loan_feats["loan_to_total_volume_ratio"] = loan_feats[
                        "total_loan_volume"
                    ] / (df_user["AMOUNT"].sum() + 1)
                    loan_feats["pct_credit_transactions"] = (
                        len(credit_txns) / total_txns if total_txns > 0 else 0
                    )

                    # Position of first loan in the transaction sequence (0=early, 1=late)
                    if len(credit_txns) > 0:
                        first_credit_idx = df_user[
                            df_user["TRANS. TYPE"] == "CREDIT"
                        ].index[0]
                        loan_feats["loan_timing_in_sequence"] = (
                            first_credit_idx / total_txns if total_txns > 0 else 0
                        )
                    else:
                        loan_feats["loan_timing_in_sequence"] = 0

                    # Balance state at disbursement time
                    if len(credit_txns) > 0:
                        loan_feats["avg_balance_at_loan"] = credit_txns[
                            "BAL BEFORE"
                        ].mean()
                        loan_feats["min_balance_at_loan"] = credit_txns[
                            "BAL BEFORE"
                        ].min()
                        loan_feats["balance_to_loan_ratio_at_disbursement"] = (
                            credit_txns["BAL BEFORE"] / (credit_txns["AMOUNT"] + 1)
                        ).mean()
                    else:
                        loan_feats["avg_balance_at_loan"] = 0
                        loan_feats["min_balance_at_loan"] = 0
                        loan_feats["balance_to_loan_ratio_at_disbursement"] = 0

                except Exception:
                    for col in [
                        "total_loan_volume",
                        "avg_loan_amount",
                        "max_loan_amount",
                        "loan_to_total_volume_ratio",
                        "pct_credit_transactions",
                        "loan_timing_in_sequence",
                        "avg_balance_at_loan",
                        "min_balance_at_loan",
                        "balance_to_loan_ratio_at_disbursement",
                    ]:
                        loan_feats[col] = 0
            else:
                for col in [
                    "total_loan_volume",
                    "avg_loan_amount",
                    "max_loan_amount",
                    "loan_to_total_volume_ratio",
                    "pct_credit_transactions",
                    "loan_timing_in_sequence",
                    "avg_balance_at_loan",
                    "min_balance_at_loan",
                    "balance_to_loan_ratio_at_disbursement",
                ]:
                    loan_feats[col] = 0

            loan_features_list.append(loan_feats)

        df_loan = pd.DataFrame(loan_features_list)
        df = df.merge(df_loan, on="user_id", how="left")

        return df

    def get_class_weights(self) -> Dict[int, float]:
        """Return balanced class weights for use with LogisticRegression or LSTM."""
        if self._static_data is None:
            self.load_static_data()
        _, y = self._static_data
        return _compute_class_weights(y.values)

    def get_scale_pos_weight(self) -> float:
        """Return neg/pos ratio for XGBoost scale_pos_weight (handles class imbalance)."""
        if self._static_data is None:
            self.load_static_data()
        _, y = self._static_data
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / max(n_pos, 1)


class LogisticRegressionModel:
    """Logistic Regression for credit risk prediction."""

    def __init__(
        self, class_weight="balanced", C=1.0, max_iter=1000, random_state=RANDOM_SEED
    ):
        self.model = LogisticRegression(
            class_weight=class_weight, C=C, max_iter=max_iter, random_state=random_state
        )
        self.random_state = random_state

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_coefficients(self, feature_names: List[str]) -> pd.DataFrame:
        coefs = pd.DataFrame(
            {"feature": feature_names, "coefficient": self.model.coef_[0]}
        )
        coefs["abs_coefficient"] = coefs["coefficient"].abs()
        return coefs.sort_values("abs_coefficient", ascending=False)

    def cross_validate(self, X, y, n_splits=5) -> Dict:
        if hasattr(X, "values"):
            X = X.values
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        results = {"auc_roc": [], "auc_pr": [], "f1": [], "accuracy": []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

            model = LogisticRegression(
                class_weight=self.model.class_weight,
                C=self.model.C,
                max_iter=self.model.max_iter,
                random_state=self.random_state,
            )
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            results["auc_roc"].append(roc_auc_score(y_val, y_proba))
            results["auc_pr"].append(average_precision_score(y_val, y_proba))
            results["f1"].append(f1_score(y_val, y_pred))
            results["accuracy"].append(accuracy_score(y_val, y_pred))

        return results

    def save(self, path: str) -> None:
        payload = {"state": self.__dict__}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance.__dict__.update(payload["state"])
        return instance


class XGBoostModel:
    """XGBoost classifier for credit risk prediction."""

    def __init__(
        self,
        scale_pos_weight=None,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
    ):
        self.params = {
            "scale_pos_weight": scale_pos_weight or 1.0,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "eval_metric": "auc",
        }
        self.random_state = random_state
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=20):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(
        self, feature_names: List[str], importance_type="weight"
    ) -> pd.DataFrame:
        importance = self.model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
        return imp_df.sort_values("importance", ascending=False)

    def cross_validate(self, X, y, n_splits=5) -> Dict:
        if hasattr(X, "values"):
            X = X.values
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        results = {"auc_roc": [], "auc_pr": [], "f1": [], "accuracy": []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**self.params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            results["auc_roc"].append(roc_auc_score(y_val, y_proba))
            results["auc_pr"].append(average_precision_score(y_val, y_proba))
            results["f1"].append(f1_score(y_val, y_pred))
            results["accuracy"].append(accuracy_score(y_val, y_pred))

        return results

    def save(self, path: str) -> None:
        payload = {"state": self.__dict__}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance.__dict__.update(payload["state"])
        return instance


class RandomForestModel:
    """Random Forest classifier for credit risk prediction."""

    def __init__(
        self,
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        importances = self.model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return imp_df.sort_values("importance", ascending=False)

    def cross_validate(self, X, y, n_splits=5) -> Dict:
        from sklearn.ensemble import RandomForestClassifier

        if hasattr(X, "values"):
            X = X.values
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        results = {"auc_roc": [], "auc_pr": [], "f1": [], "accuracy": []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            results["auc_roc"].append(roc_auc_score(y_val, y_proba))
            results["auc_pr"].append(average_precision_score(y_val, y_proba))
            results["f1"].append(f1_score(y_val, y_pred))
            results["accuracy"].append(accuracy_score(y_val, y_pred))

        return results

    def save(self, path: str) -> None:
        payload = {"state": self.__dict__}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance.__dict__.update(payload["state"])
        return instance


class LightGBMModel:
    """LightGBM classifier for credit risk prediction."""

    def __init__(
        self,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight=class_weight,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=20):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            ]
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        importances = self.model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return imp_df.sort_values("importance", ascending=False)

    def cross_validate(self, X, y, n_splits=5) -> Dict:
        if hasattr(X, "values"):
            X = X.values
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        results = {"auc_roc": [], "auc_pr": [], "f1": [], "accuracy": []}

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                class_weight=self.class_weight,
                random_state=self.random_state,
                verbose=-1,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            results["auc_roc"].append(roc_auc_score(y_val, y_proba))
            results["auc_pr"].append(average_precision_score(y_val, y_proba))
            results["f1"].append(f1_score(y_val, y_pred))
            results["accuracy"].append(accuracy_score(y_val, y_pred))

        return results

    def save(self, path: str) -> None:
        payload = {"state": self.__dict__}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance.__dict__.update(payload["state"])
        return instance


class LSTMModel:
    """LSTM model for sequential credit risk prediction."""

    def __init__(
        self,
        lstm_units_1=32,
        lstm_units_2=16,
        dense_units=16,
        dropout_rate=0.4,
        learning_rate=0.001,
    ):
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]):
        """Build a stacked LSTM binary classifier.

        The Masking layer lets pre-padded zero timesteps pass through without
        affecting the hidden state. Recurrent dropout regularises within-LSTM
        connections independently of the inter-layer Dropout.
        """
        self.model = Sequential(
            [
                # Ignore padded timesteps (value=0.0 from pre-padding in load_sequences)
                Masking(mask_value=0.0, input_shape=input_shape),
                # First LSTM layer: return full sequence so the second layer sees every step
                KerasLSTM(
                    self.lstm_units_1,
                    return_sequences=True,
                    recurrent_dropout=0.3,
                    kernel_regularizer=l2(1e-4),
                ),
                Dropout(self.dropout_rate),
                # Second LSTM layer: return only the final hidden state
                KerasLSTM(
                    self.lstm_units_2,
                    return_sequences=False,
                    recurrent_dropout=0.3,
                    kernel_regularizer=l2(1e-4),
                ),
                Dropout(self.dropout_rate),
                # Intermediate dense for non-linear combination before the sigmoid output
                Dense(self.dense_units, activation="relu"),
                Dropout(self.dropout_rate * 0.67),
                Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

        return self

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        class_weight=None,
    ):
        """Train the LSTM with early stopping on validation AUC.

        When no explicit validation set is provided, 15% of the training data is
        held out automatically. Early stopping restores the best-AUC weights.
        """
        # Build model lazily on first fit call (needs input_shape from data)
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        callbacks = [
            EarlyStopping(
                patience=10, monitor="val_auc", mode="max", restore_best_weights=True
            ),
            ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_auc", mode="max"),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            # Use 15% of training data for validation when no explicit val set supplied
            callbacks[0] = EarlyStopping(
                patience=10, monitor="val_auc", mode="max", restore_best_weights=True
            )

        if class_weight is None:
            from sklearn.utils.class_weight import compute_class_weight as _cw

            _classes = np.unique(y_train)
            _weights = _cw("balanced", classes=_classes, y=y_train)
            class_weight = dict(zip(_classes.astype(int), _weights))

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15 if validation_data is None else 0.0,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        return self.history

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def cross_validate(
        self, X, y, n_splits=5, epochs=100, batch_size=32, class_weight=None
    ) -> Dict:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        results = {"auc_roc": [], "auc_pr": [], "f1": [], "accuracy": []}

        input_shape = (X.shape[1], X.shape[2])

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = LSTMModel(
                lstm_units_1=self.lstm_units_1,
                lstm_units_2=self.lstm_units_2,
                dense_units=self.dense_units,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
            )
            fold_model.build_model(input_shape)
            fold_model.fit(
                X_tr,
                y_tr,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
            )

            y_proba = fold_model.predict_proba(X_val)
            y_pred = (y_proba >= 0.5).astype(int)

            results["auc_roc"].append(roc_auc_score(y_val, y_proba))
            results["auc_pr"].append(average_precision_score(y_val, y_proba))
            results["f1"].append(f1_score(y_val, y_pred))
            results["accuracy"].append(accuracy_score(y_val, y_pred))

            print(f"  Fold {fold + 1} AUC-ROC: {results['auc_roc'][-1]:.4f}")

        return results

    def save(self, path: str) -> None:
        model_path = _normalize_keras_save_path(path)
        self.model.save(model_path)
        config = {
            k: v for k, v in self.__dict__.items() if k not in ("model", "history")
        }
        with open(f"{path}.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        with open(f"{path}.json") as f:
            config = json.load(f)
        instance = cls(**config)
        model_path = _resolve_keras_load_path(path)
        instance.model = tf.keras.models.load_model(model_path)
        instance.history = None
        return instance


class HybridLSTMModel:
    """Hybrid LSTM model combining sequential and static features for credit risk prediction."""

    def __init__(
        self,
        lstm_units_1=32,
        lstm_units_2=16,
        dense_units=16,
        dropout_rate=0.3,
        learning_rate=0.001,
    ):
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self, seq_input_shape: Tuple[int, int], static_input_dim: int):
        """Build a hybrid LSTM model with both sequence and static inputs.

        Architecture:
        - LSTM branch: processes transaction sequences
        - Static branch: processes aggregated user features
        - Fusion: concatenate LSTM output + static features → Dense → sigmoid
        """
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Concatenate

        seq_input = Input(shape=seq_input_shape, name="sequence_input")
        static_input = Input(shape=(static_input_dim,), name="static_input")

        x = Masking(mask_value=0.0, name="masking")(seq_input)
        x = KerasLSTM(self.lstm_units_1, return_sequences=True, recurrent_dropout=0.2)(
            x
        )
        x = Dropout(self.dropout_rate)(x)
        x = KerasLSTM(self.lstm_units_2, return_sequences=False, recurrent_dropout=0.2)(
            x
        )
        x = Dropout(self.dropout_rate)(x)

        s = Dense(self.dense_units // 2, activation="relu")(static_input)
        s = Dropout(self.dropout_rate * 0.67)(s)

        combined = Concatenate()([x, s])
        combined = Dense(self.dense_units, activation="relu")(combined)
        combined = Dropout(self.dropout_rate * 0.67)(combined)
        output = Dense(1, activation="sigmoid")(combined)

        self.model = Model(inputs=[seq_input, static_input], outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

        return self

    def fit(
        self,
        X_train_seq,
        X_train_static,
        y_train,
        X_val_seq=None,
        X_val_static=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        class_weight=None,
    ):
        """Train the hybrid LSTM with early stopping on validation AUC."""
        if self.model is None:
            seq_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            static_dim = X_train_static.shape[1]
            self.build_model(seq_shape, static_dim)

        callbacks = [
            EarlyStopping(
                patience=10, monitor="val_auc", mode="max", restore_best_weights=True
            ),
            ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_auc", mode="max"),
        ]

        validation_data = None
        if X_val_seq is not None and X_val_static is not None:
            validation_data = ([X_val_seq, X_val_static], y_val)

        if class_weight is None:
            from sklearn.utils.class_weight import compute_class_weight as _cw

            _classes = np.unique(y_train)
            _weights = _cw("balanced", classes=_classes, y=y_train)
            class_weight = dict(zip(_classes.astype(int), _weights))

        self.history = self.model.fit(
            [X_train_seq, X_train_static],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15 if validation_data is None else 0.0,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        return self.history

    def predict_proba(self, X_seq, X_static) -> np.ndarray:
        return self.model.predict([X_seq, X_static], verbose=0).flatten()

    def predict(self, X_seq, X_static, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X_seq, X_static) >= threshold).astype(int)

    def cross_validate(
        self,
        X_seq,
        X_static,
        y,
        n_splits=5,
        epochs=100,
        batch_size=32,
        class_weight=None,
    ) -> Dict:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        results = {"auc_roc": [], "auc_pr": [], "f1": [], "accuracy": []}

        seq_shape = (X_seq.shape[1], X_seq.shape[2])
        static_dim = X_static.shape[1]

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            X_seq_tr, X_seq_val = X_seq[train_idx], X_seq[val_idx]
            X_static_tr, X_static_val = X_static[train_idx], X_static[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = HybridLSTMModel(
                lstm_units_1=self.lstm_units_1,
                lstm_units_2=self.lstm_units_2,
                dense_units=self.dense_units,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
            )
            fold_model.build_model(seq_shape, static_dim)
            fold_model.fit(
                X_seq_tr,
                X_static_tr,
                y_tr,
                X_val_seq=X_seq_val,
                X_val_static=X_static_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
            )

            y_proba = fold_model.predict_proba(X_seq_val, X_static_val)
            y_pred = (y_proba >= 0.5).astype(int)

            results["auc_roc"].append(roc_auc_score(y_val, y_proba))
            results["auc_pr"].append(average_precision_score(y_val, y_proba))
            results["f1"].append(f1_score(y_val, y_pred))
            results["accuracy"].append(accuracy_score(y_val, y_pred))

            print(f"  Fold {fold + 1} AUC-ROC: {results['auc_roc'][-1]:.4f}")

        return results

    def save(self, path: str) -> None:
        model_path = _normalize_keras_save_path(path)
        self.model.save(model_path)
        config = {
            k: v for k, v in self.__dict__.items() if k not in ("model", "history")
        }
        with open(f"{path}.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str) -> "HybridLSTMModel":
        with open(f"{path}.json") as f:
            config = json.load(f)
        instance = cls(**config)
        model_path = _resolve_keras_load_path(path)
        instance.model = tf.keras.models.load_model(model_path)
        instance.history = None
        return instance


class ModelEvaluator:
    """Collect probability outputs from multiple models and produce comparative plots and tables."""

    def __init__(self, y_test: np.ndarray):
        self.y_test = y_test
        self.results = {}

    def add_model(
        self,
        name: str,
        y_pred_proba: np.ndarray,
        compute_ci: bool = False,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ):
        """Add a model's predictions to the evaluator.

        Args:
            name: Model name for display
            y_pred_proba: Predicted probabilities for the positive class
            compute_ci: Whether to compute bootstrap confidence intervals
            n_bootstrap: Number of bootstrap iterations (if compute_ci=True)
            ci: Confidence interval level (default 0.95 for 95% CI)
        """
        y_pred = (y_pred_proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        precision_arr, recall_arr, _ = precision_recall_curve(self.y_test, y_pred_proba)

        self.results[name] = {
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred,
            "auc_roc": roc_auc_score(self.y_test, y_pred_proba),
            "auc_pr": average_precision_score(self.y_test, y_pred_proba),
            "f1": f1_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "accuracy": accuracy_score(self.y_test, y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            "fpr": fpr,
            "tpr": tpr,
            "precision_arr": precision_arr,
            "recall_arr": recall_arr,
            "bootstrap_ci": None,
        }

        if compute_ci:
            print(f"  Computing bootstrap CIs for {name} ({n_bootstrap} iterations)...")
            self.results[name]["bootstrap_ci"] = bootstrap_evaluate(
                self.y_test, y_pred_proba, n_bootstrap=n_bootstrap, ci=ci
            )

    def get_comparison_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            rows.append(
                {
                    "Model": name,
                    "AUC-ROC": res["auc_roc"],
                    "AUC-PR": res["auc_pr"],
                    "F1": res["f1"],
                    "Precision": res["precision"],
                    "Recall": res["recall"],
                    "Accuracy": res["accuracy"],
                }
            )
        return pd.DataFrame(rows).set_index("Model")

    def get_comparison_table_with_ci(self) -> pd.DataFrame:
        """Get comparison table with bootstrap confidence intervals.

        Returns a DataFrame with columns like 'AUC-ROC', 'AUC-ROC_CI' where
        the CI column contains formatted strings like '(0.78, 0.84)'.
        """
        rows = []
        for name, res in self.results.items():
            row = {"Model": name}
            ci_data = res.get("bootstrap_ci")

            for metric, key in [
                ("AUC-ROC", "auc_roc"),
                ("AUC-PR", "auc_pr"),
                ("F1", "f1"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("Accuracy", "accuracy"),
            ]:
                row[metric] = res[key]
                if ci_data and metric in ci_data:
                    ci_lower = ci_data[metric]["ci_lower"]
                    ci_upper = ci_data[metric]["ci_upper"]
                    row[f"{metric}_CI"] = f"({ci_lower:.3f}, {ci_upper:.3f})"
                else:
                    row[f"{metric}_CI"] = None

            rows.append(row)

        return pd.DataFrame(rows).set_index("Model")

    def get_comparison_csv_with_ci(self) -> pd.DataFrame:
        """Get comparison table with separate CI columns for CSV export.

        Returns a DataFrame with columns like 'AUC-ROC', 'AUC-ROC_lower', 'AUC-ROC_upper'.
        """
        rows = []
        for name, res in self.results.items():
            row = {"Model": name}
            ci_data = res.get("bootstrap_ci")

            for metric, key in [
                ("AUC-ROC", "auc_roc"),
                ("AUC-PR", "auc_pr"),
                ("F1", "f1"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("Accuracy", "accuracy"),
            ]:
                row[metric] = res[key]
                if ci_data and metric in ci_data:
                    row[f"{metric}_lower"] = ci_data[metric]["ci_lower"]
                    row[f"{metric}_upper"] = ci_data[metric]["ci_upper"]
                else:
                    row[f"{metric}_lower"] = None
                    row[f"{metric}_upper"] = None

            rows.append(row)

        return pd.DataFrame(rows).set_index("Model")

    def format_results_with_ci(self, decimals: int = 3) -> pd.DataFrame:
        """Format results table with inline confidence intervals for display.

        Returns a DataFrame with values formatted as 'mean (lower, upper)'.
        """
        rows = []
        for name, res in self.results.items():
            row = {"Model": name}
            ci_data = res.get("bootstrap_ci")

            for metric, key in [
                ("AUC-ROC", "auc_roc"),
                ("AUC-PR", "auc_pr"),
                ("F1", "f1"),
                ("Precision", "precision"),
                ("Recall", "recall"),
            ]:
                val = res[key]
                if ci_data and metric in ci_data:
                    ci_lower = ci_data[metric]["ci_lower"]
                    ci_upper = ci_data[metric]["ci_upper"]
                    row[metric] = (
                        f"{val:.{decimals}f} ({ci_lower:.{decimals}f}, {ci_upper:.{decimals}f})"
                    )
                else:
                    row[metric] = f"{val:.{decimals}f}"

            rows.append(row)

        return pd.DataFrame(rows).set_index("Model")

    def plot_roc_curves(self, ax=None, figsize=(8, 6)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for name, res in self.results.items():
            ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc_roc']:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_pr_curves(self, ax=None, figsize=(8, 6)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        prevalence = self.y_test.mean()
        for name, res in self.results.items():
            ax.plot(
                res["recall_arr"],
                res["precision_arr"],
                label=f"{name} (AP={res['auc_pr']:.3f})",
            )

        ax.axhline(
            y=prevalence,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Baseline ({prevalence:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_confusion_matrices(self, figsize=(15, 4)):
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, self.results.items()):
            cm = res["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"],
            )
            ax.set_title(f"{name}")
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")

        plt.tight_layout()
        return fig

    def plot_threshold_analysis(self, model_name: str, ax=None, figsize=(8, 6)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        y_proba = self.results[model_name]["y_pred_proba"]
        thresholds = np.arange(0.01, 1.0, 0.01)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            if y_pred.sum() == 0:
                precisions.append(0)
            else:
                precisions.append(precision_score(self.y_test, y_pred, zero_division=0))
            recalls.append(recall_score(self.y_test, y_pred, zero_division=0))
            f1s.append(f1_score(self.y_test, y_pred, zero_division=0))

        ax.plot(thresholds, precisions, label="Precision")
        ax.plot(thresholds, recalls, label="Recall")
        ax.plot(thresholds, f1s, label="F1")

        best_idx = np.argmax(f1s)
        ax.axvline(
            x=thresholds[best_idx],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Best F1={f1s[best_idx]:.3f} @ t={thresholds[best_idx]:.2f}",
        )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold Analysis: {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_feature_importance_comparison(
        self,
        lr_coefficients: pd.DataFrame,
        xgb_importance: pd.DataFrame,
        top_n=15,
        figsize=(14, 6),
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        lr_top = lr_coefficients.head(top_n)
        colors = ["#e74c3c" if c > 0 else "#3498db" for c in lr_top["coefficient"]]
        ax1.barh(range(len(lr_top)), lr_top["coefficient"], color=colors)
        ax1.set_yticks(range(len(lr_top)))
        ax1.set_yticklabels(lr_top["feature"], fontsize=9)
        ax1.set_title(f"Logistic Regression (Top {top_n})")
        ax1.set_xlabel("Coefficient")
        ax1.invert_yaxis()

        xgb_top = xgb_importance.head(top_n)
        ax2.barh(range(len(xgb_top)), xgb_top["importance"], color="#2ecc71")
        ax2.set_yticks(range(len(xgb_top)))
        ax2.set_yticklabels(xgb_top["feature"], fontsize=9)
        ax2.set_title(f"XGBoost Feature Importance (Top {top_n})")
        ax2.set_xlabel("Importance")
        ax2.invert_yaxis()

        plt.tight_layout()
        return fig

    def print_classification_reports(self):
        for name, res in self.results.items():
            print(f"\n{'=' * 50}")
            print(f"  {name}")
            print(f"{'=' * 50}")
            print(
                classification_report(
                    self.y_test, res["y_pred"], target_names=["Non-Default", "Default"]
                )
            )
