"""
Credit Risk Prediction Models for Mobile Money Transaction Data

Location: src/models.py

Provides three model implementations for binary default prediction:
  1. Logistic Regression (static, user-level features)
  2. XGBoost Classifier (static, user-level features)
  3. LSTM (sequential, per-transaction features)

Target: default (credit_risk_label=2) vs non-default (0 or 1).
Non-borrowers (credit_risk_label=-1) are excluded.

Usage:
    from src.models import (
        CreditRiskDataLoader, LogisticRegressionModel,
        XGBoostModel, LSTMModel, ModelEvaluator, set_random_seeds
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42

# LSTM feature columns selected from extract_all_features() output
LSTM_FEATURE_COLUMNS = [
    # Transaction characteristics (8)
    'log_amount', 'is_micro_txn', 'is_small_txn', 'is_medium_txn',
    'is_large_txn', 'fee_to_amount_ratio', 'total_cost', 'has_fees',
    # Transaction type one-hot (7)
    'is_transfer', 'is_debit', 'is_payment', 'is_payment_send',
    'is_cash_out', 'is_cash_in', 'is_adjustment',
    # Temporal (6)
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'time_since_last_txn_hours',
    # Balance dynamics (6)
    'log_balance_before', 'balance_pct_change', 'is_low_balance',
    'is_zero_balance', 'amount_to_balance_ratio', 'will_deplete_balance',
    # Behavioral (3)
    'is_repeated_recipient', 'is_self_transfer', 'unique_txn_types_last_10',
    # Risk indicators (3)
    'unusual_hour', 'rapid_transaction', 'risk_score',
    # Loan-specific (2)
    'is_loan_disbursement', 'is_loan_repayment',
]


def set_random_seeds(seed=RANDOM_SEED):
    """Set numpy, random, and tensorflow seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights from label array."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes.astype(int), weights))


# =============================================================================
# DATA LOADER
# =============================================================================

class CreditRiskDataLoader:
    """
    Handles all data loading, merging, filtering, and splitting for
    credit risk prediction models.

    Ensures that the same train/test user split is shared across
    static (LR, XGBoost) and sequential (LSTM) models for fair comparison.
    """

    def __init__(self,
                 features_path='data/user_features.csv',
                 summaries_path='data/user_summaries.csv',
                 transactions_dir='data/user_transactions',
                 test_size=0.2,
                 random_state=RANDOM_SEED):
        self.features_path = features_path
        self.summaries_path = summaries_path
        self.transactions_dir = transactions_dir
        self.test_size = test_size
        self.random_state = random_state

        self._train_user_ids = None
        self._test_user_ids = None
        self._static_data = None

    def load_static_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and merge user_features.csv with user_summaries.csv.
        Filter non-borrowers, create binary target, add loan features.

        Returns:
            (X, y) where X is the feature DataFrame and y is binary target.
        """
        df_features = pd.read_csv(self.features_path)
        df_summaries = pd.read_csv(self.summaries_path)

        # Merge on user_id
        df = df_features.merge(df_summaries, on='user_id', how='inner')

        # Filter out non-borrowers
        df = df[df['credit_risk_label'] != -1].copy()

        # Binary target: 1 = default (label=2), 0 = non-default (label 0 or 1)
        df['default'] = (df['credit_risk_label'] == 2).astype(int)

        # Add loan-specific features
        df = self._engineer_loan_features(df)

        # Separate features and target
        drop_cols = ['user_id', 'credit_risk_label', 'credit_archetype',
                     'default', 'total_transactions_y']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()
        y = df['default'].copy()

        # Store user_ids for later
        self._user_ids = df['user_id'].values
        self._static_data = (X, y)

        return X, y

    def prepare_static_splits(self) -> Dict:
        """
        Stratified train/test split with StandardScaler.

        Returns dict with X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler, feature_names.
        """
        if self._static_data is None:
            self.load_static_data()

        X, y = self._static_data

        # Stratified split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=self.test_size,
            stratify=y, random_state=self.random_state
        )

        self._train_user_ids = set(self._user_ids[train_idx])
        self._test_user_ids = set(self._user_ids[test_idx])

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'feature_names': list(X.columns),
        }

    def load_sequences(self, max_seq_len=50,
                       feature_columns=None,
                       cache_path='data/lstm_sequences.npz') -> Dict:
        """
        Load per-user transaction CSVs for borrowers, extract features,
        create padded sequences for LSTM.

        Uses the same train/test user split as prepare_static_splits().

        Returns dict with X_train_seq, X_test_seq, y_train, y_test,
        feature_names, seq_scaler.
        """
        from src.feature_engineering import TemporalTransactionFeatureEngineer

        if feature_columns is None:
            feature_columns = LSTM_FEATURE_COLUMNS

        # Ensure static splits have been done first
        if self._train_user_ids is None:
            self.prepare_static_splits()

        cache_file = Path(cache_path)
        if cache_file.exists():
            print(f"Loading cached sequences from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            return {
                'X_train_seq': data['X_train_seq'],
                'X_test_seq': data['X_test_seq'],
                'y_train': data['y_train'],
                'y_test': data['y_test'],
                'feature_names': list(data['feature_names']),
            }

        # Load summaries for target labels
        df_summaries = pd.read_csv(self.summaries_path)
        df_summaries = df_summaries[df_summaries['credit_risk_label'] != -1]
        user_labels = dict(zip(
            df_summaries['user_id'],
            (df_summaries['credit_risk_label'] == 2).astype(int)
        ))

        engineer = TemporalTransactionFeatureEngineer()
        transactions_path = Path(self.transactions_dir)

        train_sequences = []
        train_labels = []
        test_sequences = []
        test_labels = []

        borrower_ids = sorted(user_labels.keys())
        total = len(borrower_ids)

        print(f"Processing {total} borrower transaction files...")

        for i, user_id in enumerate(borrower_ids):
            filepath = transactions_path / f"{user_id}.csv"
            if not filepath.exists():
                continue

            try:
                df_user = pd.read_csv(filepath)
                if df_user.empty or len(df_user) < 2:
                    continue

                # Add loan-specific binary columns before feature engineering
                df_user['is_loan_disbursement'] = (
                    df_user['TRANS. TYPE'] == 'CREDIT'
                ).astype(int)
                df_user['is_loan_repayment'] = (
                    df_user['TRANS. TYPE'] == 'LOAN_REPAYMENT'
                ).astype(int)

                # Run feature engineering
                df_features = engineer.extract_all_features(df_user)

                # Select LSTM feature columns (use only those that exist)
                available_cols = [c for c in feature_columns
                                  if c in df_features.columns]
                seq = df_features[available_cols].values.astype(np.float32)

                # Replace NaN/inf with 0
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

                label = user_labels[user_id]

                if user_id in self._train_user_ids:
                    train_sequences.append(seq)
                    train_labels.append(label)
                elif user_id in self._test_user_ids:
                    test_sequences.append(seq)
                    test_labels.append(label)

            except Exception as e:
                if (i + 1) % 1000 == 0:
                    print(f"  Warning: {user_id} failed: {e}")
                continue

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"  Processed {i + 1}/{total} users...")

        print(f"  Train sequences: {len(train_sequences)}, "
              f"Test sequences: {len(test_sequences)}")

        # Scale features before padding
        # Concatenate all train sequences to fit scaler
        all_train = np.vstack(train_sequences)
        seq_scaler = StandardScaler()
        seq_scaler.fit(all_train)

        # Transform each sequence individually
        train_sequences = [seq_scaler.transform(s) for s in train_sequences]
        test_sequences = [seq_scaler.transform(s) for s in test_sequences]

        # Pad sequences (pre-padding so recent transactions are at the end)
        X_train_seq = pad_sequences(
            train_sequences, maxlen=max_seq_len,
            padding='pre', truncating='pre',
            dtype='float32', value=0.0
        )
        X_test_seq = pad_sequences(
            test_sequences, maxlen=max_seq_len,
            padding='pre', truncating='pre',
            dtype='float32', value=0.0
        )

        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Get the actual feature names used
        sample_df = pd.read_csv(
            transactions_path / f"{borrower_ids[0]}.csv"
        )
        sample_df['is_loan_disbursement'] = (
            sample_df['TRANS. TYPE'] == 'CREDIT'
        ).astype(int)
        sample_df['is_loan_repayment'] = (
            sample_df['TRANS. TYPE'] == 'LOAN_REPAYMENT'
        ).astype(int)
        sample_features = engineer.extract_all_features(sample_df)
        actual_feature_names = [c for c in feature_columns
                                if c in sample_features.columns]

        # Cache for future runs
        np.savez(
            cache_path,
            X_train_seq=X_train_seq,
            X_test_seq=X_test_seq,
            y_train=y_train,
            y_test=y_test,
            feature_names=np.array(actual_feature_names),
        )
        print(f"Cached sequences to {cache_path}")
        print(f"Shapes: train={X_train_seq.shape}, test={X_test_seq.shape}")

        return {
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': actual_feature_names,
        }

    def _engineer_loan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute loan-specific aggregate features from per-user transaction CSVs
        and merge them into the existing feature DataFrame.
        """
        transactions_path = Path(self.transactions_dir)
        loan_features_list = []

        for _, row in df.iterrows():
            user_id = row['user_id']
            filepath = transactions_path / f"{user_id}.csv"

            loan_feats = {'user_id': user_id}

            if filepath.exists():
                try:
                    df_user = pd.read_csv(filepath)

                    credit_txns = df_user[df_user['TRANS. TYPE'] == 'CREDIT']
                    repay_txns = df_user[
                        df_user['TRANS. TYPE'] == 'LOAN_REPAYMENT'
                    ]
                    total_txns = len(df_user)

                    # Loan activity features
                    loan_feats['total_loan_volume'] = (
                        credit_txns['AMOUNT'].sum() if len(credit_txns) > 0
                        else 0
                    )
                    loan_feats['avg_loan_amount'] = (
                        credit_txns['AMOUNT'].mean() if len(credit_txns) > 0
                        else 0
                    )
                    loan_feats['max_loan_amount'] = (
                        credit_txns['AMOUNT'].max() if len(credit_txns) > 0
                        else 0
                    )
                    loan_feats['loan_to_total_volume_ratio'] = (
                        loan_feats['total_loan_volume'] /
                        (df_user['AMOUNT'].sum() + 1)
                    )
                    loan_feats['pct_credit_transactions'] = (
                        len(credit_txns) / total_txns if total_txns > 0
                        else 0
                    )

                    # Repayment features
                    loan_feats['total_repayment_volume'] = (
                        repay_txns['AMOUNT'].sum() if len(repay_txns) > 0
                        else 0
                    )
                    loan_feats['repayment_to_loan_ratio'] = (
                        loan_feats['total_repayment_volume'] /
                        (loan_feats['total_loan_volume'] + 1)
                    )
                    loan_feats['avg_repayment_amount'] = (
                        repay_txns['AMOUNT'].mean() if len(repay_txns) > 0
                        else 0
                    )
                    loan_feats['has_any_repayment'] = (
                        1 if len(repay_txns) > 0 else 0
                    )
                    loan_feats['pct_repayment_transactions'] = (
                        len(repay_txns) / total_txns if total_txns > 0
                        else 0
                    )

                    # Timing features
                    if len(credit_txns) > 0 and len(repay_txns) > 0:
                        df_user['TRANSACTION DATE'] = pd.to_datetime(
                            df_user['TRANSACTION DATE']
                        )
                        days_list = []
                        for _, ct in credit_txns.iterrows():
                            ct_date = pd.to_datetime(ct['TRANSACTION DATE'])
                            later_repays = repay_txns[
                                pd.to_datetime(
                                    repay_txns['TRANSACTION DATE']
                                ) > ct_date
                            ]
                            if len(later_repays) > 0:
                                first_repay = pd.to_datetime(
                                    later_repays.iloc[0]['TRANSACTION DATE']
                                )
                                days_list.append(
                                    (first_repay - ct_date).days
                                )
                        loan_feats['avg_days_loan_to_repayment'] = (
                            np.mean(days_list) if days_list else 30
                        )
                    else:
                        loan_feats['avg_days_loan_to_repayment'] = 0

                    if len(credit_txns) > 0:
                        first_credit_idx = df_user[
                            df_user['TRANS. TYPE'] == 'CREDIT'
                        ].index[0]
                        loan_feats['loan_timing_in_sequence'] = (
                            first_credit_idx / total_txns
                            if total_txns > 0 else 0
                        )
                    else:
                        loan_feats['loan_timing_in_sequence'] = 0

                    # Balance at loan features
                    if len(credit_txns) > 0:
                        loan_feats['avg_balance_at_loan'] = (
                            credit_txns['BAL BEFORE'].mean()
                        )
                        loan_feats['min_balance_at_loan'] = (
                            credit_txns['BAL BEFORE'].min()
                        )
                        loan_feats['balance_to_loan_ratio_at_disbursement'] = (
                            (credit_txns['BAL BEFORE'] /
                             (credit_txns['AMOUNT'] + 1)).mean()
                        )
                    else:
                        loan_feats['avg_balance_at_loan'] = 0
                        loan_feats['min_balance_at_loan'] = 0
                        loan_feats['balance_to_loan_ratio_at_disbursement'] = 0

                except Exception:
                    # Fill with zeros if file reading fails
                    for col in ['total_loan_volume', 'avg_loan_amount',
                                'max_loan_amount', 'loan_to_total_volume_ratio',
                                'pct_credit_transactions',
                                'total_repayment_volume',
                                'repayment_to_loan_ratio',
                                'avg_repayment_amount', 'has_any_repayment',
                                'pct_repayment_transactions',
                                'avg_days_loan_to_repayment',
                                'loan_timing_in_sequence',
                                'avg_balance_at_loan', 'min_balance_at_loan',
                                'balance_to_loan_ratio_at_disbursement']:
                        loan_feats[col] = 0
            else:
                for col in ['total_loan_volume', 'avg_loan_amount',
                            'max_loan_amount', 'loan_to_total_volume_ratio',
                            'pct_credit_transactions',
                            'total_repayment_volume',
                            'repayment_to_loan_ratio',
                            'avg_repayment_amount', 'has_any_repayment',
                            'pct_repayment_transactions',
                            'avg_days_loan_to_repayment',
                            'loan_timing_in_sequence',
                            'avg_balance_at_loan', 'min_balance_at_loan',
                            'balance_to_loan_ratio_at_disbursement']:
                    loan_feats[col] = 0

            loan_features_list.append(loan_feats)

        df_loan = pd.DataFrame(loan_features_list)
        df = df.merge(df_loan, on='user_id', how='left')

        # Also keep loans_taken and final_credit_limit from summaries
        # (they're already in df from the earlier merge)

        return df

    def get_class_weights(self) -> Dict[int, float]:
        """Compute balanced class weights."""
        if self._static_data is None:
            self.load_static_data()
        _, y = self._static_data
        return _compute_class_weights(y.values)

    def get_scale_pos_weight(self) -> float:
        """Return count(non-default) / count(default) for XGBoost."""
        if self._static_data is None:
            self.load_static_data()
        _, y = self._static_data
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / max(n_pos, 1)


# =============================================================================
# LOGISTIC REGRESSION MODEL
# =============================================================================

class LogisticRegressionModel:
    """Logistic Regression for credit risk prediction."""

    def __init__(self, class_weight='balanced', C=1.0, max_iter=1000,
                 random_state=RANDOM_SEED):
        self.model = LogisticRegression(
            class_weight=class_weight, C=C,
            max_iter=max_iter, random_state=random_state
        )
        self.random_state = random_state

    def fit(self, X_train, y_train):
        """Fit the logistic regression model."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return probability of default (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5) -> np.ndarray:
        """Return binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_coefficients(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Return DataFrame with feature names and coefficients,
        sorted by absolute value.
        """
        coefs = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0]
        })
        coefs['abs_coefficient'] = coefs['coefficient'].abs()
        return coefs.sort_values('abs_coefficient', ascending=False)

    def cross_validate(self, X, y, n_splits=5) -> Dict:
        """Stratified K-fold cross-validation."""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        results = {
            'auc_roc': [], 'auc_pr': [], 'f1': [], 'accuracy': []
        }

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(
                class_weight=self.model.class_weight,
                C=self.model.C, max_iter=self.model.max_iter,
                random_state=self.random_state
            )
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            results['auc_roc'].append(roc_auc_score(y_val, y_proba))
            results['auc_pr'].append(average_precision_score(y_val, y_proba))
            results['f1'].append(f1_score(y_val, y_pred))
            results['accuracy'].append(accuracy_score(y_val, y_pred))

        return results


# =============================================================================
# XGBOOST MODEL
# =============================================================================

class XGBoostModel:
    """XGBoost classifier for credit risk prediction."""

    def __init__(self, scale_pos_weight=None, n_estimators=200,
                 max_depth=5, learning_rate=0.1, min_child_weight=3,
                 subsample=0.8, colsample_bytree=0.8,
                 random_state=RANDOM_SEED):
        self.params = {
            'scale_pos_weight': scale_pos_weight or 1.0,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'eval_metric': 'auc',
            'use_label_encoder': False,
        }
        self.random_state = random_state
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            early_stopping_rounds=20):
        """Fit XGBoost with optional early stopping."""
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return probability of default (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5) -> np.ndarray:
        """Return binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str],
                               importance_type='weight') -> pd.DataFrame:
        """
        Return DataFrame with feature names and importance scores,
        sorted descending.
        """
        importance = self.model.feature_importances_
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        return imp_df.sort_values('importance', ascending=False)

    def cross_validate(self, X, y, n_splits=5) -> Dict:
        """Stratified K-fold cross-validation."""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        results = {
            'auc_roc': [], 'auc_pr': [], 'f1': [], 'accuracy': []
        }

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**self.params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            results['auc_roc'].append(roc_auc_score(y_val, y_proba))
            results['auc_pr'].append(average_precision_score(y_val, y_proba))
            results['f1'].append(f1_score(y_val, y_pred))
            results['accuracy'].append(accuracy_score(y_val, y_pred))

        return results


# =============================================================================
# LSTM MODEL
# =============================================================================

class LSTMModel:
    """LSTM model for sequential credit risk prediction."""

    def __init__(self, lstm_units_1=64, lstm_units_2=32,
                 dense_units=16, dropout_rate=0.3, learning_rate=0.001):
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]):
        """
        Build the Keras LSTM model.

        Architecture:
            Masking -> LSTM(64) -> Dropout -> LSTM(32) -> Dropout
            -> Dense(16, relu) -> Dropout -> Dense(1, sigmoid)

        Args:
            input_shape: (max_seq_len, n_features)
        """
        self.model = Sequential([
            Masking(mask_value=0.0, input_shape=input_shape),
            KerasLSTM(self.lstm_units_1, return_sequences=True),
            Dropout(self.dropout_rate),
            KerasLSTM(self.lstm_units_2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout_rate * 0.67),  # lighter dropout before output
            Dense(1, activation='sigmoid'),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')],
        )

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, class_weight=None):
        """
        Train the LSTM model with early stopping.

        Returns the training history object.
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        callbacks = [
            EarlyStopping(
                patience=10, monitor='val_auc',
                mode='max', restore_best_weights=True
            ),
            ReduceLROnPlateau(
                patience=5, factor=0.5, monitor='val_auc', mode='max'
            ),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            # Use a portion of training data for validation
            callbacks[0] = EarlyStopping(
                patience=10, monitor='val_auc',
                mode='max', restore_best_weights=True
            )

        self.history = self.model.fit(
            X_train, y_train,
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
        """Return probability of default."""
        return self.model.predict(X, verbose=0).flatten()

    def predict(self, X, threshold=0.5) -> np.ndarray:
        """Return binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def cross_validate(self, X, y, n_splits=5, epochs=100,
                       batch_size=32, class_weight=None) -> Dict:
        """
        Stratified K-fold CV for LSTM.
        Rebuilds model from scratch each fold.
        """
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
        )
        results = {
            'auc_roc': [], 'auc_pr': [], 'f1': [], 'accuracy': []
        }

        input_shape = (X.shape[1], X.shape[2])

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- LSTM CV Fold {fold + 1}/{n_splits} ---")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Rebuild model from scratch
            fold_model = LSTMModel(
                lstm_units_1=self.lstm_units_1,
                lstm_units_2=self.lstm_units_2,
                dense_units=self.dense_units,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
            )
            fold_model.build_model(input_shape)
            fold_model.fit(
                X_tr, y_tr, X_val=X_val, y_val=y_val,
                epochs=epochs, batch_size=batch_size,
                class_weight=class_weight,
            )

            y_proba = fold_model.predict_proba(X_val)
            y_pred = (y_proba >= 0.5).astype(int)

            results['auc_roc'].append(roc_auc_score(y_val, y_proba))
            results['auc_pr'].append(average_precision_score(y_val, y_proba))
            results['f1'].append(f1_score(y_val, y_pred))
            results['accuracy'].append(accuracy_score(y_val, y_pred))

            print(f"  Fold {fold + 1} AUC-ROC: "
                  f"{results['auc_roc'][-1]:.4f}")

        return results


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    """Compare and visualize results across multiple models."""

    def __init__(self, y_test: np.ndarray):
        self.y_test = y_test
        self.results = {}

    def add_model(self, name: str, y_pred_proba: np.ndarray):
        """Store predictions and compute all metrics for one model."""
        y_pred = (y_pred_proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        precision_arr, recall_arr, _ = precision_recall_curve(
            self.y_test, y_pred_proba
        )

        self.results[name] = {
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'auc_pr': average_precision_score(self.y_test, y_pred_proba),
            'f1': f1_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'accuracy': accuracy_score(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'fpr': fpr,
            'tpr': tpr,
            'precision_arr': precision_arr,
            'recall_arr': recall_arr,
        }

    def get_comparison_table(self) -> pd.DataFrame:
        """Return DataFrame comparing all models."""
        rows = []
        for name, res in self.results.items():
            rows.append({
                'Model': name,
                'AUC-ROC': res['auc_roc'],
                'AUC-PR': res['auc_pr'],
                'F1': res['f1'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'Accuracy': res['accuracy'],
            })
        return pd.DataFrame(rows).set_index('Model')

    def plot_roc_curves(self, ax=None, figsize=(8, 6)):
        """Plot ROC curves for all models on the same axes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for name, res in self.results.items():
            ax.plot(res['fpr'], res['tpr'],
                    label=f"{name} (AUC={res['auc_roc']:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_pr_curves(self, ax=None, figsize=(8, 6)):
        """Plot Precision-Recall curves for all models."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        prevalence = self.y_test.mean()
        for name, res in self.results.items():
            ax.plot(res['recall_arr'], res['precision_arr'],
                    label=f"{name} (AP={res['auc_pr']:.3f})")

        ax.axhline(y=prevalence, color='k', linestyle='--', alpha=0.5,
                    label=f'Baseline ({prevalence:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_confusion_matrices(self, figsize=(15, 4)):
        """Plot confusion matrices side by side for all models."""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, self.results.items()):
            cm = res['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Non-Default', 'Default'],
                        yticklabels=['Non-Default', 'Default'])
            ax.set_title(f'{name}')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

        plt.tight_layout()
        return fig

    def plot_threshold_analysis(self, model_name: str, ax=None,
                                figsize=(8, 6)):
        """Plot precision, recall, F1 vs threshold for a specific model."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        y_proba = self.results[model_name]['y_pred_proba']
        thresholds = np.arange(0.01, 1.0, 0.01)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            if y_pred.sum() == 0:
                precisions.append(0)
            else:
                precisions.append(precision_score(
                    self.y_test, y_pred, zero_division=0
                ))
            recalls.append(recall_score(
                self.y_test, y_pred, zero_division=0
            ))
            f1s.append(f1_score(self.y_test, y_pred, zero_division=0))

        ax.plot(thresholds, precisions, label='Precision')
        ax.plot(thresholds, recalls, label='Recall')
        ax.plot(thresholds, f1s, label='F1')

        # Mark optimal F1
        best_idx = np.argmax(f1s)
        ax.axvline(x=thresholds[best_idx], color='gray', linestyle='--',
                    alpha=0.7,
                    label=f'Best F1={f1s[best_idx]:.3f} '
                          f'@ t={thresholds[best_idx]:.2f}')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Threshold Analysis: {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_feature_importance_comparison(self,
                                           lr_coefficients: pd.DataFrame,
                                           xgb_importance: pd.DataFrame,
                                           top_n=15,
                                           figsize=(14, 6)):
        """Side-by-side feature importance plots for LR and XGBoost."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # LR coefficients
        lr_top = lr_coefficients.head(top_n)
        colors = ['#e74c3c' if c > 0 else '#3498db'
                   for c in lr_top['coefficient']]
        ax1.barh(range(len(lr_top)), lr_top['coefficient'], color=colors)
        ax1.set_yticks(range(len(lr_top)))
        ax1.set_yticklabels(lr_top['feature'], fontsize=9)
        ax1.set_title(f'Logistic Regression (Top {top_n})')
        ax1.set_xlabel('Coefficient')
        ax1.invert_yaxis()

        # XGBoost importance
        xgb_top = xgb_importance.head(top_n)
        ax2.barh(range(len(xgb_top)), xgb_top['importance'], color='#2ecc71')
        ax2.set_yticks(range(len(xgb_top)))
        ax2.set_yticklabels(xgb_top['feature'], fontsize=9)
        ax2.set_title(f'XGBoost Feature Importance (Top {top_n})')
        ax2.set_xlabel('Importance')
        ax2.invert_yaxis()

        plt.tight_layout()
        return fig

    def print_classification_reports(self):
        """Print classification reports for all models."""
        for name, res in self.results.items():
            print(f"\n{'='*50}")
            print(f"  {name}")
            print(f"{'='*50}")
            print(classification_report(
                self.y_test, res['y_pred'],
                target_names=['Non-Default', 'Default']
            ))
