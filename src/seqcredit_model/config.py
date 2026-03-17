"""Centralised path constants derived from project root."""

from pathlib import Path

# Anchor to project root: src/seqcredit_model/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Synthetic data generation inputs/outputs
SYNTHETIC_PARAMS_FILE = DATA_DIR / "synthetic_params.json"
TRANSACTIONS_DIR = DATA_DIR / "user_transactions"

# Pre-built user-level feature and label tables (written by feature_engineering.py)
USER_FEATURES_FILE = DATA_DIR / "user_features.csv"
USER_LABELS_FILE = DATA_DIR / "user_labels.csv"

# Cached padded LSTM sequence arrays (written by CreditRiskDataLoader.load_sequences)
LSTM_CACHE_FILE = DATA_DIR / "lstm_sequences.npz"

LEGACY_DIR = DATA_DIR / "legacy"
MODELS_DIR = PROJECT_ROOT / "models"
