"""Centralised path constants derived from project root."""

from pathlib import Path

RANDOM_SEED = 42

# Anchor to project root: src/seqcredit_model/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Synthetic data generation inputs/outputs
SYNTHETIC_PARAMS_FILE = PROJECT_ROOT / "src" / "synthetic_params.json"
TRANSACTIONS_DIR = DATA_DIR / "user_transactions"

# Pre-built user-level feature and label tables (written by pipeline.py)
USER_FEATURES_FILE = DATA_DIR / "user_features.csv"
USER_LABELS_FILE = DATA_DIR / "user_labels.csv"

# Cached padded LSTM sequence arrays (written by CreditRiskDataLoader.load_sequences)
LSTM_CACHE_FILE = DATA_DIR / "lstm_sequences.npz"

# Raw per-user sequences built by real_data_pipeline.build_sequences_spark()
# Contains all borrowers; CreditRiskDataLoader.load_sequences() splits train/test from this.
RAW_SEQ_FILE = DATA_DIR / "lstm_sequences_raw.npz"

LEGACY_DIR = DATA_DIR / "legacy"
MODELS_DIR = PROJECT_ROOT / "models"
