"""Centralised path constants derived from project root."""

import os
from pathlib import Path

RANDOM_SEED = 42

# Anchor to project root: src/seqcredit_model/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_storage_path(path_value: str) -> Path:
    """Resolve local and DBFS-style paths to a filesystem path."""
    if path_value.startswith("dbfs:/"):
        return Path("/dbfs") / path_value.removeprefix("dbfs:/").lstrip("/")
    return Path(path_value)


def _resolve_runtime_path(env_var: str, default_path: Path) -> Path:
    """Resolve an optional runtime path override."""
    override = os.environ.get(env_var)
    if override:
        return _resolve_storage_path(override)
    return default_path


def _default_data_dir() -> Path:
    """Choose the default repo-local data root."""
    return _resolve_runtime_path("SEQCREDIT_DATA_DIR", PROJECT_ROOT / "data")


DATA_DIR = _default_data_dir()

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


def get_runtime_lstm_cache_file() -> Path:
    """Return the cache path for the current runtime."""
    return _resolve_runtime_path("SEQCREDIT_LSTM_CACHE_FILE", LSTM_CACHE_FILE)


def get_runtime_raw_seq_file() -> Path:
    """Return the raw sequence NPZ path for the current runtime."""
    return _resolve_runtime_path("SEQCREDIT_RAW_SEQ_FILE", RAW_SEQ_FILE)


def get_runtime_user_features_file() -> Path:
    """Return the feature table path for the current runtime."""
    return _resolve_runtime_path("SEQCREDIT_USER_FEATURES_FILE", USER_FEATURES_FILE)


def get_runtime_user_labels_file() -> Path:
    """Return the label table path for the current runtime."""
    return _resolve_runtime_path("SEQCREDIT_USER_LABELS_FILE", USER_LABELS_FILE)

LEGACY_DIR = DATA_DIR / "legacy"
MODELS_DIR = PROJECT_ROOT / "models"
