from pathlib import Path

# Anchor to project root: src/seqcredit_model/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

SYNTHETIC_PARAMS_FILE = DATA_DIR / 'synthetic_params.json'
USER_FEATURES_FILE    = DATA_DIR / 'user_features.csv'
USER_LABELS_FILE      = DATA_DIR / 'user_labels.csv'
TRANSACTIONS_DIR      = DATA_DIR / 'user_transactions'
LSTM_CACHE_FILE       = DATA_DIR / 'lstm_sequences.npz'
LEGACY_DIR            = DATA_DIR / 'legacy'
