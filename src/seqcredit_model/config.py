from pathlib import Path

# Anchor to project root: src/seqcredit_model/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

CALIBRATION_FILE  = DATA_DIR / 'calibration.json'
FEATURES_FILE     = DATA_DIR / 'features.csv'
SUMMARIES_FILE    = DATA_DIR / 'summary_extended.csv'
TRANSACTIONS_DIR  = DATA_DIR / 'user_transactions'
LSTM_CACHE_FILE   = DATA_DIR / 'lstm_sequences.npz'
