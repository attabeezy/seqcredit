import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from seqcredit_model.synthesize import CalibratedMoMoDataGenerator
from seqcredit_model.pipeline import build_user_feature_dataset
from seqcredit_model.credit_model import HybridLSTMModel, CreditRiskDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_flow():
    """Tests the flow from synthesis to feature engineering and model prediction."""
    try:
        # 1. Setup paths
        base_dir = Path("test_sandbox")
        if base_dir.exists():
            import shutil
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True)
        
        data_dir = base_dir / "data"
        transactions_dir = data_dir / "user_transactions"
        user_features_file = data_dir / "user_features.csv"
        user_labels_file = data_dir / "user_labels.csv"
        models_dir = Path("models")
        
        # 2. Mini-Synthesis
        logger.info("Step 1: Running mini-synthesis...")
        generator = CalibratedMoMoDataGenerator(
            n_users=100,
            avg_transactions_per_user=50,
            output_dir=str(transactions_dir),
            seed=42
        )
        summary_df = generator.generate_dataset()
        
        # The generator saves labels to output_dir.parent / "user_labels.csv"
        # We'll reload and map to is_bad (1 or 2 means bad)
        labels_df = pd.read_csv(data_dir / "user_labels.csv")
        labels_df['is_bad'] = (labels_df['credit_risk_label'] > 0).astype(int)
        labels_df.set_index('user_id', inplace=True)
        # Loader expects 'credit_risk_label' to be present for validation
        labels_df[['credit_risk_label', 'is_bad']].to_csv(user_labels_file)
        
        # 3. Feature Engineering
        logger.info("Step 2: Testing Feature Engineering...")
        build_user_feature_dataset(
            transactions_dir=str(transactions_dir),
            output_path=str(user_features_file)
        )
        
        # 4. Data Loading
        logger.info("Step 3: Testing Data Loading...")
        loader = CreditRiskDataLoader(
            features_path=str(user_features_file),
            summaries_path=str(user_labels_file),
            transactions_dir=str(transactions_dir)
        )
        
        # We need to make sure we have enough data for a split
        # Just check if we can load static data
        X_static, y = loader.load_static_data()
        logger.info(f"Static data loaded: {X_static.shape}")

        # Prepare splits for sequential loading
        splits = loader.prepare_static_splits()
        
        # Try loading sequences
        lstm_cache_file = data_dir / "lstm_sequences_test.npz"
        seq_data = loader.load_sequences(max_seq_len=20, cache_path=str(lstm_cache_file))
        X_seq_train = seq_data['X_train_seq']
        logger.info(f"Sequential data loaded: Train shape {X_seq_train.shape}")

        # 5. Hybrid Model Sanity Check
        logger.info("Step 4: Testing Hybrid Model initialization...")
        model = HybridLSTMModel()
        model.build_model(
            seq_input_shape=(X_seq_train.shape[1], X_seq_train.shape[2]),
            static_input_dim=X_static.shape[1]
        )
        # Dummy prediction
        X_static_sample = X_static.iloc[:1].values
        X_seq_sample = X_seq_train[:1]
        pred = model.model.predict([X_seq_sample, X_static_sample])
        logger.info(f"Hybrid prediction success: {pred}")

        logger.info("End-to-end flow test COMPLETED SUCCESSFULY.")

    except Exception as e:
        logger.error(f"End-to-end flow test FAILED: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        test_pipeline_flow()
        sys.exit(0)
    except Exception:
        sys.exit(1)
