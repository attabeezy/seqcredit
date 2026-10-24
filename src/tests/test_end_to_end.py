"""Exercise the legacy synthetic pipeline with isolated generated data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqcredit.credit_model import CreditRiskDataLoader, HybridLSTMModel
from seqcredit.pipeline import build_user_feature_dataset
from seqcredit.synthesize import CalibratedMoMoDataGenerator


def test_pipeline_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQCREDIT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEQCREDIT_RAW_SEQ_FILE", str(tmp_path / "raw_sequences.npz"))
    transactions_dir = tmp_path / "user_transactions"
    features_file = tmp_path / "user_features.csv"
    labels_file = tmp_path / "user_labels.csv"
    generator = CalibratedMoMoDataGenerator(
        n_users=150,
        avg_transactions_per_user=20,
        output_dir=str(transactions_dir),
        seed=42,
    )
    generator.generate_dataset()
    labels = pd.read_csv(labels_file)
    labels["is_bad"] = (labels["credit_risk_label"] > 0).astype(int)
    labels.set_index("user_id")[["credit_risk_label", "is_bad"]].to_csv(labels_file)
    build_user_feature_dataset(
        transactions_dir=str(transactions_dir), output_path=str(features_file)
    )
    loader = CreditRiskDataLoader(
        features_path=str(features_file),
        summaries_path=str(labels_file),
        transactions_dir=str(transactions_dir),
    )
    splits = loader.prepare_static_splits()
    sequences = loader.load_sequences(
        max_seq_len=20, cache_path=str(tmp_path / "sequences.npz")
    )
    np.testing.assert_array_equal(sequences["y_train"], splits["y_train"])
    assert len(sequences["X_train_seq"]) == len(splits["X_train_scaled"])
    model = HybridLSTMModel(lstm_units_1=4, lstm_units_2=4, dense_units=4)
    model.build_model(
        seq_input_shape=sequences["X_train_seq"].shape[1:],
        static_input_dim=splits["X_train_scaled"].shape[1],
    )
    predictions = model.model(
        [sequences["X_train_seq"][:2], splits["X_train_scaled"][:2]],
        training=False,
    ).numpy()
    assert predictions.shape == (2, 1)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
