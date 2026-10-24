"""Shared pytest fixtures for the seqcredit test suite.

Keep test datasets tiny (tens of users, not thousands) so the suite runs in
seconds rather than minutes. Deep-learning model tests should use minimal
units/epochs — they only need to prove the code path works, not converge.
"""
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from seqcredit.synthesize import CalibratedMoMoDataGenerator
from seqcredit.pipeline import build_user_feature_dataset

@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point config.DATA_DIR (via SEQCREDIT_DATA_DIR) at a throwaway directory.

    Must be set before `seqcredit.config` is imported for the first
    time in a process, since config.py resolves DATA_DIR at import time.
    Tests that need this should import seqcredit modules lazily
    (inside the test function) after requesting this fixture, or accept
    that config.DATA_DIR reflects whatever was set at first import.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("SEQCREDIT_DATA_DIR", str(data_dir))
    return data_dir


@pytest.fixture
def tiny_synthetic_dataset(tmp_path):
    """Generate a very small synthetic transactions + labels dataset on disk.

    Returns a dict of paths: transactions_dir, user_labels_file, user_features_file.
    """
    data_dir = tmp_path / "data"
    transactions_dir = data_dir / "user_transactions"
    user_features_file = data_dir / "user_features.csv"
    user_labels_file = data_dir / "user_labels.csv"

    generator = CalibratedMoMoDataGenerator(
        n_users=40,
        avg_transactions_per_user=30,
        output_dir=str(transactions_dir),
        seed=42,
    )
    generator.generate_dataset()

    build_user_feature_dataset(
        transactions_dir=str(transactions_dir),
        output_path=str(user_features_file),
    )

    labels_df = pd.read_csv(data_dir / "user_labels.csv")
    labels_df["is_bad"] = (labels_df["credit_risk_label"] > 0).astype(int)
    labels_df.set_index("user_id", inplace=True)
    labels_df[["credit_risk_label", "is_bad"]].to_csv(user_labels_file)

    return {
        "data_dir": data_dir,
        "transactions_dir": transactions_dir,
        "user_features_file": user_features_file,
        "user_labels_file": user_labels_file,
    }
