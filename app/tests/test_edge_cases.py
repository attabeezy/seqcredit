import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from seqcredit_mvp.dashboard import MAX_BORROWERS, MAX_ROWS, build_dashboard_payload
from seqcredit_mvp.features import _late_early_ratio, _slope, build_feature_tables
from seqcredit_mvp.scoring import DEFAULT_ARTIFACT_DIR, DemoScorer


@pytest.fixture
def mock_scorer(tmp_path):
    # Train dummy models matching standard schema
    sample_df = pd.DataFrame([
        {
            "borrower_id": f"B_{i}",
            "transaction_id": f"TX_{i}",
            "timestamp": "2026-03-01T10:00:00",
            "transaction_type": "TRANSFER",
            "amount": 50.0 + i,
            "balance_before": 200.0,
            "balance_after": 150.0,
            "fee": 1.0,
            "recipient_id": "R_1",
        }
        for i in range(5)
    ])
    static, sequence = build_feature_tables(sample_df)
    y = np.array([0, 0, 1, 0, 1])
    static_model = DummyClassifier(strategy="prior").fit(static, y)
    sequence_model = DummyClassifier(strategy="prior").fit(sequence, y)

    joblib.dump(static_model, tmp_path / "static_model.joblib")
    joblib.dump(sequence_model, tmp_path / "sequence_model.joblib")
    manifest = {
        "status": "passed",
        "notice": "Synthetic prototype only.",
        "model_version": "test-v1",
        "created_at": "2026-08-31T00:00:00+00:00",
        "cohort": {"borrowers": 5, "transactions": 5, "default_rate": 0.4},
        "sequence_gain_auc": 0.02,
        "models": {
            "static": {"features": list(static.columns)},
            "sequence": {"features": list(sequence.columns)},
        },
        "demo_risk_band_thresholds": {"watch": 0.3, "elevated": 0.6},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return DemoScorer(tmp_path)


def test_single_transaction_borrower_features():
    df = pd.DataFrame([
        {
            "borrower_id": "LONE_WOLF",
            "transaction_id": "TXN_001",
            "timestamp": "2026-01-01T12:00:00",
            "transaction_type": "CASH_OUT",
            "amount": 100.0,
            "balance_before": 250.0,
            "balance_after": 150.0,
            "fee": 2.0,
            "recipient_id": "REC_1",
        }
    ])
    static, sequence = build_feature_tables(df)
    assert len(static) == 1
    assert len(sequence) == 1
    # Check that none of the columns have NaN
    assert not static.isna().any().any()
    assert not sequence.isna().any().any()
    # Slopes should be 0.0
    assert sequence.loc["LONE_WOLF", "amount_slope"] == 0.0
    assert sequence.loc["LONE_WOLF", "balance_slope"] == 0.0
    # Ratios should be 1.0
    assert sequence.loc["LONE_WOLF", "amount_recent_ratio"] == 1.0
    assert sequence.loc["LONE_WOLF", "balance_recent_ratio"] == 1.0


def test_single_borrower_dashboard_payload(mock_scorer, tmp_path):
    df = pd.DataFrame([
        {
            "borrower_id": "SOLO_1",
            "transaction_id": "TXN_001",
            "timestamp": "2026-01-01T12:00:00",
            "transaction_type": "CASH_OUT",
            "amount": 100.0,
            "balance_before": 250.0,
            "balance_after": 150.0,
            "fee": 2.0,
            "recipient_id": "REC_1",
        },
        {
            "borrower_id": "SOLO_1",
            "transaction_id": "TXN_002",
            "timestamp": "2026-01-02T12:00:00",
            "transaction_type": "TRANSFER",
            "amount": 50.0,
            "balance_before": 150.0,
            "balance_after": 100.0,
            "fee": 0.5,
            "recipient_id": "REC_2",
        }
    ])
    benchmark_path = tmp_path / "benchmark.csv"
    pd.DataFrame([{"evidence": "synthetic_demo", "model": "Test", "auc_roc": 0.75}]).to_csv(
        benchmark_path, index=False
    )
    payload = build_dashboard_payload(df, mock_scorer, benchmark_path)
    assert payload["portfolio"]["borrowers"] == 1
    assert payload["portfolio"]["transactions"] == 2
    assert len(payload["scores"]) == 1
    assert len(payload["showcase"]) == 1
    assert payload["showcase"][0]["borrower_id"] == "SOLO_1"
    assert "SOLO_1" in payload["details"]


def test_max_rows_limit_exceeded(mock_scorer):
    # Mocking rows above MAX_ROWS
    large_df = pd.DataFrame({
        "borrower_id": ["B1"] * 2,
        "transaction_id": ["TX1", "TX2"],
        "timestamp": ["2026-01-01T10:00:00", "2026-01-01T11:00:00"],
        "transaction_type": ["TRANSFER", "DEBIT"],
        "amount": [10.0, 20.0],
        "balance_before": [100.0, 90.0],
        "balance_after": [90.0, 70.0],
        "fee": [0.0, 0.0],
        "recipient_id": ["R1", "R2"],
    })
    # We test with MAX_ROWS monkeypatched or directly
    from seqcredit_mvp import dashboard
    original_max = dashboard.MAX_ROWS
    try:
        dashboard.MAX_ROWS = 1
        with pytest.raises(ValueError, match="at most 1 transaction rows"):
            build_dashboard_payload(large_df, mock_scorer)
    finally:
        dashboard.MAX_ROWS = original_max


def test_max_borrowers_limit_exceeded(mock_scorer):
    from seqcredit_mvp import dashboard
    two_borrowers_df = pd.DataFrame({
        "borrower_id": ["B1", "B2"],
        "transaction_id": ["TX1", "TX2"],
        "timestamp": ["2026-01-01T10:00:00", "2026-01-01T11:00:00"],
        "transaction_type": ["TRANSFER", "DEBIT"],
        "amount": [10.0, 20.0],
        "balance_before": [100.0, 90.0],
        "balance_after": [90.0, 70.0],
        "fee": [0.0, 0.0],
        "recipient_id": ["R1", "R2"],
    })
    original_max = dashboard.MAX_BORROWERS
    try:
        dashboard.MAX_BORROWERS = 1
        with pytest.raises(ValueError, match="at most 1 borrowers"):
            build_dashboard_payload(two_borrowers_df, mock_scorer)
    finally:
        dashboard.MAX_BORROWERS = original_max


def test_slope_and_ratio_helpers():
    # Constant series should have 0 slope
    assert _slope(pd.Series([5.0, 5.0, 5.0])) == 0.0
    # Single element series should have 0 slope
    assert _slope(pd.Series([10.0])) == 0.0
    # Strictly increasing series should have positive slope
    assert _slope(pd.Series([1.0, 2.0, 3.0, 4.0])) > 0.0
    # Strictly decreasing series should have negative slope
    assert _slope(pd.Series([4.0, 3.0, 2.0, 1.0])) < 0.0
