import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from seqcredit_mvp.features import build_feature_tables
from seqcredit_mvp.dashboard import build_dashboard_payload
from seqcredit_mvp.scoring import DemoScorer
from seqcredit_mvp.synthetic import SyntheticConfig, generate_synthetic_cohort


def test_scorer_returns_demo_metadata_and_no_decision(tmp_path):
    transactions, labels = generate_synthetic_cohort(
        SyntheticConfig(n_borrowers=30, transactions_per_borrower=12, seed=11)
    )
    static, sequence = build_feature_tables(transactions)
    y = (labels.set_index("borrower_id").loc[static.index, "credit_risk_label"] == 2).astype(int)
    static_model = DummyClassifier(strategy="prior").fit(static, y)
    sequence_model = DummyClassifier(strategy="prior").fit(sequence, y)
    joblib.dump(static_model, tmp_path / "static_model.joblib")
    joblib.dump(sequence_model, tmp_path / "sequence_model.joblib")
    manifest = {
        "status": "passed",
        "model_version": "test-v1",
        "models": {
            "static": {"features": list(static.columns)},
            "sequence": {"features": list(sequence.columns)},
        },
        "demo_risk_band_thresholds": {"watch": 0.02, "elevated": 0.08},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    scores = DemoScorer(tmp_path).score(transactions)
    assert len(scores) == 30
    assert scores["synthetic_demo"].all()
    assert set(scores["decision"]) == {"No lending decision"}
    assert set(scores["model_version"]) == {"test-v1"}


def test_dashboard_payload_reconciles_scores_and_portfolio(tmp_path):
    transactions, labels = generate_synthetic_cohort(
        SyntheticConfig(n_borrowers=24, transactions_per_borrower=12, seed=13)
    )
    static, sequence = build_feature_tables(transactions)
    y = (labels.set_index("borrower_id").loc[static.index, "credit_risk_label"] == 2).astype(int)
    joblib.dump(DummyClassifier(strategy="prior").fit(static, y), tmp_path / "static_model.joblib")
    joblib.dump(DummyClassifier(strategy="prior").fit(sequence, y), tmp_path / "sequence_model.joblib")
    manifest = {
        "status": "passed",
        "notice": "Synthetic prototype only. Not for lending decisions.",
        "model_version": "test-v1",
        "created_at": "2026-08-31T00:00:00+00:00",
        "cohort": {"borrowers": 24, "transactions": len(transactions), "default_rate": 0.04},
        "sequence_gain_auc": 0.027,
        "models": {"static": {"features": list(static.columns)}, "sequence": {"features": list(sequence.columns)}},
        "demo_risk_band_thresholds": {"watch": 0.02, "elevated": 0.08},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    scorer = DemoScorer(tmp_path)
    benchmark_path = tmp_path / "benchmark.csv"
    pd.DataFrame(
        [{"evidence": "synthetic_demo", "model": "Test model", "auc_roc": 0.75}]
    ).to_csv(benchmark_path, index=False)
    payload = build_dashboard_payload(transactions, scorer, benchmark_path)
    assert payload["portfolio"]["borrowers"] == 24
    assert payload["portfolio"]["transactions"] == len(transactions)
    assert len(payload["scores"]) == 24
    assert len(payload["details"]) == 24
    assert 2 <= len(payload["showcase"]) <= 3
