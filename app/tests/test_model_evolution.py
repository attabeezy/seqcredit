"""Unit and integration tests for Model Evolution, Archetypes, Calibration, and Sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqcredit_mvp.calibrate import (
    compute_brier_decomposition,
    compute_calibration_curve,
)
from seqcredit_mvp.dashboard import DEFAULT_CALIBRATION, build_dashboard_payload
from seqcredit_mvp.scenarios import evaluate_window_sensitivity
from seqcredit_mvp.scoring import DemoScorer
from seqcredit_mvp.synthetic import (
    MFS_ARCHETYPES,
    SyntheticConfig,
    generate_mfs_archetype_sample,
    generate_synthetic_cohort,
)


def test_mfs_archetypes_generation() -> None:
    """Verify that MFS archetype generation creates non-empty, schema-valid data across archetypes."""
    txs, labels = generate_mfs_archetype_sample(n_borrowers=12, transactions_per_borrower=18, seed=42)
    assert not txs.empty
    assert len(labels) == 2 * len(MFS_ARCHETYPES)
    assert len(txs) == len(labels) * 18
    assert "archetype" in txs.columns
    assert set(txs["archetype"].unique()) == set(MFS_ARCHETYPES)
    assert (txs["amount"] >= 0).all()
    assert (txs["balance_before"] >= 0).all()
    assert (txs["balance_after"] >= 0).all()
    assert (txs["fee"] >= 0).all()
    assert "mfs_archetype" in labels.columns


def test_synthetic_cohort_with_mfs_flag() -> None:
    """Verify generate_synthetic_cohort with enable_mfs_archetypes flag."""
    cfg = SyntheticConfig(n_borrowers=20, transactions_per_borrower=10, enable_mfs_archetypes=True, seed=123)
    txs, labels = generate_synthetic_cohort(cfg)
    assert len(txs) == 200
    assert len(labels) == 20
    assert "archetype" in txs.columns
    assert set(txs["borrower_id"].unique()) == set(labels["borrower_id"].unique())


def test_compute_calibration_curve() -> None:
    """Verify reliability diagram calibration curve calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.12, 0.22, 0.35, 0.45, 0.55, 0.65, 0.78, 0.88, 0.95])
    curve = compute_calibration_curve(y_true, y_prob, n_bins=5)

    assert "mean_predicted_value" in curve
    assert "fraction_of_positives" in curve
    assert "bin_counts" in curve
    assert len(curve["mean_predicted_value"]) == len(curve["fraction_of_positives"])
    for p in curve["mean_predicted_value"]:
        assert 0.0 <= p <= 1.0
    for f in curve["fraction_of_positives"]:
        assert 0.0 <= f <= 1.0


def test_compute_brier_decomposition() -> None:
    """Verify Murphy (1973) Brier score resolution into reliability, resolution, and uncertainty."""
    y_true = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.8, 0.9])
    decomp = compute_brier_decomposition(y_true, y_prob, n_bins=5)

    assert "total_brier" in decomp
    assert "total" in decomp
    assert "binned_brier" in decomp
    assert "reliability" in decomp
    assert "resolution" in decomp
    assert "uncertainty" in decomp

    # Murphy (1973) identity: Brier = Reliability - Resolution + Uncertainty
    expected_brier = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    assert abs(decomp["binned_brier"] - expected_brier) < 1e-4
    assert decomp["reliability"] >= 0.0
    assert decomp["resolution"] >= 0.0
    assert decomp["uncertainty"] >= 0.0


def test_evaluate_window_sensitivity() -> None:
    """Verify observation window depth sensitivity across sequence lengths."""
    scorer = DemoScorer()
    txs, labels = generate_mfs_archetype_sample(n_borrowers=4, transactions_per_borrower=36, seed=99)
    b_id = str(txs["borrower_id"].iloc[0])
    b_txs = txs[txs["borrower_id"] == b_id]

    res = evaluate_window_sensitivity(scorer, b_txs, window_sizes=[10, 20, 30, 36])
    assert res["borrower_id"] == b_id
    assert res["total_transactions"] == 36
    points = res["window_points"]
    assert len(points) == 4
    for pt in points:
        assert pt["window_size"] in [10, 20, 30, 36]
        assert 0.0 <= pt["sequence_score"] <= 1.0
        assert 0.0 <= pt["static_score"] <= 1.0
        assert "score_delta" in pt
        assert pt["risk_band"] in ["Lower demo risk", "Watch demo risk", "Elevated demo risk"]

    summ = res["summary"]
    assert summ["evaluated_windows"] == [10, 20, 30, 36]
    assert summ["score_range"] >= 0.0
    assert summ["max_abs_divergence"] >= 0.0


def test_demo_scorer_versioning_and_discovery() -> None:
    """Verify model contract version loading and error on non-existent version."""
    scorer_v1 = DemoScorer()
    assert scorer_v1.manifest["model_version"] == "synthetic-demo-v1"

    available = DemoScorer.available_versions()
    assert "synthetic-demo-v1" in available

    # Loading non-existent version raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DemoScorer(version="non-existent-version-99")


def test_dashboard_payload_includes_calibration_and_versions() -> None:
    """Verify build_dashboard_payload exposes calibration data and available versions."""
    scorer = DemoScorer()
    txs, _ = generate_mfs_archetype_sample(n_borrowers=4, transactions_per_borrower=12, seed=7)
    payload = build_dashboard_payload(txs, scorer)

    assert "calibration" in payload
    cal = payload["calibration"]
    assert "calibration_curve" in cal
    assert "brier_decomposition" in cal
    assert "available_versions" in payload
    assert "synthetic-demo-v1" in payload["available_versions"]
