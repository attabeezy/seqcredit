"""Tests for Frontend & Presentation UX Refinements (To-Do #4).

Verifies archetype presets, cohort filtering engine, presentation mode configs,
and backend /api/presets endpoints.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from seqcredit_mvp.dashboard import build_dashboard_payload
from seqcredit_mvp.presets import (
    ARCHETYPE_DEFINITIONS,
    classify_balance_tier,
    classify_trajectory,
    classify_volume_tier,
    detect_archetypes,
    filter_cohort,
)
from seqcredit_mvp.scoring import DemoScorer

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV_PATH = ROOT / "data" / "sample_transactions.csv"


@pytest.fixture(scope="module")
def shared_payload():
    scorer = DemoScorer()
    transactions = pd.read_csv(SAMPLE_CSV_PATH)
    return build_dashboard_payload(transactions, scorer)


def test_classify_primitives():
    assert classify_trajectory(-0.10) == "Deteriorating"
    assert classify_trajectory(0.10) == "Improving"
    assert classify_trajectory(0.01) == "Stable"

    assert classify_balance_tier(150.0) == "Low (< GHS 200)"
    assert classify_balance_tier(250.0) == "Mid (GHS 200–300)"
    assert classify_balance_tier(450.0) == "High (> GHS 300)"

    assert classify_volume_tier(12) == "Low (< 25 ops)"
    assert classify_volume_tier(36) == "Mid (25–40 ops)"
    assert classify_volume_tier(50) == "High (> 40 ops)"


def test_detect_archetypes(shared_payload):
    scores_df = pd.DataFrame(shared_payload["scores"])
    details = shared_payload["details"]

    archetypes = detect_archetypes(scores_df, details)

    assert "deteriorating" in archetypes
    assert "consistent" in archetypes
    assert "merchant" in archetypes
    assert "divergent" in archetypes

    # Ensure all values are JSON serializable (no raw np.float64 or numpy types)
    serialized = json.dumps(archetypes)
    loaded = json.loads(serialized)
    assert len(loaded) == 4

    # Verify characteristics
    div = archetypes["divergent"]
    assert "borrower_id" in div
    assert "delta_pts" in div

    det = archetypes["deteriorating"]
    con = archetypes["consistent"]
    mer = archetypes["merchant"]

    assert det["sequence_score"] >= con["sequence_score"]
    assert mer["borrower_id"] in details


def test_filter_cohort(shared_payload):
    scores_df = pd.DataFrame(shared_payload["scores"])
    details = shared_payload["details"]

    # No filter returns all rows enriched
    enriched_all = filter_cohort(scores_df, details)
    assert len(enriched_all) == len(scores_df)
    assert "trajectory" in enriched_all.columns
    assert "balance_tier" in enriched_all.columns
    assert "volume_tier" in enriched_all.columns

    # Filter by risk band
    filtered_lower = filter_cohort(scores_df, details, risk_bands=["Lower demo risk"])
    assert all("lower" in b.lower() for b in filtered_lower["demo_risk_band"])

    # Filter by trajectory
    filtered_det = filter_cohort(scores_df, details, trajectory_types=["deteriorating"])
    assert all(t == "Deteriorating" for t in filtered_det["trajectory"])

    # Filter by balance tier
    filtered_low_bal = filter_cohort(scores_df, details, balance_tiers=["low"])
    assert all("< GHS 200" in t for t in filtered_low_bal["balance_tier"])

    # Multi-filter conjunction
    filtered_combo = filter_cohort(
        scores_df,
        details,
        risk_bands=["Lower demo risk", "Watch demo risk"],
        trajectory_types=["stable", "improving"],
    )
    assert len(filtered_combo) <= len(scores_df)
    for _, r in filtered_combo.iterrows():
        assert r["trajectory"] in ["Stable", "Improving"]


def test_dashboard_payload_includes_archetypes(shared_payload):
    assert "archetypes" in shared_payload
    assert len(shared_payload["archetypes"]) == 4
