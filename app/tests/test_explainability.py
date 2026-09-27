from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from seqcredit_mvp.dashboard import build_dashboard_payload
from seqcredit_mvp.explainability import (
    FEATURE_CATALOG,
    SEQUENCE_FEATURE_KEYS,
    compute_cohort_baseline,
    explain_borrower,
)
from seqcredit_mvp.features import build_feature_tables
from seqcredit_mvp.scoring import DemoScorer
from seqcredit_mvp.streamlit_app import (
    build_drivers_divergence_echarts_options,
    build_waterfall_echarts_options,
)
from seqcredit_mvp.synthetic import SyntheticConfig, generate_synthetic_cohort

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_transactions.csv"


def test_feature_catalog_covers_all_manifest_features():
    scorer = DemoScorer()
    static_feats = scorer.manifest["models"]["static"]["features"]
    sequence_feats = scorer.manifest["models"]["sequence"]["features"]

    for feat in static_feats:
        assert feat in FEATURE_CATALOG, f"Missing static feature in catalog: {feat}"
        assert FEATURE_CATALOG[feat].category == "static"

    for feat in sequence_feats:
        assert feat in FEATURE_CATALOG, f"Missing sequence feature in catalog: {feat}"
        if feat in SEQUENCE_FEATURE_KEYS:
            assert FEATURE_CATALOG[feat].category == "sequence"


def test_compute_cohort_baseline():
    data = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0],
            "balance": [100.0, 200.0, 300.0],
            "borrower_id": ["b1", "b2", "b3"],
        }
    )
    baseline = compute_cohort_baseline(data)
    assert baseline["amount"] == 20.0
    assert baseline["balance"] == 200.0


def test_explain_borrower_contract():
    scorer = DemoScorer()
    transactions, _ = generate_synthetic_cohort(
        SyntheticConfig(n_borrowers=10, transactions_per_borrower=12, seed=42)
    )
    static, sequence = build_feature_tables(transactions)
    baseline = compute_cohort_baseline(sequence)

    b_id = sequence.index[0]
    attribution = explain_borrower(
        scorer=scorer,
        sequence_row=sequence.loc[b_id],
        cohort_baseline=baseline,
        static_score=0.15,
        sequence_score=0.22,
        top_k=4,
    )

    # Check contract keys
    assert "cohort_baseline_index" in attribution
    assert "borrower_sequence_index" in attribution
    assert "trajectory_lift_pts" in attribution
    assert "total_delta_pts" in attribution
    assert "top_escalators" in attribution
    assert "top_mitigators" in attribution
    assert "sequence_drivers" in attribution
    assert "waterfall" in attribution

    assert attribution["borrower_sequence_index"] == round(0.22 * 100, 1)
    assert attribution["total_delta_pts"] == round((0.22 - 0.15) * 100, 2)

    # Escalators verification
    for esc in attribution["top_escalators"]:
        assert esc["impact_points"] >= 0
        assert esc["direction"] == "escalator"
        assert "display_name" in esc
        assert "actual_value" in esc
        assert "baseline_value" in esc

    # Mitigators verification
    for mit in attribution["top_mitigators"]:
        assert mit["impact_points"] <= 0
        assert mit["direction"] == "mitigator"
        assert "display_name" in mit

    # Waterfall verification
    wf = attribution["waterfall"]
    assert "baseline_index" in wf
    assert "final_index" in wf
    assert len(wf["steps"]) > 0


def test_echarts_waterfall_and_divergence_builders():
    sample_waterfall = {
        "baseline_index": 15.0,
        "final_index": 22.5,
        "steps": [
            {
                "name": "Liquidity Depletion Trajectory",
                "impact": 5.0,
                "category": "sequence",
                "actual": -0.3,
                "baseline": -0.05,
                "unit": "slope",
            },
            {
                "name": "Average Account Liquidity",
                "impact": -2.5,
                "category": "static",
                "actual": 450.0,
                "baseline": 250.0,
                "unit": "GHS",
            },
            {
                "name": "Other Combined Factors",
                "impact": 5.0,
                "category": "aggregate",
                "actual": 0.0,
                "baseline": 0.0,
                "unit": "",
            },
        ],
    }
    wf_opts = build_waterfall_echarts_options(sample_waterfall)
    assert "series" in wf_opts
    assert len(wf_opts["series"]) == 2  # base + impact
    assert wf_opts["series"][0]["name"] == "Base"
    assert wf_opts["series"][1]["name"] == "Attribution Impact"
    assert "xAxis" in wf_opts
    assert len(wf_opts["xAxis"]["data"]) == 5  # baseline + 3 steps + final

    sample_drivers = [
        {"display_name": "Liquidity Slope", "impact_points": 4.5},
        {"display_name": "Cash Reserve", "impact_points": -3.2},
    ]
    div_opts = build_drivers_divergence_echarts_options(sample_drivers)
    assert "series" in div_opts
    assert len(div_opts["series"]) == 1
    assert div_opts["yAxis"]["data"] == ["Cash Reserve", "Liquidity Slope"]


def test_dashboard_payload_includes_attribution():
    if not SAMPLE_CSV.exists():
        pytest.skip("sample_transactions.csv does not exist")
    scorer = DemoScorer()
    tx_df = pd.read_csv(SAMPLE_CSV)
    payload = build_dashboard_payload(tx_df, scorer)

    assert "details" in payload
    first_b_id = next(iter(payload["details"]))
    borrower_detail = payload["details"][first_b_id]

    assert "attribution" in borrower_detail
    attr = borrower_detail["attribution"]
    assert "cohort_baseline_index" in attr
    assert "trajectory_lift_pts" in attr
    assert "waterfall" in attr
    assert len(attr["sequence_drivers"]) == 8  # all 8 sequence features
