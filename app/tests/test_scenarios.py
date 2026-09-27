import numpy as np
import pandas as pd
import pytest

from seqcredit_mvp.features import build_feature_tables
from seqcredit_mvp.scenarios import (
    DEFAULT_TENURES,
    TENURE_CONFIGS,
    apply_cashflow_stress,
    compute_borrower_liquidity,
    evaluate_borrower_stress,
    scenario_index,
    simulate_multi_term_curves,
    simulate_policy_sandbox,
    simulate_term_sizing,
)
from seqcredit_mvp.scoring import DemoScorer
from seqcredit_mvp.synthetic import SyntheticConfig, generate_synthetic_cohort


def test_tenure_configs_complete():
    for t in [7, 14, 30]:
        assert t in TENURE_CONFIGS
        cfg = TENURE_CONFIGS[t]
        assert "name" in cfg
        assert "flat_fee_rate" in cfg
        assert "tenure_stress_factor" in cfg
        assert cfg["flat_fee_rate"] > 0
    # Shorter tenures have higher stress factors
    assert TENURE_CONFIGS[7]["tenure_stress_factor"] > TENURE_CONFIGS[14]["tenure_stress_factor"]
    assert TENURE_CONFIGS[14]["tenure_stress_factor"] > TENURE_CONFIGS[30]["tenure_stress_factor"]


def test_compute_borrower_liquidity():
    indicators = {
        "avg_balance": 200.0,
        "avg_amount": 50.0,
        "transaction_count": 30,
        "pct_cashout": 0.20,
    }
    liq = compute_borrower_liquidity(indicators)
    assert liq["avg_balance"] == 200.0
    assert liq["avg_amount"] == 50.0
    assert liq["daily_net_cashflow"] > 0
    assert liq["absorption_capacity"] == 400.0  # 200 + 50 * 4


def test_scenario_index_properties():
    # Bounds: always between 0 and 1
    idx = scenario_index(0.10, 250, 500, tenure_days=30)
    assert 0.0 < idx < 1.0

    # Monotonicity with amount: larger loan -> higher risk
    idx_small = scenario_index(0.10, 100, 500, tenure_days=30)
    idx_large = scenario_index(0.10, 800, 500, tenure_days=30)
    assert idx_large > idx_small

    # Monotonicity with base score
    idx_low_base = scenario_index(0.05, 250, 500, tenure_days=30)
    idx_high_base = scenario_index(0.40, 250, 500, tenure_days=30)
    assert idx_high_base > idx_low_base

    # Term pressure: 7-day loan exerts higher acute stress than 30-day loan
    idx_7d = scenario_index(0.10, 250, 500, tenure_days=7)
    idx_14d = scenario_index(0.10, 250, 500, tenure_days=14)
    idx_30d = scenario_index(0.10, 250, 500, tenure_days=30)
    assert idx_7d > idx_14d > idx_30d


def test_simulate_term_sizing():
    indicators = {
        "avg_balance": 300.0,
        "avg_amount": 60.0,
        "transaction_count": 36,
        "pct_cashout": 0.15,
    }
    res_7d = simulate_term_sizing(base_score=0.12, principal=300, tenure_days=7, indicators=indicators)
    res_30d = simulate_term_sizing(base_score=0.12, principal=300, tenure_days=30, indicators=indicators)

    # 7-day fee is 5%, 30-day fee is 14%
    assert res_7d.fee_rate == 0.05
    assert res_7d.total_fee == 15.0
    assert res_7d.total_repayment == 315.0
    assert res_30d.fee_rate == 0.14
    assert res_30d.total_fee == 42.0
    assert res_30d.total_repayment == 342.0

    # Daily debt service: 7-day is much higher daily burden
    assert res_7d.daily_debt_service == round(315.0 / 7.0, 2)
    assert res_30d.daily_debt_service == round(342.0 / 30.0, 2)
    assert res_7d.daily_debt_service > res_30d.daily_debt_service

    # DSTC ratio: 7-day is much higher ratio
    assert res_7d.dstc_ratio > res_30d.dstc_ratio
    assert res_7d.risk_assessment in [
        "Severe Debt Service Strain (DSTC > 125%)",
        "Elevated Debt Service Pressure (DSTC 75–125%)",
        "Moderate Liquidity Absorption (DSTC 40–75%)",
        "Well-Supported Liquidity (DSTC < 40%)",
    ]


def test_simulate_multi_term_curves():
    amounts = [25, 50, 100, 200]
    curves = simulate_multi_term_curves(base_score=0.15, capacity=500.0, amounts_range=amounts)
    assert set(curves.keys()) == {7, 14, 30}
    for t in [7, 14, 30]:
        assert len(curves[t]) == len(amounts)
        # Each curve increases with loan amount
        assert curves[t][-1] > curves[t][0]
    # For any amount, 7d risk > 30d risk
    for i in range(len(amounts)):
        assert curves[7][i] > curves[30][i]


def test_apply_cashflow_stress():
    raw = pd.Series(
        {
            "total_volume": 1000.0,
            "avg_amount": 50.0,
            "median_amount": 45.0,
            "avg_balance": 300.0,
            "min_balance": 50.0,
            "pct_low_balance": 0.10,
            "balance_volatility": 25.0,
            "total_fees": 20.0,
            "avg_hours_between": 12.0,
            "balance_slope": 0.05,
            "amount_slope": 0.02,
            "balance_recent_ratio": 1.10,
            "amount_recent_ratio": 1.05,
            "recent_low_balance_rate": 0.05,
            "recent_cashout_rate": 0.10,
            "transaction_count": 20,
        }
    )
    stressed = apply_cashflow_stress(
        raw,
        inbound_drop_pct=0.25,
        fee_surge_pct=0.40,
        inactivity_days=10.0,
    )

    # Volume and balance contracted
    assert stressed["total_volume"] < raw["total_volume"]
    assert stressed["avg_balance"] < raw["avg_balance"]
    assert stressed["min_balance"] < raw["min_balance"]
    # Low balance incidents and fees increased
    assert stressed["pct_low_balance"] > raw["pct_low_balance"]
    assert stressed["total_fees"] > raw["total_fees"]
    # Sequence order dynamics degraded
    assert stressed["avg_hours_between"] > raw["avg_hours_between"]
    assert stressed["balance_slope"] < raw["balance_slope"]
    assert stressed["balance_recent_ratio"] < raw["balance_recent_ratio"]


def test_evaluate_borrower_stress_integration():
    scorer = DemoScorer()
    transactions, _ = generate_synthetic_cohort(
        SyntheticConfig(n_borrowers=8, transactions_per_borrower=15, seed=123)
    )
    static, sequence = build_feature_tables(transactions)
    b_id = sequence.index[0]

    eval_result = evaluate_borrower_stress(
        scorer=scorer,
        sequence_row=sequence.loc[b_id],
        static_row=static.loc[b_id],
        inbound_drop_pct=0.30,
        fee_surge_pct=0.40,
        inactivity_days=10.0,
    )

    assert "baseline" in eval_result
    assert "stressed" in eval_result
    assert "delta_pts" in eval_result
    assert "band_shifted" in eval_result

    # Adverse stress should raise or preserve risk score
    assert eval_result["stressed"]["sequence_score"] >= eval_result["baseline"]["sequence_score"] - 1e-4
    assert eval_result["delta_pts"] >= -0.01


def test_simulate_policy_sandbox_conservation():
    # Construct synthetic score distribution
    n = 200
    rng = np.random.default_rng(42)
    scores = rng.beta(2, 20, size=n)  # realistic credit score distribution
    df = pd.DataFrame(
        {
            "borrower_id": [f"B_{i:03d}" for i in range(n)],
            "sequence_demo_score": scores,
            "static_demo_score": scores,
        }
    )

    watch_th = 0.035
    elev_th = 0.090
    sim = simulate_policy_sandbox(
        scores_df=df,
        watch_threshold=watch_th,
        elevated_threshold=elev_th,
        baseline_watch=0.030,
        baseline_elevated=0.085,
    )

    acc = sim["acceptance"]
    # Portfolio count conservation
    assert acc["full_approval_count"] + acc["conditional_review_count"] + acc["decline_count"] == n
    assert round(acc["full_approval_rate"] + acc["conditional_review_rate"] + acc["decline_rate"], 4) == 1.0000

    # Migration matrix cell conservation
    mat = sim["migration_matrix"]
    mat_sum = sum(sum(row.values()) for row in mat.values())
    assert mat_sum == n

    # Migration summary sum
    mig = sim["migration_summary"]
    assert mig["upgrades"] + mig["unchanged"] + mig["downgrades"] == n


def test_simulate_policy_sandbox_with_ground_truth():
    n = 100
    df = pd.DataFrame(
        {
            "borrower_id": [f"B_{i:03d}" for i in range(n)],
            "sequence_demo_score": [0.01 * (i + 1) for i in range(n)],
        }
    )
    # 5 ground truth defaults in the highest score range
    labels = pd.Series(
        {f"B_{i:03d}": 2 if i >= 95 else 0 for i in range(n)}
    )

    sim = simulate_policy_sandbox(
        scores_df=df,
        watch_threshold=0.50,
        elevated_threshold=0.80,
        ground_truth_labels=labels,
    )

    dc = sim["default_capture"]
    assert dc["has_empirical_ground_truth"] is True
    assert dc["total_defaults"] == 5
    # All 5 defaults are in the top scores (> 0.95), which is >= 0.80 (elevated)
    assert dc["captured_count"] == 5
    assert dc["capture_rate"] == 1.0


def test_simulate_policy_sandbox_threshold_bounds():
    df = pd.DataFrame(
        {
            "borrower_id": ["B1", "B2", "B3"],
            "sequence_demo_score": [0.02, 0.05, 0.10],
        }
    )
    # If inverted thresholds passed, function should gracefully handle min/max
    sim = simulate_policy_sandbox(df, watch_threshold=0.08, elevated_threshold=0.03)
    assert sim["thresholds"]["watch"] == 0.03
    assert sim["thresholds"]["elevated"] == 0.08
