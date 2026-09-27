"""Explainability and feature attribution engine for SeqCredit MVP.

Provides counterfactual marginal feature attribution, sequence vs. static
trajectory delta decomposition, and chart-ready waterfall data without
introducing heavy C-extension dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from seqcredit_mvp.scoring import DemoScorer


@dataclass(frozen=True)
class FeatureMeta:
    code: str
    display_name: str
    category: str  # "sequence" or "static"
    unit: str
    description: str


FEATURE_CATALOG: dict[str, FeatureMeta] = {
    # Static aggregate features (18)
    "transaction_count": FeatureMeta(
        code="transaction_count",
        display_name="Transaction Count",
        category="static",
        unit="txs",
        description="Total validated transactions in the evaluation window",
    ),
    "total_volume": FeatureMeta(
        code="total_volume",
        display_name="Total Turnover Volume",
        category="static",
        unit="GHS",
        description="Aggregate monetary turnover transacted across all operations",
    ),
    "avg_amount": FeatureMeta(
        code="avg_amount",
        display_name="Average Transaction Sizing",
        category="static",
        unit="GHS",
        description="Mean monetary value per transaction event",
    ),
    "median_amount": FeatureMeta(
        code="median_amount",
        display_name="Median Transaction Sizing",
        category="static",
        unit="GHS",
        description="50th percentile monetary value per transaction event",
    ),
    "std_amount": FeatureMeta(
        code="std_amount",
        display_name="Transaction Amount Volatility",
        category="static",
        unit="GHS",
        description="Standard deviation of transaction amounts",
    ),
    "max_amount": FeatureMeta(
        code="max_amount",
        display_name="Maximum Single Operation",
        category="static",
        unit="GHS",
        description="Highest single transaction amount in history",
    ),
    "pct_cashout": FeatureMeta(
        code="pct_cashout",
        display_name="Cash-Out Frequency Ratio",
        category="static",
        unit="%",
        description="Proportion of events withdrawing physical currency",
    ),
    "pct_transfer": FeatureMeta(
        code="pct_transfer",
        display_name="P2P Transfer Ratio",
        category="static",
        unit="%",
        description="Proportion of peer-to-peer transfer operations",
    ),
    "pct_debit": FeatureMeta(
        code="pct_debit",
        display_name="Debit Transaction Ratio",
        category="static",
        unit="%",
        description="Proportion of direct debit settlement operations",
    ),
    "pct_payment": FeatureMeta(
        code="pct_payment",
        display_name="Merchant Payment Ratio",
        category="static",
        unit="%",
        description="Proportion of commercial utility and merchant payments",
    ),
    "pct_night": FeatureMeta(
        code="pct_night",
        display_name="Off-Hours Activity Ratio",
        category="static",
        unit="%",
        description="Proportion of transactions executed between 22:00 and 06:00",
    ),
    "avg_balance": FeatureMeta(
        code="avg_balance",
        display_name="Average Account Liquidity",
        category="static",
        unit="GHS",
        description="Mean wallet balance observed across transaction points",
    ),
    "min_balance": FeatureMeta(
        code="min_balance",
        display_name="Minimum Reserve Liquidity",
        category="static",
        unit="GHS",
        description="Lowest recorded wallet balance in observation window",
    ),
    "balance_volatility": FeatureMeta(
        code="balance_volatility",
        display_name="Liquidity Dispersion",
        category="static",
        unit="GHS",
        description="Standard deviation of account wallet balance",
    ),
    "pct_low_balance": FeatureMeta(
        code="pct_low_balance",
        display_name="Depleted Liquidity Incident Rate",
        category="static",
        unit="%",
        description="Frequency of operations conducted below minimum reserve threshold",
    ),
    "total_fees": FeatureMeta(
        code="total_fees",
        display_name="Cumulative Network Tariffs",
        category="static",
        unit="GHS",
        description="Total mobile-money ecosystem fees and levies paid",
    ),
    "unique_recipients": FeatureMeta(
        code="unique_recipients",
        display_name="Counterparty Diversity",
        category="static",
        unit="entities",
        description="Distinct counterparty identifiers receiving disbursements",
    ),
    "avg_hours_between": FeatureMeta(
        code="avg_hours_between",
        display_name="Inter-Event Velocity",
        category="static",
        unit="hours",
        description="Mean elapsed hours between consecutive transaction events",
    ),
    # Order-aware sequence features (8)
    "amount_slope": FeatureMeta(
        code="amount_slope",
        display_name="Disbursement Sizing Trajectory",
        category="sequence",
        unit="slope",
        description="Linear trend direction and acceleration of transaction amounts over time",
    ),
    "balance_slope": FeatureMeta(
        code="balance_slope",
        display_name="Liquidity Depletion Trajectory",
        category="sequence",
        unit="slope",
        description="Linear trend direction of account balance over sequential events",
    ),
    "amount_recent_ratio": FeatureMeta(
        code="amount_recent_ratio",
        display_name="Recent Velocity Multiplier",
        category="sequence",
        unit="ratio",
        description="Ratio of recent transaction amounts compared to full historical average",
    ),
    "balance_recent_ratio": FeatureMeta(
        code="balance_recent_ratio",
        display_name="Recent Liquidity Coverage",
        category="sequence",
        unit="ratio",
        description="Ratio of recent balance levels compared to full historical average",
    ),
    "cashout_slope": FeatureMeta(
        code="cashout_slope",
        display_name="Cash-Out Acceleration",
        category="sequence",
        unit="slope",
        description="Rate of change in physical cash withdrawal frequency over time",
    ),
    "night_slope": FeatureMeta(
        code="night_slope",
        display_name="Off-Hours Velocity Shift",
        category="sequence",
        unit="slope",
        description="Trend acceleration in late-night transaction activity",
    ),
    "recent_cashout_rate": FeatureMeta(
        code="recent_cashout_rate",
        display_name="Recent Cash-Out Concentration",
        category="sequence",
        unit="%",
        description="Share of cash-out events concentrated in the most recent transaction quarter",
    ),
    "recent_low_balance_rate": FeatureMeta(
        code="recent_low_balance_rate",
        display_name="Recent Liquidity Strain Frequency",
        category="sequence",
        unit="%",
        description="Frequency of low-balance events occurring in the recent evaluation window",
    ),
}

SEQUENCE_FEATURE_KEYS = [
    "amount_slope",
    "balance_slope",
    "amount_recent_ratio",
    "balance_recent_ratio",
    "cashout_slope",
    "night_slope",
    "recent_cashout_rate",
    "recent_low_balance_rate",
]


@dataclass
class DriverAttribution:
    code: str
    display_name: str
    category: str
    unit: str
    description: str
    actual_value: float
    baseline_value: float
    impact_points: float  # scaled to 0-100 risk index points
    direction: str  # "escalator" (increases risk) or "mitigator" (decreases risk)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cohort_baseline(features_df: pd.DataFrame) -> pd.Series:
    """Compute the median feature profile across the cohort to serve as reference point."""
    return features_df.median(numeric_only=True)


def explain_borrower(
    scorer: DemoScorer,
    sequence_row: pd.Series,
    cohort_baseline: pd.Series,
    static_score: float | None = None,
    sequence_score: float | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Compute marginal feature attributions and sequence delta decomposition for a borrower.

    Args:
        scorer: DemoScorer instance holding fitted models and manifest.
        sequence_row: Feature series containing all sequence model features.
        cohort_baseline: Cohort reference median series.
        static_score: Optional precomputed static score (0-1).
        sequence_score: Optional precomputed sequence score (0-1).
        top_k: Number of top escalators and mitigators to extract.

    Returns:
        Structured dictionary with drivers, sequence delta breakdown, and waterfall series.
    """
    seq_cols = scorer.manifest["models"]["sequence"]["features"]
    borrower_df = pd.DataFrame([sequence_row[seq_cols].to_dict()])
    baseline_series = cohort_baseline[seq_cols]
    baseline_df = pd.DataFrame([baseline_series.to_dict()])

    # Compute baseline score and borrower sequence score
    if sequence_score is None:
        sequence_score = float(scorer.sequence_model.predict_proba(borrower_df)[:, 1][0])
    cohort_baseline_score = float(scorer.sequence_model.predict_proba(baseline_df)[:, 1][0])

    # Counterfactual perturbation: replace each feature j with cohort baseline value
    perturbed = pd.concat([borrower_df.copy() for _ in range(len(seq_cols))], ignore_index=True)
    for idx, col in enumerate(seq_cols):
        perturbed.loc[idx, col] = baseline_series[col]

    perturbed_scores = scorer.sequence_model.predict_proba(perturbed)[:, 1]

    drivers: list[DriverAttribution] = []
    for idx, col in enumerate(seq_cols):
        meta = FEATURE_CATALOG.get(
            col,
            FeatureMeta(
                code=col,
                display_name=col.replace("_", " ").title(),
                category="sequence" if col in SEQUENCE_FEATURE_KEYS else "static",
                unit="",
                description=col,
            ),
        )
        # Impact: positive means borrower's actual value increases the risk score
        marginal_diff = sequence_score - float(perturbed_scores[idx])
        # Convert to 100-point risk index space for institutional readability
        impact_pts = round(marginal_diff * 100.0, 2)
        direction = "escalator" if impact_pts >= 0 else "mitigator"

        drivers.append(
            DriverAttribution(
                code=col,
                display_name=meta.display_name,
                category=meta.category,
                unit=meta.unit,
                description=meta.description,
                actual_value=round(float(sequence_row[col]), 4),
                baseline_value=round(float(baseline_series[col]), 4),
                impact_points=impact_pts,
                direction=direction,
            )
        )

    # Separate escalators and mitigators
    escalators = sorted(
        [d for d in drivers if d.impact_points > 0.05],
        key=lambda d: d.impact_points,
        reverse=True,
    )[:top_k]

    mitigators = sorted(
        [d for d in drivers if d.impact_points < -0.05],
        key=lambda d: d.impact_points,
    )[:top_k]

    # --- Static vs. Sequence Delta Decomposition ---
    # Measure the joint impact of all 8 order-aware sequence trajectory features
    counterfactual_no_trajectory = borrower_df.copy()
    for seq_key in SEQUENCE_FEATURE_KEYS:
        if seq_key in counterfactual_no_trajectory.columns:
            counterfactual_no_trajectory.loc[0, seq_key] = baseline_series[seq_key]

    score_without_trajectory = float(
        scorer.sequence_model.predict_proba(counterfactual_no_trajectory)[:, 1][0]
    )
    trajectory_lift_prob = sequence_score - score_without_trajectory
    trajectory_lift_pts = round(trajectory_lift_prob * 100.0, 2)

    total_score_delta_pts = (
        round((sequence_score - static_score) * 100.0, 2) if static_score is not None else None
    )

    # Sequence-specific drivers breakdown
    sequence_drivers = sorted(
        [d for d in drivers if d.category == "sequence"],
        key=lambda d: abs(d.impact_points),
        reverse=True,
    )

    # --- Waterfall Data Generation ---
    # Step 1: Cohort Baseline Risk Index
    base_idx = round(cohort_baseline_score * 100.0, 1)
    final_idx = round(sequence_score * 100.0, 1)

    # Select top contributors for the waterfall chart
    top_contributors = sorted(drivers, key=lambda d: abs(d.impact_points), reverse=True)[:6]

    waterfall_items = []
    running_total = base_idx
    for item in top_contributors:
        waterfall_items.append(
            {
                "name": item.display_name,
                "impact": item.impact_points,
                "category": item.category,
                "actual": item.actual_value,
                "baseline": item.baseline_value,
                "unit": item.unit,
            }
        )
        running_total += item.impact_points

    residual = round(final_idx - running_total, 1)
    if abs(residual) >= 0.1:
        waterfall_items.append(
            {
                "name": "Other Combined Factors",
                "impact": residual,
                "category": "aggregate",
                "actual": 0.0,
                "baseline": 0.0,
                "unit": "",
            }
        )

    return {
        "cohort_baseline_index": base_idx,
        "borrower_sequence_index": final_idx,
        "total_delta_pts": total_score_delta_pts,
        "trajectory_lift_pts": trajectory_lift_pts,
        "top_escalators": [d.to_dict() for d in escalators],
        "top_mitigators": [d.to_dict() for d in mitigators],
        "sequence_drivers": [d.to_dict() for d in sequence_drivers],
        "waterfall": {
            "baseline_index": base_idx,
            "final_index": final_idx,
            "steps": waterfall_items,
        },
    }
