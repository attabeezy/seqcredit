"""Behavioral archetype presets and cohort filtering engine for SeqCredit.

Provides detection of canonical behavioral archetypes in transaction cohorts
and multi-criteria filtering across trajectory, liquidity, and volume dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArchetypeDefinition:
    key: str
    title: str
    badge_label: str
    icon: str
    description: str
    narrative: str


ARCHETYPE_DEFINITIONS: dict[str, ArchetypeDefinition] = {
    "deteriorating": ArchetypeDefinition(
        key="deteriorating",
        title="High-Risk Deteriorating Counterparty",
        badge_label="High-Risk Deteriorating",
        icon="trending_down",
        description="Sharp liquidity degradation, rapid balance depletion, and escalating cash-out velocity.",
        narrative="This counterparty demonstrates acute temporal distress: while initial balances appeared adequate, recent transaction velocity and cash-out concentration have accelerated downward, triggering supervisory risk escalation under sequence modeling.",
    ),
    "consistent": ArchetypeDefinition(
        key="consistent",
        title="Low-Risk Consistent Counterparty",
        badge_label="Low-Risk Consistent",
        icon="verified_user",
        description="Disciplined cash-flow turnover, resilient retained balances, and low cash-out friction.",
        narrative="This counterparty exhibits predictable, sustainable turnover with positive or stable retained liquidity buffers. The temporal order confirms steady replenishment, placing them comfortably in the prime approval tier.",
    ),
    "merchant": ArchetypeDefinition(
        key="merchant",
        title="High-Volume Volatile Merchant",
        badge_label="High-Volume Merchant",
        icon="storefront",
        description="Rapid working-capital churn, elevated disbursement velocity, and high transaction density.",
        narrative="A commercial merchant pattern characterized by rapid daily transactions, wide turnover swings, and substantial aggregate volume. Temporal evaluation separates operational volatility from insolvency risk.",
    ),
    "divergent": ArchetypeDefinition(
        key="divergent",
        title="Order-Aware Divergent Counterparty",
        badge_label="Static vs. Sequence Divergence",
        icon="compare_arrows",
        description="Aggregate static indicators mask underlying temporal contraction detected only by sequence surrogate.",
        narrative="The signature showcase case: static aggregate models score this borrower favorably due to robust cumulative historical totals, but order-aware sequence evaluation detects recent sharp trajectory deterioration, yielding a major model divergence.",
    ),
}


def detect_archetypes(
    scores_df: pd.DataFrame,
    details_dict: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Identify the optimal representative borrower ID for each behavioral archetype.

    Returns a mapping from archetype key to borrower metadata including ID, score, and rationale.
    """
    if scores_df.empty or not details_dict:
        return {}

    borrower_metrics: list[dict[str, Any]] = []
    for _, row in scores_df.iterrows():
        b_id = str(row["borrower_id"])
        detail = details_dict.get(b_id)
        if not detail:
            continue
        ind = detail["indicators"]
        seq_score = float(row["sequence_demo_score"])
        stat_score = float(row["static_demo_score"])
        delta = seq_score - stat_score

        borrower_metrics.append(
            {
                "borrower_id": b_id,
                "sequence_score": seq_score,
                "static_score": stat_score,
                "delta": delta,
                "risk_band": str(row["demo_risk_band"]),
                "balance_slope": float(ind.get("balance_slope", 0.0)),
                "amount_slope": float(ind.get("amount_slope", 0.0)),
                "cashout_slope": float(ind.get("cashout_slope", 0.0)),
                "pct_cashout": float(ind.get("pct_cashout", 0.0)),
                "avg_balance": float(ind.get("avg_balance", 0.0)),
                "avg_amount": float(ind.get("avg_amount", 0.0)),
                "tx_count": int(ind.get("transaction_count", 0)),
            }
        )

    if not borrower_metrics:
        return {}

    metrics_df = pd.DataFrame(borrower_metrics)

    assigned: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()

    # 1. Divergent: Maximum positive delta (sequence score higher than static score)
    div_candidates = metrics_df.sort_values(by="delta", ascending=False)
    for _, c in div_candidates.iterrows():
        b_id = str(c["borrower_id"])
        if b_id not in used_ids:
            assigned["divergent"] = {
                **ARCHETYPE_DEFINITIONS["divergent"].__dict__,
                "borrower_id": b_id,
                "sequence_score": float(c["sequence_score"]),
                "static_score": float(c["static_score"]),
                "delta_pts": float(round(c["delta"] * 100, 1)),
                "key_metric": f"{'+' if c['delta'] >= 0 else ''}{c['delta']*100:.1f} pts sequence delta",
            }
            used_ids.add(b_id)
            break

    # 2. High-Risk Deteriorating: High sequence score with negative balance slope / elevated cashout
    det_candidates = metrics_df[~metrics_df["borrower_id"].isin(used_ids)].copy()
    if not det_candidates.empty:
        # Score = sequence_score - balance_slope + pct_cashout
        det_candidates["det_rank"] = (
            det_candidates["sequence_score"]
            - 0.5 * det_candidates["balance_slope"]
            + 0.5 * det_candidates["pct_cashout"]
        )
        best_det = det_candidates.sort_values(by="det_rank", ascending=False).iloc[0]
        b_id = str(best_det["borrower_id"])
        assigned["deteriorating"] = {
            **ARCHETYPE_DEFINITIONS["deteriorating"].__dict__,
            "borrower_id": b_id,
            "sequence_score": float(best_det["sequence_score"]),
            "static_score": float(best_det["static_score"]),
            "delta_pts": float(round(best_det["delta"] * 100, 1)),
            "key_metric": f"Risk: {best_det['sequence_score']*100:.0f}/100 · Slope: {best_det['balance_slope']:.2f}",
        }
        used_ids.add(b_id)

    # 3. Low-Risk Consistent: Low sequence score with positive/stable balance slope
    con_candidates = metrics_df[~metrics_df["borrower_id"].isin(used_ids)].copy()
    if not con_candidates.empty:
        con_candidates["con_rank"] = (
            -con_candidates["sequence_score"]
            + 0.5 * con_candidates["balance_slope"]
            - 0.3 * con_candidates["pct_cashout"]
        )
        best_con = con_candidates.sort_values(by="con_rank", ascending=False).iloc[0]
        b_id = str(best_con["borrower_id"])
        assigned["consistent"] = {
            **ARCHETYPE_DEFINITIONS["consistent"].__dict__,
            "borrower_id": b_id,
            "sequence_score": float(best_con["sequence_score"]),
            "static_score": float(best_con["static_score"]),
            "delta_pts": float(round(best_con["delta"] * 100, 1)),
            "key_metric": f"Risk: {best_con['sequence_score']*100:.0f}/100 · Stable Buffer",
        }
        used_ids.add(b_id)

    # 4. High-Volume Merchant: Highest transaction count, high volume/amounts
    mer_candidates = metrics_df[~metrics_df["borrower_id"].isin(used_ids)].copy()
    if not mer_candidates.empty:
        mer_candidates["mer_rank"] = (
            mer_candidates["tx_count"] * 10.0 + mer_candidates["avg_amount"]
        )
        best_mer = mer_candidates.sort_values(by="mer_rank", ascending=False).iloc[0]
        b_id = str(best_mer["borrower_id"])
        assigned["merchant"] = {
            **ARCHETYPE_DEFINITIONS["merchant"].__dict__,
            "borrower_id": b_id,
            "sequence_score": float(best_mer["sequence_score"]),
            "static_score": float(best_mer["static_score"]),
            "delta_pts": float(round(best_mer["delta"] * 100, 1)),
            "key_metric": f"{best_mer['tx_count']} ops · GHS {best_mer['avg_amount']:.1f} avg",
        }
        used_ids.add(b_id)

    return assigned


def classify_trajectory(balance_slope: float) -> str:
    """Classify balance trajectory into qualitative direction."""
    if balance_slope < -0.05:
        return "Deteriorating"
    if balance_slope > 0.05:
        return "Improving"
    return "Stable"


def classify_balance_tier(avg_balance: float) -> str:
    """Classify counterparty average balance into liquidity tiers."""
    if avg_balance < 200.0:
        return "Low (< GHS 200)"
    if avg_balance <= 300.0:
        return "Mid (GHS 200–300)"
    return "High (> GHS 300)"


def classify_volume_tier(tx_count: int) -> str:
    """Classify counterparty transaction activity level."""
    if tx_count < 25:
        return "Low (< 25 ops)"
    if tx_count <= 40:
        return "Mid (25–40 ops)"
    return "High (> 40 ops)"


def filter_cohort(
    scores_df: pd.DataFrame,
    details_dict: dict[str, dict[str, Any]],
    risk_bands: list[str] | None = None,
    trajectory_types: list[str] | None = None,
    balance_tiers: list[str] | None = None,
    volume_tiers: list[str] | None = None,
) -> pd.DataFrame:
    """Filter cohort DataFrame across risk band, trajectory, balance, and volume dimensions.

    Returns a filtered DataFrame enriched with trajectory, balance tier, and volume tier tags.
    """
    if scores_df.empty:
        return scores_df

    records = []
    for _, row in scores_df.iterrows():
        b_id = str(row["borrower_id"])
        detail = details_dict.get(b_id, {})
        ind = detail.get("indicators", {})

        bal_slope = float(ind.get("balance_slope", 0.0))
        avg_bal = float(ind.get("avg_balance", 0.0))
        tx_cnt = int(ind.get("transaction_count", 0))

        traj = classify_trajectory(bal_slope)
        bal_tier = classify_balance_tier(avg_bal)
        vol_tier = classify_volume_tier(tx_cnt)

        rec = row.to_dict()
        rec["trajectory"] = traj
        rec["balance_tier"] = bal_tier
        rec["volume_tier"] = vol_tier
        rec["balance_slope"] = bal_slope
        rec["avg_balance"] = avg_bal
        rec["transaction_count"] = tx_cnt
        rec["score_delta"] = float(row["sequence_demo_score"]) - float(row["static_demo_score"])
        records.append(rec)

    enriched = pd.DataFrame(records)

    # Apply filters if provided
    if risk_bands:
        # Standardize matching (handling case / "demo risk" suffix)
        enriched = enriched[
            enriched["demo_risk_band"].apply(
                lambda b: any(rb.lower() in str(b).lower() for rb in risk_bands)
            )
        ]

    if trajectory_types and "all" not in [t.lower() for t in trajectory_types]:
        valid_trajs = [t.lower() for t in trajectory_types]
        enriched = enriched[enriched["trajectory"].str.lower().isin(valid_trajs)]

    if balance_tiers and "all" not in [t.lower() for t in balance_tiers]:
        valid_tiers = [t.lower() for t in balance_tiers]
        enriched = enriched[
            enriched["balance_tier"].apply(lambda bt: any(vt in bt.lower() for vt in valid_tiers))
        ]

    if volume_tiers and "all" not in [t.lower() for t in volume_tiers]:
        valid_vols = [t.lower() for t in volume_tiers]
        enriched = enriched[
            enriched["volume_tier"].apply(lambda vt: any(vv in vt.lower() for vv in valid_vols))
        ]

    return enriched
