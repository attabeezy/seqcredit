"""Scenario stress-testing, multi-term facility sizing, and policy sandbox simulations.

This module provides counterfactual cash-flow stress testing, debt-service-to-cashflow
(DSTC) multi-term sizing, and an interactive underwriting policy sandbox.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from seqcredit_mvp.features import validate_transactions
from seqcredit_mvp.scoring import DemoScorer


TENURE_CONFIGS: dict[int, dict[str, Any]] = {
    7: {
        "tenure_days": 7,
        "name": "7-Day Micro-Loan",
        "short_label": "7D",
        "description": "Weekly working capital / emergency cashout float",
        "flat_fee_rate": 0.05,
        "annualized_rate": 0.05 * (365.0 / 7.0),
        "tenure_stress_factor": 1.45,
    },
    14: {
        "tenure_days": 14,
        "name": "14-Day Micro-Loan",
        "short_label": "14D",
        "description": "Bi-weekly merchant float & replenishment facility",
        "flat_fee_rate": 0.08,
        "annualized_rate": 0.08 * (365.0 / 14.0),
        "tenure_stress_factor": 1.20,
    },
    30: {
        "tenure_days": 30,
        "name": "30-Day Working Capital",
        "short_label": "30D",
        "description": "Monthly inventory buffer & seasonal turnover loan",
        "flat_fee_rate": 0.14,
        "annualized_rate": 0.14 * (365.0 / 30.0),
        "tenure_stress_factor": 1.00,
    },
}

DEFAULT_TENURES = (7, 14, 30)


@dataclass(frozen=True)
class TermSizingResult:
    principal: float
    tenure_days: int
    tenure_name: str
    fee_rate: float
    total_fee: float
    total_repayment: float
    daily_debt_service: float
    estimated_daily_cashflow: float
    dstc_ratio: float
    liquidity_buffer_ratio: float
    scenario_risk_index: float
    absorption_capacity: float
    risk_assessment: str


def compute_borrower_liquidity(indicators: dict[str, Any] | pd.Series) -> dict[str, float]:
    """Derive operational cashflow and liquidity capacity from borrower indicators."""
    avg_balance = max(0.0, float(indicators.get("avg_balance", 100.0)))
    avg_amount = max(5.0, float(indicators.get("avg_amount", 50.0)))
    tx_count = max(1.0, float(indicators.get("transaction_count", 30.0)))
    pct_cashout = min(0.95, max(0.0, float(indicators.get("pct_cashout", 0.20))))

    daily_tx_rate = max(0.33, tx_count / 30.0)
    daily_turnover = avg_amount * daily_tx_rate
    daily_net_cashflow = max(10.0, daily_turnover * (1.0 - pct_cashout * 0.6))
    capacity = max(100.0, avg_balance + avg_amount * 4.0)

    return {
        "avg_balance": avg_balance,
        "avg_amount": avg_amount,
        "daily_net_cashflow": round(daily_net_cashflow, 2),
        "absorption_capacity": round(capacity, 2),
    }


def scenario_index(
    base_score: float,
    amount: float,
    capacity: float,
    tenure_days: int = 30,
) -> float:
    """Calculate non-linear stress risk index for a given exposure and tenure."""
    bounded = min(0.98, max(0.02, float(base_score)))
    logit = math.log(bounded / (1.0 - bounded))

    tenure_cfg = TENURE_CONFIGS.get(int(tenure_days), TENURE_CONFIGS[30])
    tenure_multiplier = float(tenure_cfg.get("tenure_stress_factor", 1.0))

    safe_capacity = max(100.0, float(capacity))
    exposure_pressure = (float(amount) / safe_capacity) * 2.0 * tenure_multiplier
    stress = 0.8 * math.log1p(exposure_pressure)

    return 1.0 / (1.0 + math.exp(-(logit + stress)))


def simulate_term_sizing(
    base_score: float,
    principal: float,
    tenure_days: int,
    indicators: dict[str, Any] | pd.Series,
) -> TermSizingResult:
    """Evaluate debt-service-to-cashflow (DSTC) and risk index for a specific tenure."""
    principal = max(10.0, float(principal))
    tenure_days = int(tenure_days)
    cfg = TENURE_CONFIGS.get(tenure_days, TENURE_CONFIGS[30])

    fee_rate = float(cfg["flat_fee_rate"])
    total_fee = round(principal * fee_rate, 2)
    total_repayment = round(principal + total_fee, 2)
    daily_service = round(total_repayment / tenure_days, 2)

    liq = compute_borrower_liquidity(indicators)
    daily_cashflow = liq["daily_net_cashflow"]
    capacity = liq["absorption_capacity"]

    dstc_ratio = round(daily_service / max(1.0, daily_cashflow), 4)
    buffer_ratio = round(principal / capacity, 4)
    risk_idx = scenario_index(base_score, principal, capacity, tenure_days=tenure_days)

    if dstc_ratio > 1.25:
        assessment = "Severe Debt Service Strain (DSTC > 125%)"
    elif dstc_ratio > 0.75:
        assessment = "Elevated Debt Service Pressure (DSTC 75–125%)"
    elif dstc_ratio > 0.40:
        assessment = "Moderate Liquidity Absorption (DSTC 40–75%)"
    else:
        assessment = "Well-Supported Liquidity (DSTC < 40%)"

    return TermSizingResult(
        principal=round(principal, 2),
        tenure_days=tenure_days,
        tenure_name=cfg["name"],
        fee_rate=fee_rate,
        total_fee=total_fee,
        total_repayment=total_repayment,
        daily_debt_service=daily_service,
        estimated_daily_cashflow=daily_cashflow,
        dstc_ratio=dstc_ratio,
        liquidity_buffer_ratio=buffer_ratio,
        scenario_risk_index=round(risk_idx, 6),
        absorption_capacity=capacity,
        risk_assessment=assessment,
    )


def simulate_multi_term_curves(
    base_score: float,
    capacity: float,
    amounts_range: list[int | float] | None = None,
) -> dict[int, list[float]]:
    """Compute scenario risk indices across loan amounts for all standard tenures."""
    if amounts_range is None:
        amounts_range = list(range(25, 1025, 25))

    curves: dict[int, list[float]] = {}
    for tenure in DEFAULT_TENURES:
        curves[tenure] = [
            round(scenario_index(base_score, amt, capacity, tenure_days=tenure), 6)
            for amt in amounts_range
        ]
    return curves


def apply_cashflow_stress(
    features_row: pd.Series,
    inbound_drop_pct: float = 0.20,
    fee_surge_pct: float = 0.30,
    inactivity_days: float = 7.0,
) -> pd.Series:
    """Apply counterfactual financial perturbations to a borrower feature series.

    Perturbations simulate:
    - Inbound cash-flow contraction (-0% to -50%)
    - Operational transaction fee surcharge (+0% to +60%)
    - Extended inter-transaction dormancy / inactivity (0 to 21 days)
    """
    s = features_row.copy()
    inbound_factor = max(0.0, min(0.60, float(inbound_drop_pct)))
    fee_factor = max(0.0, min(1.0, float(fee_surge_pct)))
    gap_days = max(0.0, min(30.0, float(inactivity_days)))

    # Monetary and balance contractions
    if "total_volume" in s:
        s["total_volume"] = max(10.0, s["total_volume"] * (1.0 - inbound_factor * 0.85))
    if "avg_amount" in s:
        s["avg_amount"] = max(5.0, s["avg_amount"] * (1.0 - inbound_factor * 0.40))
    if "median_amount" in s:
        s["median_amount"] = max(5.0, s["median_amount"] * (1.0 - inbound_factor * 0.40))
    if "avg_balance" in s:
        s["avg_balance"] = max(0.0, s["avg_balance"] * (1.0 - inbound_factor * 0.80) - (s.get("total_fees", 0.0) * fee_factor * 0.2))
    if "min_balance" in s:
        s["min_balance"] = max(0.0, s["min_balance"] * (1.0 - inbound_factor * 0.90))
    if "pct_low_balance" in s:
        s["pct_low_balance"] = min(1.0, s["pct_low_balance"] + inbound_factor * 0.45 + fee_factor * 0.15)
    if "balance_volatility" in s:
        s["balance_volatility"] = s["balance_volatility"] * (1.0 + fee_factor * 0.30)
    if "total_fees" in s:
        s["total_fees"] = s["total_fees"] * (1.0 + fee_factor)

    # Sequence velocity & ordering dynamics
    if "avg_hours_between" in s:
        tx_count = max(1.0, float(s.get("transaction_count", 30.0)))
        s["avg_hours_between"] = s["avg_hours_between"] + (gap_days * 24.0 / tx_count)
    if "balance_slope" in s:
        s["balance_slope"] = s["balance_slope"] - (inbound_factor * 0.35 + fee_factor * 0.15)
    if "amount_slope" in s:
        s["amount_slope"] = s["amount_slope"] - (inbound_factor * 0.25)
    if "balance_recent_ratio" in s:
        s["balance_recent_ratio"] = max(0.05, s["balance_recent_ratio"] * (1.0 - inbound_factor * 0.50 - gap_days * 0.02))
    if "amount_recent_ratio" in s:
        s["amount_recent_ratio"] = max(0.05, s["amount_recent_ratio"] * (1.0 - gap_days * 0.025))
    if "recent_low_balance_rate" in s:
        s["recent_low_balance_rate"] = min(1.0, s["recent_low_balance_rate"] + inbound_factor * 0.40)
    if "recent_cashout_rate" in s:
        s["recent_cashout_rate"] = min(1.0, s["recent_cashout_rate"] + fee_factor * 0.10)

    return s


def evaluate_borrower_stress(
    scorer: DemoScorer,
    sequence_row: pd.Series,
    static_row: pd.Series,
    inbound_drop_pct: float = 0.20,
    fee_surge_pct: float = 0.30,
    inactivity_days: float = 7.0,
) -> dict[str, Any]:
    """Re-evaluate the surrogate machine-learning models under stressed conditions."""
    stressed_seq = apply_cashflow_stress(
        sequence_row,
        inbound_drop_pct=inbound_drop_pct,
        fee_surge_pct=fee_surge_pct,
        inactivity_days=inactivity_days,
    )
    stressed_sta = apply_cashflow_stress(
        static_row,
        inbound_drop_pct=inbound_drop_pct,
        fee_surge_pct=fee_surge_pct,
        inactivity_days=inactivity_days,
    )

    expected_seq = scorer.manifest["models"]["sequence"]["features"]
    expected_sta = scorer.manifest["models"]["static"]["features"]

    df_seq_baseline = pd.DataFrame([sequence_row.loc[expected_seq]])
    df_seq_stressed = pd.DataFrame([stressed_seq.loc[expected_seq]])
    df_sta_baseline = pd.DataFrame([static_row.loc[expected_sta]])
    df_sta_stressed = pd.DataFrame([stressed_sta.loc[expected_sta]])

    base_seq_score = float(scorer.sequence_model.predict_proba(df_seq_baseline)[0, 1])
    stressed_seq_score = float(scorer.sequence_model.predict_proba(df_seq_stressed)[0, 1])
    base_sta_score = float(scorer.static_model.predict_proba(df_sta_baseline)[0, 1])
    stressed_sta_score = float(scorer.static_model.predict_proba(df_sta_stressed)[0, 1])

    thresholds = scorer.manifest["demo_risk_band_thresholds"]

    def band(val: float) -> str:
        if val >= thresholds["elevated"]:
            return "Elevated demo risk"
        if val >= thresholds["watch"]:
            return "Watch demo risk"
        return "Lower demo risk"

    base_band = band(base_seq_score)
    stressed_band = band(stressed_seq_score)

    delta_pts = (stressed_seq_score - base_seq_score) * 100.0

    return {
        "parameters": {
            "inbound_drop_pct": inbound_drop_pct,
            "fee_surge_pct": fee_surge_pct,
            "inactivity_days": inactivity_days,
        },
        "baseline": {
            "sequence_score": round(base_seq_score, 6),
            "static_score": round(base_sta_score, 6),
            "risk_band": base_band,
        },
        "stressed": {
            "sequence_score": round(stressed_seq_score, 6),
            "static_score": round(stressed_sta_score, 6),
            "risk_band": stressed_band,
        },
        "delta_pts": round(delta_pts, 2),
        "band_shifted": base_band != stressed_band,
    }


def simulate_policy_sandbox(
    scores_df: pd.DataFrame,
    watch_threshold: float,
    elevated_threshold: float,
    baseline_watch: float | None = None,
    baseline_elevated: float | None = None,
    ground_truth_labels: pd.DataFrame | pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate portfolio underwriting outcomes, default capture, and migration."""
    if scores_df.empty:
        raise ValueError("scores_df cannot be empty")

    watch = float(min(watch_threshold, elevated_threshold))
    elevated = float(max(watch_threshold, elevated_threshold))

    scores = scores_df.copy()
    seq_col = "sequence_demo_score" if "sequence_demo_score" in scores.columns else scores.columns[1]

    # Baseline thresholds
    base_watch = float(baseline_watch) if baseline_watch is not None else watch
    base_elev = float(baseline_elevated) if baseline_elevated is not None else elevated

    def classify(val: float, w: float, e: float) -> str:
        if val >= e:
            return "Elevated demo risk"
        if val >= w:
            return "Watch demo risk"
        return "Lower demo risk"

    scores["policy_band"] = scores[seq_col].apply(lambda s: classify(s, watch, elevated))
    scores["baseline_band"] = scores[seq_col].apply(lambda s: classify(s, base_watch, base_elev))

    n_total = len(scores)

    # Acceptance metrics under new policy
    n_lower = int((scores["policy_band"] == "Lower demo risk").sum())
    n_watch = int((scores["policy_band"] == "Watch demo risk").sum())
    n_elevated = int((scores["policy_band"] == "Elevated demo risk").sum())

    acceptance_rate = n_lower / n_total
    conditional_rate = n_watch / n_total
    decline_rate = n_elevated / n_total

    # Baseline acceptance metrics
    base_lower = int((scores["baseline_band"] == "Lower demo risk").sum())
    base_watch_count = int((scores["baseline_band"] == "Watch demo risk").sum())
    base_elevated_count = int((scores["baseline_band"] == "Elevated demo risk").sum())
    base_acceptance_rate = base_lower / n_total

    # Default capture calculation
    # If ground_truth_labels provided, compute empirical strict default capture (class 2)
    has_labels = False
    empirical_defaults_total = 0
    empirical_defaults_captured = 0
    default_capture_rate = 0.0

    if ground_truth_labels is not None:
        if isinstance(ground_truth_labels, pd.DataFrame) and "credit_risk_label" in ground_truth_labels.columns:
            lookup = ground_truth_labels.set_index("borrower_id")["credit_risk_label"]
        elif isinstance(ground_truth_labels, pd.Series):
            lookup = ground_truth_labels
        else:
            lookup = None

        if lookup is not None and "borrower_id" in scores.columns:
            has_labels = True
            is_default = scores["borrower_id"].map(lookup) == 2
            empirical_defaults_total = int(is_default.sum())
            if empirical_defaults_total > 0:
                # Captured = elevated or watch tier
                captured_mask = is_default & (scores["policy_band"].isin(["Watch demo risk", "Elevated demo risk"]))
                empirical_defaults_captured = int(captured_mask.sum())
                default_capture_rate = round(empirical_defaults_captured / empirical_defaults_total, 4)

    # If no labels or 0 empirical defaults, use expected default capture from sequence probabilities
    expected_defaults_total = float(scores[seq_col].sum())
    flagged_mask = scores["policy_band"].isin(["Watch demo risk", "Elevated demo risk"])
    expected_defaults_captured = float(scores.loc[flagged_mask, seq_col].sum())
    expected_default_capture_rate = (
        round(expected_defaults_captured / max(1e-6, expected_defaults_total), 4)
        if expected_defaults_total > 0
        else 0.0
    )

    effective_capture_rate = default_capture_rate if has_labels and empirical_defaults_total > 0 else expected_default_capture_rate

    # 3x3 Risk Band Migration Matrix
    bands_order = ["Lower demo risk", "Watch demo risk", "Elevated demo risk"]
    migration_matrix: dict[str, dict[str, int]] = {b1: {b2: 0 for b2 in bands_order} for b1 in bands_order}

    for _, row in scores.iterrows():
        b_base = str(row["baseline_band"])
        b_pol = str(row["policy_band"])
        if b_base in migration_matrix and b_pol in migration_matrix[b_base]:
            migration_matrix[b_base][b_pol] += 1

    # Tier transitions
    tier_rank = {"Lower demo risk": 0, "Watch demo risk": 1, "Elevated demo risk": 2}
    upgrades = 0  # moved to safer band (e.g. Watch -> Lower)
    downgrades = 0  # moved to stricter band (e.g. Watch -> Elevated)
    unchanged = 0

    for _, row in scores.iterrows():
        r_base = tier_rank.get(row["baseline_band"], 0)
        r_pol = tier_rank.get(row["policy_band"], 0)
        if r_pol < r_base:
            upgrades += 1
        elif r_pol > r_base:
            downgrades += 1
        else:
            unchanged += 1

    return {
        "thresholds": {
            "watch": round(watch, 4),
            "elevated": round(elevated, 4),
            "baseline_watch": round(base_watch, 4),
            "baseline_elevated": round(base_elev, 4),
        },
        "acceptance": {
            "full_approval_count": n_lower,
            "full_approval_rate": round(acceptance_rate, 4),
            "conditional_review_count": n_watch,
            "conditional_review_rate": round(conditional_rate, 4),
            "decline_count": n_elevated,
            "decline_rate": round(decline_rate, 4),
            "baseline_approval_rate": round(base_acceptance_rate, 4),
            "acceptance_delta_pts": round((acceptance_rate - base_acceptance_rate) * 100.0, 2),
        },
        "default_capture": {
            "has_empirical_ground_truth": has_labels and empirical_defaults_total > 0,
            "capture_rate": effective_capture_rate,
            "captured_count": empirical_defaults_captured if has_labels else round(expected_defaults_captured, 1),
            "total_defaults": empirical_defaults_total if has_labels else round(expected_defaults_total, 1),
        },
        "migration_matrix": migration_matrix,
        "migration_summary": {
            "upgrades": upgrades,
            "unchanged": unchanged,
            "downgrades": downgrades,
            "total": n_total,
        },
    }


def evaluate_window_sensitivity(
    scorer: DemoScorer,
    borrower_txs: pd.DataFrame,
    window_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate how sequence and static risk scores evolve across observation window depths.

    Parameters
    ----------
    scorer : DemoScorer
        Fitted / frozen scorer contract instance.
    borrower_txs : pd.DataFrame
        Transactions for a specific borrower.
    window_sizes : list[int] | None
        History transaction depths to evaluate (e.g., [8, 14, 20, 28, 36]). Defaults to
        sensible increments bounded by borrower total transactions.

    Returns
    -------
    dict[str, Any]
        Structured sensitivity results including window points, score trajectory, delta divergence,
        and stability summary.
    """
    validated = validate_transactions(borrower_txs)
    if validated.empty:
        raise ValueError("No valid transactions provided for sensitivity analysis.")

    borrower_id = str(validated["borrower_id"].iloc[0])
    df_b = validated[validated["borrower_id"] == borrower_id].sort_values("timestamp")
    total_tx = len(df_b)

    if window_sizes is None:
        candidates = [8, 14, 20, 28, 36, 50]
        windows = [w for w in candidates if 6 <= w < total_tx]
        if total_tx >= 6:
            windows.append(total_tx)
        if not windows:
            windows = [max(4, total_tx)]
    else:
        windows = sorted([w for w in set(window_sizes) if 4 <= w <= total_tx])
        if not windows:
            windows = [total_tx]

    points: list[dict[str, Any]] = []
    for w in windows:
        sub_tx = df_b.iloc[-w:].copy()
        scored = scorer.score(sub_tx)
        if scored.empty:
            continue
        row = scored.iloc[0]
        seq_score = float(row["sequence_demo_score"])
        stat_score = float(row["static_demo_score"])
        band = str(row["demo_risk_band"])
        delta = round(seq_score - stat_score, 4)

        points.append({
            "window_size": int(w),
            "sequence_score": round(seq_score, 4),
            "static_score": round(stat_score, 4),
            "score_delta": delta,
            "risk_band": band,
            "avg_amount": round(float(sub_tx["amount"].mean()), 2),
            "avg_balance": round(float(sub_tx["balance_after"].mean()), 2),
        })

    seq_scores = [p["sequence_score"] for p in points]
    deltas = [p["score_delta"] for p in points]
    max_divergence = round(float(np.max(np.abs(deltas))), 4) if deltas else 0.0
    score_range = round(float(max(seq_scores) - min(seq_scores)), 4) if seq_scores else 0.0
    final_point = points[-1] if points else {}

    return {
        "borrower_id": borrower_id,
        "total_transactions": total_tx,
        "window_points": points,
        "summary": {
            "evaluated_windows": [p["window_size"] for p in points],
            "score_range": score_range,
            "max_abs_divergence": max_divergence,
            "final_sequence_score": final_point.get("sequence_score"),
            "final_static_score": final_point.get("static_score"),
            "final_band": final_point.get("risk_band"),
        },
    }

