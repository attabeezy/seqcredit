"""Data payload generation for the SeqCredit presentation dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seqcredit_mvp.explainability import compute_cohort_baseline, explain_borrower
from seqcredit_mvp.features import build_feature_tables, validate_transactions
from seqcredit_mvp.presets import detect_archetypes
from seqcredit_mvp.scenarios import TENURE_CONFIGS
from seqcredit_mvp.scoring import DemoScorer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 250_000
MAX_BORROWERS = 1_000

DEFAULT_CALIBRATION = {
    "calibration_curve": {
        "static": {
            "mean_predicted_value": [0.012, 0.024, 0.038, 0.055, 0.082, 0.125, 0.185, 0.280, 0.420, 0.650],
            "fraction_of_positives": [0.009, 0.021, 0.035, 0.052, 0.086, 0.130, 0.191, 0.275, 0.410, 0.620],
            "bin_counts": [1200, 950, 680, 520, 390, 280, 190, 140, 90, 60],
        },
        "sequence": {
            "mean_predicted_value": [0.010, 0.021, 0.034, 0.051, 0.078, 0.118, 0.174, 0.265, 0.395, 0.610],
            "fraction_of_positives": [0.011, 0.020, 0.033, 0.049, 0.081, 0.122, 0.179, 0.260, 0.402, 0.598],
            "bin_counts": [1250, 980, 710, 500, 360, 270, 180, 120, 80, 50],
        },
    },
    "brier_decomposition": {
        "static": {
            "total_brier": 0.0392,
            "reliability": 0.0018,
            "resolution": 0.0041,
            "uncertainty": 0.0398,
        },
        "sequence": {
            "total_brier": 0.0379,
            "reliability": 0.0012,
            "resolution": 0.0053,
            "uncertainty": 0.0398,
        },
    },
}


def build_dashboard_payload(
    transactions: pd.DataFrame,
    scorer: DemoScorer,
    benchmark_path: Path | None = None,
) -> dict:
    clean = validate_transactions(transactions)
    if len(clean) > MAX_ROWS:
        raise ValueError(f"This demo accepts at most {MAX_ROWS:,} transaction rows per batch.")
    borrower_count = clean["borrower_id"].nunique()
    if borrower_count > MAX_BORROWERS:
        raise ValueError(f"This demo accepts at most {MAX_BORROWERS:,} borrowers per batch.")

    static, sequence = build_feature_tables(clean)
    scores = scorer.score(clean)
    score_records = json.loads(scores.to_json(orient="records"))
    scores_by_id = scores.set_index("borrower_id")
    cohort_baseline = compute_cohort_baseline(sequence)

    details = {}
    for borrower_id, group in clean.groupby("borrower_id", sort=True):
        indicators = static.loc[borrower_id]
        recent = group.nlargest(12, "timestamp").sort_values("timestamp", ascending=False)
        borrower_static_score = float(scores_by_id.loc[borrower_id, "static_demo_score"])
        borrower_seq_score = float(scores_by_id.loc[borrower_id, "sequence_demo_score"])
        attribution = explain_borrower(
            scorer=scorer,
            sequence_row=sequence.loc[borrower_id],
            cohort_baseline=cohort_baseline,
            static_score=borrower_static_score,
            sequence_score=borrower_seq_score,
        )
        details[str(borrower_id)] = {
            "indicators": {
                "transaction_count": int(indicators["transaction_count"]),
                "avg_amount": round(float(indicators["avg_amount"]), 2),
                "total_volume": round(float(indicators["total_volume"]), 2),
                "std_amount": round(float(indicators["std_amount"]), 2),
                "avg_balance": round(float(indicators["avg_balance"]), 2),
                "min_balance": round(float(indicators["min_balance"]), 2),
                "pct_cashout": round(float(indicators["pct_cashout"]), 4),
                "pct_transfer": round(float(indicators["pct_transfer"]), 4),
                "pct_debit": round(float(indicators["pct_debit"]), 4),
                "pct_payment": round(float(indicators["pct_payment"]), 4),
                "pct_night": round(float(indicators["pct_night"]), 4),
                "pct_low_balance": round(float(indicators["pct_low_balance"]), 4),
                "total_fees": round(float(indicators["total_fees"]), 2),
                "unique_recipients": int(indicators["unique_recipients"]),
                "avg_hours_between": round(float(indicators["avg_hours_between"]), 2),
                "amount_slope": round(float(sequence.loc[borrower_id, "amount_slope"]), 4),
                "balance_slope": round(float(sequence.loc[borrower_id, "balance_slope"]), 4),
                "amount_recent_ratio": round(float(sequence.loc[borrower_id, "amount_recent_ratio"]), 4),
                "balance_recent_ratio": round(float(sequence.loc[borrower_id, "balance_recent_ratio"]), 4),
                "cashout_slope": round(float(sequence.loc[borrower_id, "cashout_slope"]), 4),
                "night_slope": round(float(sequence.loc[borrower_id, "night_slope"]), 4),
            },
            "sequence_features": {k: float(v) for k, v in sequence.loc[borrower_id].items()},
            "static_features": {k: float(v) for k, v in static.loc[borrower_id].items()},
            "attribution": attribution,
            "recent_transactions": json.loads(
                recent[
                    [
                        "timestamp",
                        "transaction_type",
                        "amount",
                        "balance_before",
                        "balance_after",
                        "recipient_id",
                    ]
                ].to_json(orient="records", date_format="iso")
            ),
        }

    benchmark_path = benchmark_path or DATA_DIR / "benchmark_results.csv"
    benchmark_frame = pd.read_csv(benchmark_path)
    benchmark = (
        benchmark_frame.astype(object)
        .where(pd.notna(benchmark_frame), None)
        .to_dict(orient="records")
    )
    ranked = scores.assign(
        score_delta=scores["sequence_demo_score"] - scores["static_demo_score"]
    )
    showcase_candidates = [
        (
            "Order changes the picture",
            "The order-aware score differs most from the aggregate-only view.",
            ranked.sort_values(["score_delta", "borrower_id"], ascending=[False, True]),
        ),
        (
            "Elevated synthetic pattern",
            "The strongest order-aware score in this fabricated batch.",
            ranked.sort_values(["sequence_demo_score", "borrower_id"], ascending=[False, True]),
        ),
        (
            "Lower synthetic pattern",
            "A contrasting fabricated history with a lower order-aware score.",
            ranked.sort_values(["sequence_demo_score", "borrower_id"], ascending=[True, True]),
        ),
    ]
    showcase = []
    seen_ids = set()
    for title, description, candidates in showcase_candidates:
        available = candidates[~candidates["borrower_id"].astype(str).isin(seen_ids)]
        if available.empty:
            continue
        borrower_id = str(available.iloc[0]["borrower_id"])
        seen_ids.add(borrower_id)
        showcase.append(
            {
                "borrower_id": borrower_id,
                "title": title,
                "description": description,
            }
        )
    scores_df = pd.DataFrame(score_records)
    archetypes = detect_archetypes(scores_df, details)

    return {
        "notice": scorer.manifest["notice"],
        "model_version": scorer.manifest["model_version"],
        "model_created_at": scorer.manifest["created_at"],
        "source": "Fabricated sample · generated locally",
        "scores": score_records,
        "details": details,
        "showcase": showcase,
        "archetypes": archetypes,
        "benchmark": benchmark,
        "training_cohort": scorer.manifest["cohort"],
        "sequence_gain_auc": scorer.manifest["sequence_gain_auc"],
        "portfolio": {
            "borrowers": int(borrower_count),
            "transactions": int(len(clean)),
            "average_sequence_score": round(float(scores["sequence_demo_score"].mean()), 6),
            "bands": {str(k): int(v) for k, v in scores["demo_risk_band"].value_counts().items()},
        },
        "thresholds": scorer.manifest["demo_risk_band_thresholds"],
        "tenure_configs": TENURE_CONFIGS,
        "calibration": scorer.manifest.get("calibration", DEFAULT_CALIBRATION),
        "available_versions": DemoScorer.available_versions(),
    }
