"""Train and verify versioned synthetic demo models."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seqcredit_mvp.features import build_feature_tables
from seqcredit_mvp.synthetic import SyntheticConfig, generate_synthetic_cohort


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
MODEL_VERSION = "synthetic-demo-v1"

REAL_REFERENCE = {
    "default_rate": 6170 / 149667,
    "static_auc": 0.7211,
    "lstm_auc": 0.7418,
    "gru_auc": 0.7517,
}

ACCEPTANCE = {
    "default_rate": [0.035, 0.050],
    "static_auc": [0.700, 0.740],
    "sequence_auc": [0.735, 0.770],
    "sequence_gain": [0.015, 0.050],
}


def compute_calibration_curve(y: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> dict[str, list[Any]]:
    """Compute empirical binned true positive rates vs predicted probabilities."""
    prob_true, prob_pred = calibration_curve(y, probas, n_bins=n_bins, strategy="uniform")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probas, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = [int((bin_indices == k).sum()) for k in range(n_bins) if (bin_indices == k).sum() > 0]

    pred_list = [float(round(p, 4)) for p in prob_pred]
    true_list = [float(round(p, 4)) for p in prob_true]
    return {
        "prob_pred": pred_list,
        "prob_true": true_list,
        "mean_predicted_value": pred_list,
        "fraction_of_positives": true_list,
        "bin_counts": bin_counts,
    }


def compute_brier_decomposition(y: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Decompose Brier score into reliability, resolution, and uncertainty components."""
    total_brier = float(brier_score_loss(y, probas))
    o_bar = float(y.mean())
    unc = o_bar * (1.0 - o_bar)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probas, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_total = len(y)
    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        n_k = mask.sum()
        if n_k > 0:
            p_k = probas[mask].mean()
            o_k = y[mask].mean()
            rel += (n_k / n_total) * ((p_k - o_k) ** 2)
            res += (n_k / n_total) * ((o_k - o_bar) ** 2)

    rounded_brier = float(round(total_brier, 5))
    binned_brier = float(round(max(0.0, rel - res + unc), 5))
    return {
        "total": rounded_brier,
        "total_brier": rounded_brier,
        "binned_brier": binned_brier,
        "reliability": float(round(rel, 5)),
        "resolution": float(round(res, 5)),
        "uncertainty": float(round(unc, 5)),
    }


def _model_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y, probabilities)),
        "auc_pr": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_calibration(
    config: SyntheticConfig,
    version: str = "v1",
    output_dir: Path | None = None,
) -> dict:
    target_dir = Path(output_dir) if output_dir else ARTIFACT_DIR
    model_version_str = f"synthetic-demo-{version}"

    transactions, labels = generate_synthetic_cohort(config)
    static_features, sequence_features = build_feature_tables(transactions)
    label_lookup = labels.set_index("borrower_id")
    y = (label_lookup.loc[static_features.index, "credit_risk_label"] == 2).astype(int).to_numpy()

    static_model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=260,
            max_depth=5,
            min_samples_leaf=18,
            class_weight="balanced_subsample",
            random_state=config.seed,
            n_jobs=1,
        ),
    )
    sequence_model = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.045,
            max_leaf_nodes=9,
            l2_regularization=2.0,
            random_state=config.seed,
        ),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    static_oof = cross_val_predict(static_model, static_features, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    sequence_oof = cross_val_predict(sequence_model, sequence_features, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    static_metrics = _model_metrics(y, static_oof)
    sequence_metrics = _model_metrics(y, sequence_oof)

    # Compute empirical reliability curves and Brier decompositions
    static_calib = compute_calibration_curve(y, static_oof)
    sequence_calib = compute_calibration_curve(y, sequence_oof)
    static_brier = compute_brier_decomposition(y, static_oof)
    sequence_brier = compute_brier_decomposition(y, sequence_oof)

    static_model.fit(static_features, y)
    sequence_model.fit(sequence_features, y)
    fitted_sequence_scores = sequence_model.predict_proba(sequence_features)[:, 1]
    risk_band_thresholds = {
        "watch": float(np.quantile(fitted_sequence_scores, 0.65)),
        "elevated": float(np.quantile(fitted_sequence_scores, 0.90)),
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    version_dir = target_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    static_path = target_dir / "static_model.joblib"
    sequence_path = target_dir / "sequence_model.joblib"
    joblib.dump(static_model, static_path)
    joblib.dump(sequence_model, sequence_path)

    # Also save in version_dir
    joblib.dump(static_model, version_dir / "static_model.joblib")
    joblib.dump(sequence_model, version_dir / "sequence_model.joblib")

    default_rate = float(y.mean())
    gain = sequence_metrics["auc_roc"] - static_metrics["auc_roc"]
    checks = {
        "default_rate": ACCEPTANCE["default_rate"][0] <= default_rate <= ACCEPTANCE["default_rate"][1],
        "static_auc": ACCEPTANCE["static_auc"][0] <= static_metrics["auc_roc"] <= ACCEPTANCE["static_auc"][1],
        "sequence_auc": ACCEPTANCE["sequence_auc"][0] <= sequence_metrics["auc_roc"] <= ACCEPTANCE["sequence_auc"][1],
        "sequence_gain": ACCEPTANCE["sequence_gain"][0] <= gain <= ACCEPTANCE["sequence_gain"][1],
    }

    benchmark = pd.DataFrame(
        [
            {"evidence": "synthetic_demo", "model": "Static random forest", **static_metrics},
            {"evidence": "synthetic_demo", "model": "Order-aware sequence surrogate", **sequence_metrics},
            {"evidence": "archived_real_reference", "model": "XGBoost", "auc_roc": REAL_REFERENCE["static_auc"]},
            {"evidence": "archived_real_reference", "model": "LSTM", "auc_roc": REAL_REFERENCE["lstm_auc"]},
            {"evidence": "archived_real_reference", "model": "GRU", "auc_roc": REAL_REFERENCE["gru_auc"]},
        ]
    )
    benchmark_path = DATA_DIR / "benchmark_results.csv"
    benchmark.to_csv(benchmark_path, index=False)

    sample_ids = labels.sample(n=60, random_state=config.seed)["borrower_id"]
    sample_path = DATA_DIR / "sample_transactions.csv"
    sample = transactions[transactions["borrower_id"].isin(sample_ids)].drop(
        columns=["transaction_number"], errors="ignore"
    )
    sample.to_csv(sample_path, index=False)

    manifest = {
        "model_version": model_version_str,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "notice": "Synthetic prototype only. Not for lending decisions.",
        "generator": config.__dict__,
        "cohort": {
            "borrowers": int(len(labels)),
            "transactions": int(len(transactions)),
            "default_rate": default_rate,
            "class_counts": {str(k): int(v) for k, v in labels["credit_risk_label"].value_counts().sort_index().items()},
        },
        "models": {
            "static": {
                "family": "RandomForestClassifier",
                "features": list(static_features.columns),
                "metrics": static_metrics,
                "sha256": _hash_file(static_path),
            },
            "sequence": {
                "family": "HistGradientBoostingClassifier over order-aware features",
                "features": list(sequence_features.columns),
                "metrics": sequence_metrics,
                "sha256": _hash_file(sequence_path),
            },
        },
        "calibration": {
            "reliability_curves": {
                "static": static_calib,
                "sequence": sequence_calib,
            },
            "brier_decomposition": {
                "static": static_brier,
                "sequence": sequence_brier,
            },
        },
        "sequence_gain_auc": gain,
        "demo_risk_band_thresholds": risk_band_thresholds,
        "real_reference": REAL_REFERENCE,
        "acceptance": ACCEPTANCE,
        "checks": checks,
        "input_schema": sorted(
            [
                "borrower_id",
                "transaction_id",
                "timestamp",
                "transaction_type",
                "amount",
                "balance_before",
                "balance_after",
                "fee",
                "recipient_id",
            ]
        ),
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borrowers", type=int, default=4_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v1", help="Model artifact contract version (e.g. v1, v2)")
    parser.add_argument("--allow-miss", action="store_true", help="Write diagnostics without failing on calibration bands.")
    args = parser.parse_args()
    manifest = run_calibration(SyntheticConfig(n_borrowers=args.borrowers, seed=args.seed), version=args.version)
    print(json.dumps({"status": manifest["status"], "checks": manifest["checks"], "models": manifest["models"], "sequence_gain_auc": manifest["sequence_gain_auc"]}, indent=2))
    if manifest["status"] != "passed" and not args.allow_miss:
        raise SystemExit("Synthetic calibration missed one or more acceptance bands.")


if __name__ == "__main__":
    main()
