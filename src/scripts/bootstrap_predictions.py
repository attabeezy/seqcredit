"""Compute paired borrower bootstrap intervals from locked test predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from competition import metrics, write_json


METRICS = ("average_precision", "roc_auc", "brier", "log_loss", "precision", "recall", "f1")


def bootstrap(run: Path, draws: int = 1000, seed: int = 42) -> dict:
    rows = []
    with (run / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "test"]
    if not rows:
        raise ValueError("Run has no locked test predictions")
    models = ("logistic_regression", "lightgbm", "gru")
    y = np.array([int(row["target"]) for row in rows], dtype=np.int32)
    probabilities = {model: np.array([float(row[model]) for row in rows]) for model in models}
    rng = np.random.default_rng(seed)
    samples = {model: {metric: [] for metric in METRICS} for model in models}
    skipped = 0
    for _ in range(draws):
        index = rng.integers(0, len(y), size=len(y))
        sampled_y = y[index]
        if np.unique(sampled_y).size < 2:
            skipped += 1
            continue
        for model in models:
            result = metrics(sampled_y, probabilities[model][index])
            for metric in METRICS:
                samples[model][metric].append(result[metric])
    if not any(samples[models[0]].values()):
        raise ValueError("All bootstrap replicates were single-class")
    intervals = {
        model: {
            metric: {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
                "replicates": len(values),
            }
            for metric, values in model_samples.items()
        }
        for model, model_samples in samples.items()
    }
    differences = {}
    for baseline in ("logistic_regression", "lightgbm"):
        differences[f"gru_minus_{baseline}"] = {}
        for metric in ("roc_auc", "average_precision"):
            values = np.asarray(samples["gru"][metric]) - np.asarray(samples[baseline][metric])
            differences[f"gru_minus_{baseline}"][metric] = {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
                "replicates": len(values),
            }
    return {
        "status": "passed",
        "run": run.name,
        "partition": "test",
        "borrowers": len(y),
        "draws_requested": draws,
        "draws_used": draws - skipped,
        "single_class_draws_skipped": skipped,
        "seed": seed,
        "models": list(models),
        "intervals": intervals,
        "paired_differences": differences,
        "sampling": "paired borrower indices shared across models within each draw",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.draws <= 0:
        raise ValueError("Draw count must be positive")
    report = bootstrap(args.run.resolve(), args.draws, args.seed)
    write_json(args.run / "bootstrap_ci.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
