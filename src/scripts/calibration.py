"""Validation-only sigmoid calibration of saved borrower probabilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from contract_checks import fingerprint


MODELS = ("logistic_regression", "lightgbm", "gru")
CLIP = 1e-7
POLICY = {"method": "sigmoid_on_logit", "fit_partition": "validation", "seed": 42,
          "folds": 5, "class_weight": None, "regularization": None, "solver": "lbfgs",
          "tol": 1e-10, "max_iter": 1000, "clip_min": CLIP, "clip_max": 1 - CLIP,
          "ece_bins": 10, "threshold": 0.5,
          "test_interpretation": "follow-up on an already inspected synthetic test set"}


def _scores(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Require finite one-dimensional probabilities in [0, 1]")
    return logit(np.clip(probability, CLIP, 1 - CLIP)).reshape(-1, 1)


def fit(probability: np.ndarray, labels: np.ndarray) -> dict:
    """Fit an unweighted sigmoid using only the supplied calibration labels."""
    labels = np.asarray(labels)
    if labels.shape != np.shape(probability) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires aligned labels with both classes")
    estimator = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-10,
                                   max_iter=1000, random_state=42, class_weight=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore.*", category=UserWarning)
        estimator.fit(_scores(probability), labels)
    return {**POLICY, "slope": float(estimator.coef_[0, 0]),
            "intercept": float(estimator.intercept_[0]), "iterations": int(estimator.n_iter_[0])}


def apply(parameters: dict, probability: np.ndarray) -> np.ndarray:
    """Apply portable JSON coefficients to raw base-model probabilities."""
    if any(parameters.get(key) != POLICY[key] for key in ("method", "clip_min", "clip_max")):
        raise ValueError("Unsupported calibration method or clipping rule")
    coefficients = np.array([parameters["slope"], parameters["intercept"]], dtype=float)
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite calibration coefficients")
    return expit(coefficients[0] * _scores(probability).ravel() + coefficients[1])


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    """Include fixed-bin reliability counts alongside discrimination and loss."""
    from competition import metrics

    probability = np.asarray(probability, dtype=np.float64)
    _scores(probability)
    result = metrics(labels, probability)
    bins = np.minimum((probability * 10).astype(int), 9)
    reliability = []
    ece = 0.0
    for index in range(10):
        selected = bins == index
        count = int(selected.sum())
        mean = float(probability[selected].mean()) if count else None
        rate = float(labels[selected].mean()) if count else None
        reliability.append(dict(lower=index / 10, upper=(index + 1) / 10, count=count,
                                mean_predicted_probability=mean, observed_event_rate=rate))
        if count:
            ece += count / len(labels) * abs(mean - rate)
    return {**result, "ece": ece, "mean_predicted_probability": float(probability.mean()),
            "predicted_positives_at_0_5": int((probability >= 0.5).sum()), "reliability_bins": reliability}


def analyze(rows: list[dict], fit_prevalence: float, draws: int = 1000) -> tuple[dict, dict, list[dict]]:
    """Cross-fit on validation; final coefficients never use test scores or labels."""
    if draws < 1 or not 0 < fit_prevalence < 1:
        raise ValueError("Require positive bootstrap draws and an inner-fit prevalence in (0, 1)")
    parts = {name: [r for r in rows if r["partition"] == name] for name in ("validation", "test")}
    if sum(map(len, parts.values())) != len(rows):
        raise ValueError("Unexpected prediction partition")
    identities = {name: [r["borrower_id"] for r in values] for name, values in parts.items()}
    if any(not values or len(values) != len(set(values)) for values in identities.values()):
        raise ValueError("Missing or duplicate borrower identities")
    if set(identities["validation"]) & set(identities["test"]):
        raise ValueError("Calibration/test borrower overlap")
    labels = {name: np.array([int(r["target"]) for r in values], dtype=np.int32) for name, values in parts.items()}
    for name, y in labels.items():
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"Both classes required in {name}")
    if np.bincount(labels["validation"]).min() < 5:
        raise ValueError("Five-fold calibration requires at least five validation borrowers per class")
    raw = {name: {model: np.array([float(r[model]) for r in values]) for model in MODELS}
           for name, values in parts.items()}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(raw["validation"][MODELS[0]], labels["validation"]))
    fold_ids = np.empty(len(labels["validation"]), dtype=int)
    for index, (_, assessment) in enumerate(folds):
        fold_ids[assessment] = index
    report = {"policy": POLICY, "validation_role": "out-of-fold assessment; final coefficients fit all validation",
              "baseline_fit_prevalence": fit_prevalence, "models": {}, "baseline": {},
              "ece_caveat": "Fixed 0.1-wide bins pool most calibrated scores; inspect counts, especially sparse tails"}
    parameters, calibrated, fold_records = {}, {}, []
    for model in MODELS:
        validation = raw["validation"][model]
        oof = np.empty_like(validation)
        for index, (train, assessment) in enumerate(folds):
            coefficients = fit(validation[train], labels["validation"][train])
            oof[assessment] = apply(coefficients, validation[assessment])
            fold_records.append({"model": model, "fold": index,
                                 "fit_ids_hash": fingerprint([identities["validation"][i] for i in train]),
                                 "assessment_ids_hash": fingerprint([identities["validation"][i] for i in assessment]),
                                 "coefficients": coefficients})
        parameters[model] = {**fit(validation, labels["validation"]),
                             "fit_ids_hash": fingerprint(identities["validation"]),
                             "fit_labels_hash": fingerprint(labels["validation"].tolist())}
        test = apply(parameters[model], raw["test"][model])
        calibrated[model] = {"validation": oof, "test": test}
        before = probability_metrics(labels["test"], raw["test"][model])
        after = probability_metrics(labels["test"], test)
        report["models"][model] = {
            "validation_oof": {"raw": probability_metrics(labels["validation"], validation),
                               "calibrated": probability_metrics(labels["validation"], oof)},
            "test": {"raw": before, "calibrated": after},
            "ranking": {"positive_slope": parameters[model]["slope"] > 0,
                        "roc_auc_delta": after["roc_auc"] - before["roc_auc"],
                        "average_precision_delta": after["average_precision"] - before["average_precision"],
                        "clipped_inputs": int(np.count_nonzero((raw["test"][model] < CLIP) | (raw["test"][model] > 1 - CLIP)))}}
    for name, y in labels.items():
        report["baseline"][name] = probability_metrics(y, np.full(len(y), fit_prevalence))
    # Loss differences share one borrower draw across models; no calibrator refits.
    y = labels["test"]
    differences = {}
    samples = {model: {key: [] for key in ("brier", "log_loss")} for model in MODELS}
    for model in MODELS:
        a, b = raw["test"][model], calibrated[model]["test"]
        def losses(p: np.ndarray) -> np.ndarray:
            p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
            return -(y * np.log(p) + (1 - y) * np.log1p(-p))
        differences[model] = {"brier": (b - y) ** 2 - (a - y) ** 2, "log_loss": losses(b) - losses(a)}
    rng = np.random.default_rng(42)
    skipped = 0
    for _ in range(draws):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            skipped += 1
            continue
        for model in MODELS:
            for metric in samples[model]:
                samples[model][metric].append(float(differences[model][metric][index].mean()))
    if skipped == draws:
        raise ValueError("All calibration bootstrap draws were single-class")
    report["bootstrap"] = {"seed": 42, "draws_requested": draws, "draws_used": draws - skipped,
                           "single_class_draws_skipped": skipped, "direction": "calibrated minus raw",
                           "interpretation": "nominal conditional 95% paired intervals; no calibration-fit or training uncertainty",
                           "differences": {model: {metric: {"point": float(differences[model][metric].mean()),
                                                           "lower": float(np.percentile(values, 2.5)),
                                                           "upper": float(np.percentile(values, 97.5))}
                                                   for metric, values in metrics.items()} for model, metrics in samples.items()}}
    report["folds"] = fold_records
    predictions = []
    for name, values in parts.items():
        for index, row in enumerate(values):
            predictions.append({"borrower_id": row["borrower_id"], "partition": name, "target": int(row["target"]),
                                "calibration_fold": int(fold_ids[index]) if name == "validation" else "",
                                **{f"{model}_{kind}": float(raw[name][model][index] if kind == "raw" else calibrated[model][name][index])
                                   for model in MODELS for kind in ("raw", "calibrated")}})
    return report, parameters, predictions


def _equivalent(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _equivalent(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))
    return actual == expected


def write_analysis(run: Path, fit_prevalence: float) -> dict:
    """Publish calibrated results separately from the immutable raw-score schema."""
    from competition import file_hash, write_csv, write_json

    with (run / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report, parameters, predictions = analyze(rows, fit_prevalence)
    output = run / "calibration"
    output.mkdir(exist_ok=False)
    for model, coefficients in parameters.items():
        write_json(output / f"{model}.json", coefficients)
        restored = json.loads((output / f"{model}.json").read_text())
        raw = np.array([float(r[model]) for r in rows if r["partition"] == "test"])
        np.testing.assert_array_equal(apply(restored, raw), apply(coefficients, raw))
    write_csv(output / "predictions.csv", predictions)
    write_json(output / "metrics.json", report)
    write_json(output / "source.json", {"raw_predictions_sha256": file_hash(run / "predictions.csv"),
                                       "implementation_sha256": file_hash(Path(__file__))})
    return report


def validate_analysis(run: Path, fit_prevalence: float) -> dict:
    """Reproduce fits, out-of-fold assessment and test metrics from raw scores."""
    from competition import file_hash

    with (run / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = run / "calibration"
    source = json.loads((output / "source.json").read_text())
    if source != {"raw_predictions_sha256": file_hash(run / "predictions.csv"),
                  "implementation_sha256": file_hash(Path(__file__))}:
        raise ValueError("Calibration source identity mismatch")
    expected, parameters, predictions = analyze(rows, fit_prevalence)
    actual = json.loads((output / "metrics.json").read_text())
    if not _equivalent(actual, expected):
        raise ValueError("Calibration metrics or validation-fold reconstruction mismatch")
    for model, coefficients in parameters.items():
        if not _equivalent(json.loads((output / f"{model}.json").read_text()), coefficients):
            raise ValueError("Calibrator does not match validation-only fit")
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    if len(saved) != len(predictions):
        raise ValueError("Calibrated prediction row count mismatch")
    for row, expected_row in zip(saved, predictions):
        if set(row) != set(expected_row):
            raise ValueError("Calibrated prediction identity or probability mismatch")
        for key, value in expected_row.items():
            if isinstance(value, (str, int)):
                matches = row[key] == str(value)
            else:
                matches = _equivalent(float(row[key]), value)
            if not matches:
                raise ValueError("Calibrated prediction identity or probability mismatch")
    return {"status": "passed", "validation_only_coefficients_reproduced": True,
            "out_of_fold_assessment_reproduced": True, "calibrated_metrics_and_bootstrap_recalculated": True,
            "predictions_verified": True}
