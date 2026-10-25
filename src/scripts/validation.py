"""Verify event provenance, fitted preprocessing, labels and reported metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from competition import file_hash, metrics
from dataset_io import load


def validate(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("Require a completed competition")
    for name, digest in manifest["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    config = json.loads((run / "effective_config.json").read_text())
    reference = json.loads((run / "data_reference.json").read_text())
    raw, split, actual_reference = load(Path(reference["path"]), config["contract"])
    if reference != actual_reference or manifest["dataset"] != reference:
        raise ValueError("Event file reference mismatch")
    target, control = config["target"], config["control"]
    labels = (raw["labels"] == 2 if target == "primary" else np.isin(raw["labels"], [1, 2])).astype(np.int32)
    expected_training = labels.copy()
    if control == "label_permutation":
        expected_training[split["fit"]] = np.random.default_rng(42 + 1009).permutation(labels[split["fit"]])
    elif control not in {"none", "order_shuffle"}:
        raise ValueError("Unknown control")
    with np.load(run / "model_inputs.npz", allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(inputs["ids"], raw["ids"])
    np.testing.assert_array_equal(inputs["labels"], labels)
    np.testing.assert_array_equal(inputs["training_labels"], expected_training)
    fitted = joblib.load(run / "preprocessors.joblib")
    fit = split["fit"]
    np.testing.assert_array_equal(fitted["fit_ids"], raw["ids"][fit])
    static_scaler = StandardScaler().fit(raw["static"][fit])
    columns = [0, 2, 3]
    sequence_scaler = StandardScaler().fit(np.concatenate([raw["sequences"][i][:, columns] for i in fit]))
    for name, scaler in [("static", static_scaler), ("sequence", sequence_scaler)]:
        for attribute in ("mean_", "scale_"):
            np.testing.assert_allclose(getattr(fitted[name], attribute), getattr(scaler, attribute), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inputs["static"], static_scaler.transform(raw["static"]), rtol=1e-12, atol=1e-12)
    expected_sequence = np.zeros_like(inputs["sequence"])
    lengths = np.array([len(sequence) for sequence in raw["sequences"]], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    rng = np.random.default_rng(42 + 1013)
    for i, sequence in enumerate(raw["sequences"]):
        sequence = sequence.copy()
        sequence[:, columns] = sequence_scaler.transform(sequence[:, columns])
        if control == "order_shuffle":
            payload = [0, 1, *range(4, sequence.shape[1])]
            sequence[:, payload] = sequence[rng.permutation(len(sequence))][:, payload]
        expected_sequence[offsets[i]:offsets[i + 1]] = sequence
    np.testing.assert_array_equal(inputs["sequence"], expected_sequence)
    np.testing.assert_array_equal(inputs["offsets"], offsets)
    np.testing.assert_array_equal(inputs["lengths"], lengths)
    with (run / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    saved = json.loads((run / "metrics.json").read_text())
    lookup = {uid: i for i, uid in enumerate(raw["ids"])}
    for partition in ("validation", "test"):
        rows = [row for row in predictions if row["partition"] == partition]
        if sorted(row["borrower_id"] for row in rows) != sorted(raw["ids"][split[partition]]):
            raise ValueError("Prediction partition identity mismatch")
        y = np.array([int(row["target"]) for row in rows])
        np.testing.assert_array_equal(y, [labels[lookup[row["borrower_id"]]] for row in rows])
        for model in ("logistic_regression", "lightgbm", "gru"):
            probability = np.array([float(row[model]) for row in rows], dtype=np.float32 if model == "gru" else np.float64)
            recalculated = metrics(y, probability)
            for key, value in recalculated.items():
                if not np.isclose(value, saved[partition][model][key], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Metric mismatch: {partition}/{model}/{key}")
    if len(predictions) != len(split["validation"]) + len(split["test"]):
        raise ValueError("Unexpected prediction rows")
    calibration = None
    if "posthoc_calibration" in config and control == "none":
        from calibration import POLICY, validate_analysis
        if config["posthoc_calibration"] != POLICY:
            raise ValueError("Unexpected calibration policy")
        calibration = validate_analysis(run, float(labels[fit].mean()))
    return {"status": "passed", "borrowers": len(raw["ids"]), "events": raw["event_count"],
            "event_labels_rederived": True, "preprocessing_rebuilt_from_inner_train": True,
            "artifact_hashes_verified": True, "metrics_recalculated": True,
            **({"calibration": calibration} if calibration is not None else {})}
