"""Independently verify v2 run artifacts; never tune models or rerun fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from contract_checks import derive_label, validate_events, validate_partitions
from run_smoke import metrics, write_json


ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_split_sizes(profile: str, contract: dict) -> list[int]:
    borrowers = contract["profiles"][profile]["borrowers"]
    train = int(borrowers * 0.6)
    return [train, (borrowers - train) // 2, borrowers - train - (borrowers - train) // 2]


def target_values(target: str, labels: np.ndarray) -> np.ndarray:
    if target == "primary":
        return (labels == 2).astype(np.int32)
    if target == "secondary":
        return np.isin(labels, [1, 2]).astype(np.int32)
    raise ValueError(f"Unknown target: {target}")


def validate(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "passed" or not manifest["kind"].startswith("v2_"):
        raise ValueError("Require a completed v2 run")
    for name, expected in manifest["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError("Manifest path outside run")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
    config = json.loads((run / "effective_config.json").read_text())
    contract = config["contract"]
    profile = config["profile"]
    target = config.get("target", "primary")
    control = config.get("control", "none")
    if profile not in contract["profiles"] or target not in {"primary", "secondary"}:
        raise ValueError("Run configuration is invalid")
    if manifest.get("target", target) != target or manifest.get("control", control) != control:
        raise ValueError("Manifest/config identity mismatch")
    expected_models = ["logistic_regression", "gru"]
    if "lightgbm" in config.get("models", []):
        expected_models.insert(1, "lightgbm")
    label_rows = read_rows(run / "event_derived_labels.csv")
    labels = {row["borrower_id"]: int(row["label"]) for row in label_rows}
    expected_borrowers = contract["profiles"][profile]["borrowers"]
    if len(labels) != expected_borrowers or len(labels) != len(label_rows):
        raise ValueError(f"{profile} borrower count/uniqueness mismatch")
    for row in label_rows:
        if [float(row[key]) for key in ["observation_start_day", "index_day", "followup_end_day"]] != [-90, 0, 60]:
            raise ValueError("Incomplete observation coverage")
    verified = set()
    with (run / "events.csv").open(newline="", encoding="utf-8") as handle:
        for uid, group in groupby(csv.DictReader(handle), key=lambda row: row["borrower_id"]):
            events = [{**row, "day": float(row["day"]), "amount": float(row["amount"])} for row in group]
            if uid in verified or uid not in labels:
                raise ValueError("Event borrower duplication or population mismatch")
            validate_events(events, [uid])
            loan_id = next(row["loan_id"] for row in events if row["phase"] == "index_loan")
            if derive_label(events, uid, loan_id) != labels[uid]:
                raise ValueError("Saved label disagrees with saved future events")
            verified.add(uid)
    if verified != set(labels):
        raise ValueError("Missing borrower events")
    rows = read_rows(run / "split_ids.csv")
    partitions = {name: [row["borrower_id"] for row in rows if row["partition"] == name]
                  for name in ["train", "validation", "test"]}
    fit = [row["borrower_id"] for row in rows if row["training_role"] == "fit"]
    stop = [row["borrower_id"] for row in rows if row["training_role"] == "early_stop"]
    validate_partitions(partitions, list(labels), fit, stop)
    if [len(partitions[name]) for name in ["train", "validation", "test"]] != expected_split_sizes(profile, contract):
        raise ValueError(f"{profile} split-size mismatch")
    has_training_targets = all("training_target" in row for row in rows)
    has_targets = all("target" in row for row in rows)
    for row in rows:
        expected = int(target_values(target, np.array([labels[row["borrower_id"]]]))[0])
        if not has_targets:
            row["target"] = str(expected)
        if int(row["target"]) != expected:
            raise ValueError("Split target disagrees with event-derived label")
        if not has_training_targets:
            row["training_target"] = row["target"]
    if control == "none" or control == "order_shuffle":
        if any(int(row["training_target"]) != int(row["target"]) for row in rows):
            raise ValueError("Unexpected training-target mutation")
    elif control == "label_permutation":
        if any(int(row["training_target"]) != int(row["target"]) for row in rows
               if row["training_role"] != "fit"):
            raise ValueError("Label permutation changed held-out training targets")
        fit_rows = [row for row in rows if row["training_role"] == "fit"]
        if sorted(int(row["training_target"]) for row in fit_rows) != sorted(int(row["target"]) for row in fit_rows):
            raise ValueError("Label permutation changed fit class counts")
    fitted = joblib.load(run / "preprocessors.joblib")
    if set(fitted["fit_ids"]) != set(fit) or len(fitted["fit_ids"]) != len(fit):
        raise ValueError("Preprocessing fit IDs do not match inner-fit borrowers")
    with np.load(run / "raw_features.npz", allow_pickle=False) as raw, np.load(run / "model_inputs.npz", allow_pickle=False) as inputs:
        raw = {key: raw[key] for key in raw.files}
        inputs = {key: inputs[key] for key in inputs.files}
        np.testing.assert_array_equal(raw["ids"], inputs["ids"])
        np.testing.assert_array_equal(raw["labels"], [labels[uid] for uid in raw["ids"]])
        np.testing.assert_array_equal(inputs["labels"], target_values(target, raw["labels"]))
        np.testing.assert_array_equal(raw["lengths"], inputs["lengths"])
        lookup = {uid: i for i, uid in enumerate(raw["ids"])}
        indices = [lookup[uid] for uid in fitted["fit_ids"]]
        reference = StandardScaler().fit(raw["static"][indices])
        for attribute in ["mean_", "scale_"]:
            np.testing.assert_allclose(getattr(reference, attribute), getattr(fitted["static"], attribute), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(reference.transform(raw["static"]), inputs["static"], atol=1e-12, rtol=1e-12)
        columns = [0, 2, 3]
        training_steps = np.concatenate([raw["sequence"][i, -int(raw["lengths"][i]):][:, columns] for i in indices])
        reference_seq = StandardScaler().fit(training_steps)
        for attribute in ["mean_", "scale_"]:
            np.testing.assert_allclose(getattr(reference_seq, attribute), getattr(fitted["sequence"], attribute), atol=1e-12, rtol=1e-12)
        expected_sequence = np.zeros_like(inputs["sequence"])
        for i, length in enumerate(raw["lengths"]):
            sequence = raw["sequence"][i, -int(length):].copy()
            sequence[:, columns] = reference_seq.transform(sequence[:, columns])
            expected_sequence[i, -int(length):] = sequence
        if control == "order_shuffle":
            rng = np.random.default_rng(config["seed"] + 1013)
            for i, length in enumerate(raw["lengths"]):
                valid = np.arange(expected_sequence.shape[1] - int(length), expected_sequence.shape[1])
                expected_sequence[i, valid] = expected_sequence[i, rng.permutation(valid)]
        for i, length in enumerate(raw["lengths"]):
            np.testing.assert_array_equal(inputs["sequence"][i, -int(length):], expected_sequence[i, -int(length):])
            if not inputs["mask"][i, -int(length):].all() or inputs["mask"][i, :-int(length)].any():
                raise ValueError("Explicit padding mask mismatch")
            if np.any(inputs["sequence"][i, :-int(length)] != 0):
                raise ValueError("Padding values are nonzero")
    predictions = read_rows(run / "predictions.csv")
    saved_metrics = json.loads((run / "metrics.json").read_text())
    if not predictions or any(model not in predictions[0] for model in expected_models):
        raise ValueError("Prediction artifact is missing a required model")
    for name in ["validation", "test"]:
        selected = [row for row in predictions if row["partition"] == name]
        if sorted(row["borrower_id"] for row in selected) != sorted(partitions[name]):
            raise ValueError("Prediction IDs do not match partition exactly")
        y = np.array([int(row["target"]) for row in selected])
        expected_y = target_values(target, np.array([labels[row["borrower_id"]] for row in selected]))
        if not np.array_equal(y, expected_y):
            raise ValueError("Prediction targets do not match labels")
        for model in expected_models:
            dtype = np.float32 if model == "gru" else np.float64
            recalculated = metrics(y, np.array([float(row[model]) for row in selected], dtype=dtype))
            for key, value in recalculated.items():
                if not np.isclose(value, saved_metrics[name][model][key], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Saved metric mismatch: {name}/{model}/{key}")
    return {"run": run.relative_to(ROOT).as_posix(), "profile": profile, "scenario": config["scenario"],
            "seed": config["seed"], "target": target, "control": control,
            "labels_rederived": len(labels), "manifest_files_verified": len(manifest["files"]),
            "split_and_prediction_identity": "passed", "metrics_recalculated": "passed",
            "preprocessing_rebuilt_from_inner_fit_raw_rows": "passed",
            "resources": json.loads((run / "timing_memory.json").read_text()), "test_metrics": saved_metrics["test"]}


def compare_order_inputs(base: Path, ordered: Path) -> dict:
    with np.load(base / "raw_features.npz", allow_pickle=False) as a, np.load(ordered / "raw_features.npz", allow_pickle=False) as b:
        a = {key: a[key] for key in a.files}
        b = {key: b[key] for key in b.files}
        np.testing.assert_array_equal(a["ids"], b["ids"])
        np.testing.assert_array_equal(a["lengths"], b["lengths"])
        np.testing.assert_allclose(a["static"], b["static"], rtol=1e-12, atol=1e-9)
        # Timestamp/gap columns remain fixed; payload multisets are preserved.
        np.testing.assert_array_equal(a["sequence"][:, :, [2, 3]], b["sequence"][:, :, [2, 3]])
        columns = [0, 1, *range(4, a["sequence"].shape[2])]
        changed = 0
        for i, length in enumerate(a["lengths"]):
            first = a["sequence"][i, -int(length):][:, columns]
            second = b["sequence"][i, -int(length):][:, columns]
            if sorted(map(tuple, first)) != sorted(map(tuple, second)):
                raise ValueError("Order scenario changed payload multiset")
            changed += int(not np.array_equal(first, second))
    return {"static_aggregates": "unchanged", "timestamps": "unchanged", "payload_multisets": "unchanged",
            "borrowers_with_changed_order": changed, "note": "Different outcome mechanisms can yield different stratified splits across scenarios; no paired test-score inference"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    args = parser.parse_args()
    runs = [path.resolve() for path in args.run]
    results = [validate(path) for path in runs]
    if len({result["scenario"] for result in results}) != len(results):
        raise ValueError("Choose exactly one run per scenario; no best-score selection")
    if len({result["seed"] for result in results}) != 1:
        raise ValueError("Cross-scenario input checks require the same seed")
    report = {"status": "passed", "scope": "independent v2 artifact validation",
              "runs": results}
    by_scenario = {result["scenario"]: path for result, path in zip(results, runs)}
    if {"no_signal", "order_signal"} <= set(by_scenario):
        report["order_invariance"] = compare_order_inputs(by_scenario["no_signal"], by_scenario["order_signal"])
    output = ROOT / "docs" / ("pilot_validation.json" if all(result["profile"] == "pilot" for result in results)
                              else "run_validation.json")
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
