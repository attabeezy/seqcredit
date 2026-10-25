"""Fixed-model competition using one event file and train-only preprocessing."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from dataset_io import DATA_FILE, build, load
from contract_checks import fingerprint, validate_feature_schema
from experiment_v2 import ROOT, load_contract, preprocess, sequence_batch
from calibration import POLICY as CALIBRATION_POLICY, write_analysis


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def peak_memory_gib() -> float:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]

        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if not get_memory(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            raise OSError("Cannot measure peak working set")
        return counter.PeakWorkingSetSize / 1024**3
    import resource
    import sys
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    if probability.shape != y.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid model probabilities")
    predicted = probability >= 0.5
    return {"borrowers": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)), "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0))}


def execution_settings(contract: dict) -> dict:
    stopping = contract["execution"]["early_stopping"]
    return dict(borrowers=contract["dataset"]["borrowers"],
                history="all eligible events",
                epochs=stopping["epoch_cap"], batch_size=contract["execution"]["batch_size"],
                patience=stopping["patience"], min_delta=stopping["min_delta"])


def run(destination: Path, contract: dict | None = None, data_path: Path = DATA_FILE,
        target: str = "primary", control: str = "none") -> dict:
    started = time.perf_counter()
    contract = load_contract() if contract is None else contract
    seed = 42
    settings = execution_settings(contract)
    if target not in {"primary", "secondary"}:
        raise ValueError("Unknown target")
    if control not in {"none", "label_permutation", "order_shuffle"}:
        raise ValueError("Unknown control")
    print("[1/5] Reading the event dataset", flush=True)
    load_started = time.perf_counter()
    if not data_path.exists():
        build(contract, data_path)
    data, split, reference = load(data_path, contract)
    dataset_load_seconds = time.perf_counter() - load_started
    y = (data["labels"] == 2 if target == "primary" else np.isin(data["labels"], [1, 2])).astype(np.int32)
    training_y = y.copy()
    if control == "label_permutation":
        training_y[split["fit"]] = np.random.default_rng(seed + 1009).permutation(training_y[split["fit"]])
    effective = {"contract": contract, "resolved_execution": settings, "seed": seed,
                 "target": target, "control": control,
                 "scope": "isolated v2 experiment run", "cache": "canonical_read_only",
                 "dataset": reference,
                 "models": ["logistic_regression", "lightgbm", "small_gru"],
                 "targets": ["no_repayment" if target == "primary" else "penalty_or_no_repayment"]}
    effective["posthoc_calibration"] = (CALIBRATION_POLICY if control == "none" else
                                       {"status": "not_applied", "reason": "preserve raw negative-control interpretation"})
    effective["reported_probability_metrics"] = "calibration/metrics.json" if control == "none" else "metrics.json"
    write_json(destination / "effective_config.json", effective)
    write_json(destination / "data_reference.json", reference)
    source_files = [Path(__file__), ROOT / "src/scripts/experiment_v2.py", ROOT / "src/scripts/contract_checks.py",
                    ROOT / "src/scripts/dataset_io.py",
                    ROOT / "src/configs/experiment_design.json", ROOT / contract["background_events"]["mapping"],
                    ROOT / "docs/calibration/ledger.json", ROOT / "uv.lock"]
    source_files.append(ROOT / "src/scripts/calibration.py")
    if "accepted_snapshot_sha256" in reference:
        source_files.extend([ROOT / "src/configs/accepted_dataset.json", ROOT / "src/scripts/input_freeze.py"])
    write_json(destination / "source_and_input_hashes.json", {
        **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files},
        "event_file_sha256": reference["sha256"],
    })
    print("[2/5] Fitting preprocessing on inner-fit borrowers only", flush=True)
    transformed = preprocess(data, split)
    if control == "order_shuffle":
        shuffled = transformed["sequence"].copy()
        rng = np.random.default_rng(seed + 1013)
        for start, end in zip(transformed["offsets"][:-1], transformed["offsets"][1:]):
            valid = np.arange(start, end)
            payload = [0, 1, *range(4, shuffled.shape[1])]
            shuffled[np.ix_(valid, payload)] = shuffled[np.ix_(rng.permutation(valid), payload)]
        transformed["sequence"] = shuffled
    validate_feature_schema(contract["features"]["static"], ["event_count", "total_amount", "mean_amount", "std_amount",
        "min_amount", "max_amount", "mean_gap_hours", "unique_counterparties", "incoming_share", "net_flow"] +
        ["share_" + name for name in contract["features"]["categories"]])
    assert data["static"].shape[1] == len(contract["features"]["static"])
    assert transformed["sequence"].shape[1] == len(contract["features"]["sequence"])
    write_json(destination / "feature_schema.json", contract["features"])
    joblib.dump(transformed["preprocessors"], destination / "preprocessors.joblib")
    np.savez_compressed(destination / "model_inputs.npz", ids=data["ids"], labels=y, training_labels=training_y,
                        static=transformed["static"], sequence=transformed["sequence"], offsets=transformed["offsets"],
                        lengths=transformed["lengths"])
    restored = joblib.load(destination / "preprocessors.joblib")
    np.testing.assert_allclose(restored["static"].transform(data["static"]), transformed["static"], rtol=0, atol=0)
    assert restored["fit_ids"] == data["ids"][split["fit"]].tolist()
    assert restored["static"].n_samples_seen_ == len(split["fit"])
    assert restored["sequence"].n_samples_seen_ == sum(len(data["sequences"][i]) for i in split["fit"])
    with np.load(destination / "model_inputs.npz", allow_pickle=False) as saved:
        for name in ["ids", "static", "sequence", "offsets", "lengths", "training_labels"]:
            np.testing.assert_array_equal(saved[name], data["ids"] if name == "ids" else training_y if name == "training_labels" else transformed[name])
    fit, stop = split["fit"], split["early_stop"]
    print("[3/5] Training logistic regression", flush=True)
    fit_started = time.perf_counter()
    static = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    static.fit(transformed["static"][fit], training_y[fit])
    (destination / "models").mkdir()
    joblib.dump(static, destination / "models/logistic_regression.joblib")
    static_seconds = time.perf_counter() - fit_started
    print("[3/5] Training LightGBM", flush=True)
    from lightgbm import LGBMClassifier, early_stopping
    fit_started = time.perf_counter()
    boosted = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        class_weight="balanced",
        n_jobs=2,
        random_state=seed,
        verbosity=-1,
    )
    boosted.fit(
        transformed["static"][fit],
        training_y[fit],
        eval_set=[(transformed["static"][stop], y[stop])],
        callbacks=[early_stopping(settings["patience"], verbose=False)],
    )
    joblib.dump(boosted, destination / "models/lightgbm.joblib")
    lightgbm_seconds = time.perf_counter() - fit_started
    print("[4/5] Training GRU with explicit masks and separate early-stop borrowers", flush=True)
    import tensorflow as tf
    if tf.config.threading.get_intra_op_parallelism_threads() != 2:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    seq_input = tf.keras.Input(shape=(None, transformed["sequence"].shape[1]), name="sequence")
    mask_input = tf.keras.Input(shape=(None,), dtype="bool", name="valid_mask")
    hidden = tf.keras.layers.GRU(contract["execution"]["gru_units"])(seq_input, mask=mask_input)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
    sequence_model = tf.keras.Model(inputs={"sequence": seq_input, "valid_mask": mask_input}, outputs=output)
    sequence_model.compile(optimizer=tf.keras.optimizers.Adam(contract["execution"]["gru_learning_rate"]), loss="binary_crossentropy")
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=training_y[fit])

    def dataset(idx: np.ndarray, training: bool = False):
        rng = np.random.default_rng(seed)
        batch_size = settings["batch_size"]

        def batches():
            order = idx.copy()
            if training:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                inputs = sequence_batch(transformed, selected)
                if training:
                    yield inputs, training_y[selected], weights[training_y[selected]].astype(np.float32)
                else:
                    yield inputs, y[selected]

        signature = ({"sequence": tf.TensorSpec((None, None, transformed["sequence"].shape[1]), tf.float32),
                      "valid_mask": tf.TensorSpec((None, None), tf.bool)}, tf.TensorSpec((None,), tf.int32))
        if training:
            signature = (*signature, tf.TensorSpec((None,), tf.float32))
        ds = tf.data.Dataset.from_generator(batches, output_signature=signature)
        ds = ds.apply(tf.data.experimental.assert_cardinality((len(idx) + batch_size - 1) // batch_size))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 2
        return ds.with_options(options)

    fit_started = time.perf_counter()
    history = sequence_model.fit(dataset(fit, True), validation_data=dataset(stop), epochs=settings["epochs"], shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=settings["patience"],
                                                    min_delta=settings["min_delta"], restore_best_weights=True)], verbose=2)
    sequence_seconds = time.perf_counter() - fit_started
    sequence_model.save(destination / "models/gru.keras")
    reloaded_sequence = tf.keras.models.load_model(destination / "models/gru.keras")
    reloaded_static = joblib.load(destination / "models/logistic_regression.joblib")
    write_json(destination / "training_history.json", {**{k: [float(v) for v in values] for k, values in history.history.items()},
                                                        "early_stop_ids_hash": fingerprint(data["ids"][stop].tolist()),
                                                        "class_weights_from_fit_only": weights.tolist()})
    print("[5/5] Evaluating fixed settings and checking model/artifact reloads", flush=True)
    reloaded_boosted = joblib.load(destination / "models/lightgbm.joblib")
    results, prediction_rows = {}, []

    @tf.autograph.experimental.do_not_convert
    def prediction_inputs(inputs, labels):
        return inputs

    for name in ["validation", "test"]:
        idx = split[name]
        static_probability = static.predict_proba(transformed["static"][idx])[:, 1]
        boosted_probability = boosted.predict_proba(transformed["static"][idx])[:, 1]
        sequence_probability = sequence_model.predict(dataset(idx).map(prediction_inputs), verbose=0).ravel()
        np.testing.assert_allclose(static_probability, reloaded_static.predict_proba(transformed["static"][idx])[:, 1], atol=1e-12)
        np.testing.assert_allclose(boosted_probability, reloaded_boosted.predict_proba(transformed["static"][idx])[:, 1], atol=1e-12)
        np.testing.assert_allclose(sequence_probability, reloaded_sequence.predict(dataset(idx).map(prediction_inputs), verbose=0).ravel(), rtol=1e-6, atol=1e-6)
        results[name] = {
            "logistic_regression": metrics(y[idx], static_probability),
            "lightgbm": metrics(y[idx], boosted_probability),
            "gru": metrics(y[idx], sequence_probability),
        }
        for j, i in enumerate(idx):
            prediction_rows.append(dict(borrower_id=data["ids"][i], partition=name, target=int(y[i]),
                                        logistic_regression=float(static_probability[j]),
                                        lightgbm=float(boosted_probability[j]), gru=float(sequence_probability[j])))
    write_csv(destination / "predictions.csv", prediction_rows)
    write_json(destination / "metrics.json", results)
    if control == "none":
        print("Calibrating probabilities on validation borrowers (five-fold assessment)", flush=True)
        write_analysis(destination, float(y[fit].mean()))
    peak = peak_memory_gib()
    artifact_size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if peak > contract["execution"]["initial_peak_process_memory_budget_gib"] or artifact_size > contract["execution"]["initial_run_artifact_budget_gib"] * 1024**3:
        raise MemoryError("Resource budget exceeded")
    write_json(destination / "timing_memory.json", dict(total_seconds=time.perf_counter()-started,
        dataset_load_seconds=dataset_load_seconds, logistic_fit_seconds=static_seconds, lightgbm_fit_seconds=lightgbm_seconds,
        gru_fit_seconds=sequence_seconds, peak_process_working_set_gib=peak,
        artifact_bytes_before_manifest=artifact_size,
        measurement="OS peak working set checked after evaluation; not a hard runtime memory limiter"))
    write_json(destination / "audit_checks.json", dict(events_validated=True, coverage_days=[-90,60], shared_complete_history=True, packed_sequences=True, per_batch_padding=True,
        fit_ids_persisted=True, scalers_fit_only_on_inner_fit=True, caches_read=False, numeric_arrays_reload=True,
        dataset_provenance_verified=True, dataset_path=reference["path"],
        dataset_sha256=reference["sha256"], input_hashes_verified=True,
        model_reload_predictions_agree=True, test_used_for_fit_or_stopping=False,
        event_count=data["event_count"], history_lengths={"min": int(transformed["lengths"].min()),
            "mean": float(transformed["lengths"].mean()), "max": int(transformed["lengths"].max()),
            "total": int(transformed["lengths"].sum())},
        generator_calibration=data["calibration"], note="No required winner or performance acceptance threshold"))
    manifest = dict(status="passed", kind="v2_model_competition", seed=seed,
        target=target, control=control, dataset=reference,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        python=platform.python_version(), versions={name: version(name) for name in ["numpy", "scikit-learn", "tensorflow", "joblib"]},
        files={p.relative_to(destination).as_posix(): file_hash(p) for p in destination.rglob("*") if p.is_file()})
    write_json(destination / "manifest.json", manifest)
    return results


def main() -> None:
    destination = ROOT / "artifacts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    print(f"Results: {destination}", flush=True)
    try:
        results = run(destination)
        from validation import validate
        from bootstrap_predictions import bootstrap
        write_json(destination / "validation.json", validate(destination))
        write_json(destination / "bootstrap_ci.json", bootstrap(destination, draws=1000, seed=42))
    except Exception as error:
        write_json(destination / "failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__})
        raise
    calibrated = json.loads((destination / "calibration/metrics.json").read_text())
    print("\nHeld-out test results (validation-calibrated probabilities):")
    print(f"{'Model':24} {'ROC-AUC':>9} {'Avg precision':>14} {'Brier':>9} {'Log loss':>9} {'ECE':>9}")
    for name in sorted(results["test"], key=lambda item: calibrated["models"][item]["test"]["calibrated"]["average_precision"], reverse=True):
        values = calibrated["models"][name]["test"]["calibrated"]
        print(f"{name:24} {values['roc_auc']:9.3f} {values['average_precision']:14.3f} "
              f"{values['brier']:9.4f} {values['log_loss']:9.4f} {values['ece']:9.4f}")
        if not calibrated["models"][name]["ranking"]["positive_slope"]:
            print(f"  {name}: validation sigmoid has nonpositive slope; see recorded ranking changes.")
    print("Raw scores: metrics.json / predictions.csv; calibrated scores and coefficients: calibration/")
    print("ECE uses 10 equal-width bins; inspect bin counts. Threshold remains 0.5.")
    print("Calibration evaluation is a follow-up on the previously inspected synthetic test set.")
    print(f"\nSaved: {destination}")


if __name__ == "__main__":
    main()
