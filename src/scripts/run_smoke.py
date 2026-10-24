"""Execute one isolated v2 smoke or pilot run. No legacy data/cache reads."""

from __future__ import annotations

import argparse
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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from contract_checks import fingerprint, validate_feature_schema
from experiment_v2 import ROOT, generate, load_contract, partitions, preprocess


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def execution_settings(contract: dict, profile: str) -> dict:
    if profile not in {"smoke", "pilot", "main"}:
        raise ValueError("Unknown execution profile")
    stopping = contract["execution"]["early_stopping"]
    return dict(profile=profile, borrowers=contract["profiles"][profile]["borrowers"],
                sequence_length=contract["profiles"][profile]["max_sequence_length"],
                epochs=stopping["smoke_epoch_cap"] if profile == "smoke" else stopping["epoch_cap"],
                batch_size=contract["execution"]["batch_size"], patience=stopping["patience"],
                min_delta=stopping["min_delta"])


def run(destination: Path, seed: int, scenario: str, profile: str = "smoke",
        target: str = "primary", control: str = "none") -> dict:
    started = time.perf_counter()
    contract = load_contract()
    settings = execution_settings(contract, profile)
    if target not in {"primary", "secondary"}:
        raise ValueError("Unknown target")
    if control not in {"none", "label_permutation", "order_shuffle"}:
        raise ValueError("Unknown control")
    effective = {"contract": contract, "profile": profile, "resolved_execution": settings, "seed": seed,
                 "scenario": scenario, "target": target, "control": control,
                 "scope": "isolated v2 experiment run", "cache": "disabled",
                 "models": ["logistic_regression", "lightgbm", "small_gru"],
                 "targets": ["no_repayment" if target == "primary" else "penalty_or_no_repayment"]}
    write_json(destination / "effective_config.json", effective)
    source_files = [Path(__file__), ROOT / "src/scripts/experiment_v2.py", ROOT / "src/scripts/contract_checks.py",
                    ROOT / "src/configs/experiment_design.json", ROOT / contract["background_events"]["mapping"],
                    ROOT / "docs/calibration/ledger.json", ROOT / "uv.lock"]
    write_json(destination / "source_and_input_hashes.json", {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files})
    print(f"[1/5] Generating isolated {profile}: {settings['borrowers']} borrowers", flush=True)
    with (destination / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["borrower_id", "event_id", "day", "phase", "amount", "loan_id",
                                                   "category", "raw_type", "is_incoming", "counterparty"])
        writer.writeheader()
        data = generate(contract, profile, seed, scenario, event_sink=writer.writerows)
    generation_seconds = time.perf_counter() - started
    split = partitions(data["ids"], data["labels"], seed)
    y = (data["labels"] == 2 if target == "primary" else np.isin(data["labels"], [1, 2])).astype(np.int32)
    training_y = y.copy()
    if control == "label_permutation":
        training_y[split["fit"]] = np.random.default_rng(seed + 1009).permutation(training_y[split["fit"]])
    write_csv(destination / "event_derived_labels.csv", data["label_rows"])
    split_rows = []
    fit_indices = set(split["fit"])
    for name in ["train", "validation", "test"]:
        for i in split[name]:
            split_rows.append(dict(borrower_id=data["ids"][i], partition=name,
                                   training_role="fit" if i in fit_indices else "early_stop" if name == "train" else "heldout",
                                   row_index=int(i), label=int(data["labels"][i]),
                                   target=int(y[i]), training_target=int(training_y[i])))
    write_csv(destination / "split_ids.csv", split_rows)
    print("[2/5] Fitting preprocessing on inner-fit borrowers only", flush=True)
    transformed = preprocess(data, split, settings["sequence_length"])
    if control == "order_shuffle":
        shuffled = transformed["sequence"].copy()
        rng = np.random.default_rng(seed + 1013)
        for i, length in enumerate(transformed["lengths"]):
            valid = np.arange(settings["sequence_length"] - int(length), settings["sequence_length"])
            shuffled[i, valid] = shuffled[i, rng.permutation(valid)]
        transformed["sequence"] = shuffled
    validate_feature_schema(contract["features"]["static"], ["event_count", "total_amount", "mean_amount", "std_amount",
        "min_amount", "max_amount", "mean_gap_hours", "unique_counterparties", "incoming_share", "net_flow"] +
        ["share_" + name for name in contract["features"]["categories"]])
    assert data["static"].shape[1] == len(contract["features"]["static"])
    assert transformed["sequence"].shape[2] == len(contract["features"]["sequence"])
    write_json(destination / "feature_schema.json", contract["features"])
    joblib.dump(transformed["preprocessors"], destination / "preprocessors.joblib")
    limit = settings["sequence_length"]
    np.savez_compressed(destination / "raw_features.npz", ids=data["ids"], labels=data["labels"], static=data["static"],
                        sequence=np.stack([np.pad(s, ((limit-len(s), 0), (0, 0))) for s in data["sequences"]]), lengths=transformed["lengths"])
    np.savez_compressed(destination / "model_inputs.npz", ids=data["ids"], labels=y, static=transformed["static"],
                        sequence=transformed["sequence"], mask=transformed["mask"], lengths=transformed["lengths"])
    restored = joblib.load(destination / "preprocessors.joblib")
    np.testing.assert_allclose(restored["static"].transform(data["static"]), transformed["static"], rtol=0, atol=0)
    assert restored["fit_ids"] == data["ids"][split["fit"]].tolist()
    assert restored["static"].n_samples_seen_ == len(split["fit"])
    assert restored["sequence"].n_samples_seen_ == sum(len(data["sequences"][i]) for i in split["fit"])
    with np.load(destination / "model_inputs.npz", allow_pickle=False) as saved:
        for name in ["ids", "static", "sequence", "mask", "lengths"]:
            np.testing.assert_array_equal(saved[name], data["ids"] if name == "ids" else transformed[name])
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
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    seq_input = tf.keras.Input(shape=transformed["sequence"].shape[1:], name="sequence")
    mask_input = tf.keras.Input(shape=(transformed["sequence"].shape[1],), dtype="bool", name="valid_mask")
    hidden = tf.keras.layers.GRU(contract["execution"]["gru_units"])(seq_input, mask=mask_input)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
    sequence_model = tf.keras.Model(inputs={"sequence": seq_input, "valid_mask": mask_input}, outputs=output)
    sequence_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy")
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=training_y[fit])

    def dataset(idx: np.ndarray, training: bool = False):
        inputs = {"sequence": transformed["sequence"][idx], "valid_mask": transformed["mask"][idx]}
        payload = (inputs, training_y[idx], weights[training_y[idx]]) if training else (inputs, y[idx])
        ds = tf.data.Dataset.from_tensor_slices(payload)
        if training:
            ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 2
        return ds.batch(settings["batch_size"]).with_options(options)

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
    for name in ["validation", "test"]:
        idx = split[name]
        static_probability = static.predict_proba(transformed["static"][idx])[:, 1]
        boosted_probability = boosted.predict_proba(transformed["static"][idx])[:, 1]
        sequence_probability = sequence_model.predict(dataset(idx).map(lambda x, target: x), verbose=0).ravel()
        np.testing.assert_allclose(static_probability, reloaded_static.predict_proba(transformed["static"][idx])[:, 1], atol=1e-12)
        np.testing.assert_allclose(boosted_probability, reloaded_boosted.predict_proba(transformed["static"][idx])[:, 1], atol=1e-12)
        np.testing.assert_allclose(sequence_probability, reloaded_sequence.predict(dataset(idx).map(lambda x, target: x), verbose=0).ravel(), rtol=1e-6, atol=1e-6)
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
    peak = peak_memory_gib()
    artifact_size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if peak > contract["execution"]["initial_peak_process_memory_budget_gib"] or artifact_size > 2 * 1024**3:
        raise MemoryError(f"{profile} resource budget exceeded")
    write_json(destination / "timing_memory.json", dict(total_seconds=time.perf_counter()-started,
        generation_seconds=generation_seconds, logistic_fit_seconds=static_seconds,         lightgbm_fit_seconds=lightgbm_seconds, gru_fit_seconds=sequence_seconds,
        peak_process_working_set_gib=peak, artifact_bytes_before_manifest=artifact_size,
        measurement="OS peak working set checked after evaluation; not a hard runtime memory limiter"))
    write_json(destination / "audit_checks.json", dict(events_validated=True, coverage_days=[-90,60], shared_truncated_history=True,
        fit_ids_persisted=True, scalers_fit_only_on_inner_fit=True, caches_read=False, numeric_arrays_reload=True,
        model_reload_predictions_agree=True, test_used_for_fit_or_stopping=False,
        order_scenario_static_invariance_borrowers=data["order_invariance_checked"], event_count=data["event_count"],
        generator_calibration=data["calibration"], note="Negative-signal performance is not an acceptance threshold"))
    manifest = dict(status="passed", kind=f"v2_{profile}_not_full_benchmark", seed=seed, scenario=scenario,
        target=target, control=control,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        python=platform.python_version(), versions={name: version(name) for name in ["numpy", "scikit-learn", "tensorflow", "joblib"]},
        files={p.relative_to(destination).as_posix(): file_hash(p) for p in destination.rglob("*") if p.is_file()})
    write_json(destination / "manifest.json", manifest)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "pilot", "main"], default="smoke")
    parser.add_argument("--scenario", choices=["no_signal", "aggregate_signal", "order_signal"], default="no_signal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["primary", "secondary"], default="primary")
    parser.add_argument("--control", choices=["none", "label_permutation", "order_shuffle"], default="none")
    args = parser.parse_args()
    destination = ROOT / "data/runs" / (datetime.now(timezone.utc).strftime(args.profile + "-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {destination}", flush=True)
    try:
        run(destination, args.seed, args.scenario, args.profile, args.target, args.control)
    except Exception as error:
        write_json(destination / "failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__})
        raise
    print(f"PASS: {destination}", flush=True)


if __name__ == "__main__":
    main()
