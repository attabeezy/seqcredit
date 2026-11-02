"""Isolated experiment comparing baseline balanced-loss training against pre-training minority upsampling."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from dataset_io import DATA_FILE, build, load
from experiment_v2 import load_contract, preprocess, sequence_batch


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, model_name: str, condition: str) -> dict:
    roc_auc = float(roc_auc_score(y_true, proba))
    ap = float(average_precision_score(y_true, proba))
    brier = float(brier_score_loss(y_true, proba))
    loss = float(log_loss(y_true, proba, labels=[0, 1]))

    # Threshold 0.5
    pred_05 = (proba >= 0.5).astype(int)
    cr_05 = classification_report(y_true, pred_05, target_names=["Repaid (0)", "Default (1)"], output_dict=True, zero_division=0)
    cm_05 = confusion_matrix(y_true, pred_05).tolist()

    # Prevalence threshold
    prevalence = float(y_true.mean())
    cutoff_prev = float(np.percentile(proba, 100 * (1 - prevalence)))
    pred_prev = (proba >= cutoff_prev).astype(int)
    cr_prev = classification_report(y_true, pred_prev, target_names=["Repaid (0)", "Default (1)"], output_dict=True, zero_division=0)
    cm_prev = confusion_matrix(y_true, pred_prev).tolist()

    return {
        "condition": condition,
        "model": model_name,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "brier": brier,
        "log_loss": loss,
        "pred_mean": float(proba.mean()),
        "threshold_05": {
            "report": cr_05,
            "cm": cm_05,
            "precision": cr_05["Default (1)"]["precision"],
            "recall": cr_05["Default (1)"]["recall"],
            "f1": cr_05["Default (1)"]["f1-score"],
        },
        "prevalence_threshold": {
            "cutoff": cutoff_prev,
            "report": cr_prev,
            "cm": cm_prev,
            "precision": cr_prev["Default (1)"]["precision"],
            "recall": cr_prev["Default (1)"]["recall"],
            "f1": cr_prev["Default (1)"]["f1-score"],
        },
    }


def train_and_eval(data: dict, split: dict, contract: dict, upsample: bool = False, seed: int = 42) -> dict:
    condition_name = "Upsampled (Pre-Training)" if upsample else "Baseline (Class Weights)"
    print(f"\n=======================================================")
    print(f" Training: {condition_name}")
    print(f"=======================================================")

    y = (data["labels"] == 2).astype(np.int32)
    fit_idx = split["fit"].copy()
    stop_idx = split["early_stop"]
    test_idx = split["test"]

    # Preprocessing is fit strictly on distinct original inner-fit borrowers
    transformed = preprocess(data, split)
    static_X = transformed["static"]

    fit_y = y[fit_idx]
    pos_mask = fit_y == 1
    neg_mask = fit_y == 0
    pos_fit = fit_idx[pos_mask]
    neg_fit = fit_idx[neg_mask]

    print(f"Original fit partition: {len(fit_idx):,} borrowers ({len(pos_fit)} defaults, {len(neg_fit)} repaid)")

    if upsample:
        # Random oversampling of minority defaults with replacement to reach 1:1 balance
        rng = np.random.default_rng(seed + 2026)
        num_needed = len(neg_fit) - len(pos_fit)
        resampled_pos = rng.choice(pos_fit, size=num_needed, replace=True)
        train_idx = np.concatenate([neg_fit, pos_fit, resampled_pos])
        rng.shuffle(train_idx)
        print(f"Upsampled fit partition: {len(train_idx):,} borrowers (1:1 balance: {len(neg_fit)} defaults, {len(neg_fit)} repaid)")
        # Class weights are uniform because data is already 1:1 balanced
        lr_class_weight = None
        lgb_class_weight = None
        use_sample_weights = False
    else:
        train_idx = fit_idx
        lr_class_weight = "balanced"
        lgb_class_weight = "balanced"
        use_sample_weights = True

    train_y = y[train_idx]
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[fit_idx])

    # 1. Logistic Regression
    print("-> Fitting Logistic Regression...")
    lr = LogisticRegression(class_weight=lr_class_weight, max_iter=1000, random_state=seed)
    lr.fit(static_X[train_idx], train_y)
    lr_test_probs = lr.predict_proba(static_X[test_idx])[:, 1]
    lr_metrics = evaluate_predictions(y[test_idx], lr_test_probs, "Logistic Regression", condition_name)

    # 2. LightGBM
    print("-> Fitting LightGBM...")
    from lightgbm import LGBMClassifier, early_stopping

    patience = contract["execution"]["early_stopping"]["patience"]
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        class_weight=lgb_class_weight,
        n_jobs=2,
        random_state=seed,
        verbosity=-1,
    )
    lgb.fit(
        static_X[train_idx],
        train_y,
        eval_set=[(static_X[stop_idx], y[stop_idx])],
        callbacks=[early_stopping(patience, verbose=False)],
    )
    lgb_test_probs = lgb.predict_proba(static_X[test_idx])[:, 1]
    lgb_metrics = evaluate_predictions(y[test_idx], lgb_test_probs, "LightGBM", condition_name)

    # 3. GRU
    print("-> Fitting GRU...")
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    seq_dim = transformed["sequence"].shape[1]
    seq_input = tf.keras.Input(shape=(None, seq_dim), name="sequence")
    mask_input = tf.keras.Input(shape=(None,), dtype="bool", name="valid_mask")
    hidden = tf.keras.layers.GRU(contract["execution"]["gru_units"])(seq_input, mask=mask_input)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)
    gru = tf.keras.Model(inputs={"sequence": seq_input, "valid_mask": mask_input}, outputs=output)
    gru.compile(
        optimizer=tf.keras.optimizers.Adam(contract["execution"]["gru_learning_rate"]),
        loss="binary_crossentropy",
    )

    batch_size = contract["execution"]["batch_size"]

    def make_tf_dataset(idx_list: np.ndarray, is_training: bool = False):
        batch_rng = np.random.default_rng(seed)

        def batches():
            order = idx_list.copy()
            if is_training:
                batch_rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                batch_inputs = sequence_batch(transformed, selected)
                if is_training:
                    if use_sample_weights:
                        yield batch_inputs, y[selected], weights[y[selected]].astype(np.float32)
                    else:
                        yield batch_inputs, y[selected], np.ones(len(selected), dtype=np.float32)
                else:
                    yield batch_inputs, y[selected]

        sig = (
            {"sequence": tf.TensorSpec((None, None, seq_dim), tf.float32), "valid_mask": tf.TensorSpec((None, None), tf.bool)},
            tf.TensorSpec((None,), tf.int32),
        )
        if is_training:
            sig = (*sig, tf.TensorSpec((None,), tf.float32))
        ds = tf.data.Dataset.from_generator(batches, output_signature=sig)
        ds = ds.apply(tf.data.experimental.assert_cardinality((len(idx_list) + batch_size - 1) // batch_size))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 2
        return ds.with_options(options)

    train_ds = make_tf_dataset(train_idx, is_training=True)
    stop_ds = make_tf_dataset(stop_idx, is_training=False)
    test_ds = make_tf_dataset(test_idx, is_training=False)

    gru.fit(
        train_ds,
        validation_data=stop_ds,
        epochs=contract["execution"]["early_stopping"]["epoch_cap"],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
                min_delta=0.0001,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    @tf.autograph.experimental.do_not_convert
    def pred_inp(inputs, labels):
        return inputs

    gru_test_probs = gru.predict(test_ds.map(pred_inp), verbose=0).ravel()
    gru_metrics = evaluate_predictions(y[test_idx], gru_test_probs, "GRU", condition_name)

    return {
        "logistic_regression": lr_metrics,
        "lightgbm": lgb_metrics,
        "gru": gru_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Test minority upsampling vs balanced weights.")
    parser.add_argument("--borrowers", type=int, default=5000, help="Borrower count (default: 5000 pilot)")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch cap for GRU (default: 10)")
    parser.add_argument("--use-accepted-data", action="store_true", help="Use full accepted dataset data/yara_telecel_dump.csv.gz")
    args = parser.parse_args()

    contract = deepcopy(load_contract())

    if args.use_accepted_data and DATA_FILE.exists():
        print(f"Loading full accepted dataset: {DATA_FILE}")
        data, split, _ = load(DATA_FILE, contract)
    else:
        contract["dataset"]["borrowers"] = args.borrowers
        contract["execution"]["early_stopping"]["epoch_cap"] = args.epochs
        print(f"Generating temporary isolated test dataset with {args.borrowers:,} borrowers...")
        with tempfile.TemporaryDirectory(prefix="yara-upsample-test-") as temp_dir:
            temp_path = Path(temp_dir) / "yara_telecel_dump.csv.gz"
            build(contract, temp_path)
            data, split, _ = load(temp_path, contract)

    print(f"Loaded dataset: {len(data['ids']):,} borrowers. Defaults: {(data['labels']==2).sum():,} ({(data['labels']==2).mean()*100:.2f}%)")

    # 1. Run Baseline (Class Weights)
    baseline_results = train_and_eval(data, split, contract, upsample=False)

    # 2. Run Upsampled (Pre-Training Resampling)
    upsampled_results = train_and_eval(data, split, contract, upsample=True)

    # Summary table
    print("\n" + "=" * 90)
    print(" COMPARATIVE RESULTS ON HELD-OUT TEST SET (Baseline vs. Pre-Training Upsampling)")
    print("=" * 90)
    print(f"{'Model':20} {'Condition':24} {'ROC-AUC':>8} {'Avg Prec':>9} {'Brier':>8} {'Log Loss':>9} {'F1 (prev)':>10}")
    print("-" * 90)

    for model_key in ["logistic_regression", "lightgbm", "gru"]:
        b = baseline_results[model_key]
        u = upsampled_results[model_key]
        model_name = b["model"]
        print(f"{model_name:20} {'Baseline (Weights)':24} {b['roc_auc']:8.4f} {b['average_precision']:9.4f} {b['brier']:8.4f} {b['log_loss']:9.4f} {b['prevalence_threshold']['f1']:10.4f}")
        print(f"{'':20} {'Pre-Training Upsampled':24} {u['roc_auc']:8.4f} {u['average_precision']:9.4f} {u['brier']:8.4f} {u['log_loss']:9.4f} {u['prevalence_threshold']['f1']:10.4f}")
        auc_diff = u['roc_auc'] - b['roc_auc']
        ap_diff = u['average_precision'] - b['average_precision']
        f1_diff = u['prevalence_threshold']['f1'] - b['prevalence_threshold']['f1']
        print(f"{'':20} {'Difference (Upsample - Base)':24} {auc_diff:+8.4f} {ap_diff:+9.4f} {'-':>8} {'-':>9} {f1_diff:+10.4f}")
        print("-" * 90)

    import json

    results_payload = {"baseline": baseline_results, "upsampled": upsampled_results}
    results_path = ROOT / "artifacts/upsampling_experiment_results.json"
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full results to: {results_path}")

    # Detailed Class-Specific Breakdown Table
    for threshold_mode, title in [("threshold_05", "AT FIXED 0.50 THRESHOLD"), ("prevalence_threshold", "AT PREVALENCE-MATCHED THRESHOLD")]:
        print("\n" + "=" * 105)
        print(f" CLASS-SPECIFIC BREAKDOWN {title} (Held-Out Test Set)")
        print("=" * 105)
        print(f"{'Model':20} {'Condition':22} {'Class':14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 105)
        for model_key in ["logistic_regression", "lightgbm", "gru"]:
            for cond_key, cond_label in [("baseline", "Baseline (Weights)"), ("upsampled", "Pre-Train Upsampled")]:
                res = results_payload[cond_key][model_key]
                report = res[threshold_mode]["report"]
                model_name = res["model"]
                for cls_name in ["Repaid (0)", "Default (1)"]:
                    p = report[cls_name]["precision"]
                    r = report[cls_name]["recall"]
                    f1 = report[cls_name]["f1-score"]
                    supp = int(report[cls_name]["support"])
                    print(f"{model_name:20} {cond_label:22} {cls_name:14} {p:10.4f} {r:10.4f} {f1:10.4f} {supp:10d}")
                    model_name = ""
                    cond_label = ""
            print("-" * 105)


if __name__ == "__main__":
    main()
