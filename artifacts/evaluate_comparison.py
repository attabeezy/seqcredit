"""Reproduce the planned full-data controls and paired comparison report."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))

from competition import metrics, write_json
from dataset_io import DATA_FILE
from input_freeze import file_sha256, verify_accepted_dataset

MODELS = ("logistic_regression", "lightgbm", "gru")
METRICS = ("roc_auc", "average_precision", "brier", "log_loss")
CONDITIONS = ("headline", "label_permutation", "order_shuffle")


def verify_completed(path: Path, plan: dict, control: str) -> None:
    manifest = json.loads((path / "manifest.json").read_text())
    validation = json.loads((path / "validation.json").read_text())
    reference = json.loads((path / "data_reference.json").read_text())
    if manifest["status"] != "passed" or validation["status"] != "passed":
        raise ValueError(f"Run did not pass: {path}")
    config = json.loads((path / "effective_config.json").read_text())
    hashes = json.loads((path / "source_and_input_hashes.json").read_text())
    if (config["target"] != "primary" or config["control"] != control or
            hashes["src/scripts/competition.py"] != plan["training_source_sha256"]):
        raise ValueError("Run does not match the planned target, control or training implementation")
    if (reference["sha256"] != plan["dataset_sha256"] or
            reference["accepted_snapshot_sha256"] != plan["accepted_snapshot_sha256"]):
        raise ValueError("Runs must reference the same accepted dataset")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_sha256(target) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")


def align_predictions(runs: dict[str, Path]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    ids, labels, values = None, None, {}
    for condition in CONDITIONS:
        with (runs[condition] / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["partition"] == "test"]
        lookup = {row["borrower_id"]: row for row in rows}
        if len(lookup) != len(rows) or not rows:
            raise ValueError("Missing or repeated test borrower IDs")
        if ids is None:
            ids = [row["borrower_id"] for row in rows]
            labels = np.array([int(row["target"]) for row in rows], dtype=np.int32)
        if set(lookup) != set(ids):
            raise ValueError("Control test borrowers do not match the headline run")
        ordered = [lookup[uid] for uid in ids]
        np.testing.assert_array_equal(labels, [int(row["target"]) for row in ordered])
        for model in MODELS:
            values[f"{condition}/{model}"] = np.array(
                [float(row[model]) for row in ordered], dtype=np.float32 if model == "gru" else np.float64)
    return ids, labels, values


def comparisons() -> dict[str, tuple[str, str]]:
    pairs = {
        "gru_minus_logistic_regression": ("headline/gru", "headline/logistic_regression"),
        "gru_minus_lightgbm": ("headline/gru", "headline/lightgbm"),
        "lightgbm_minus_logistic_regression": ("headline/lightgbm", "headline/logistic_regression"),
    }
    for control in ("order_shuffle", "label_permutation"):
        for model in MODELS:
            pairs[f"{model}_headline_minus_{control}"] = (f"headline/{model}", f"{control}/{model}")
    for model in MODELS:
        pairs[f"{model}_label_permutation_minus_constant_reference"] = (
            f"label_permutation/{model}", "reference/constant_fit_prevalence")
    return pairs


def paired_bootstrap(labels: np.ndarray, values: dict[str, np.ndarray], draws: int = 1000) -> dict:
    if any(len(probability) != len(labels) for probability in values.values()):
        raise ValueError("Prediction lengths disagree")
    samples = {key: {metric: [] for metric in METRICS} for key in values}
    rng = np.random.default_rng(42)
    skipped = 0
    for draw in range(draws):
        index = rng.integers(0, len(labels), len(labels))
        y = labels[index]
        if np.unique(y).size < 2:
            skipped += 1
            continue
        for key, probability in values.items():
            scores = metrics(y, probability[index])
            for metric in METRICS:
                samples[key][metric].append(scores[metric])
        if draws >= 100 and (draw + 1) % 200 == 0:
            print(f"Paired bootstrap: {draw + 1}/{draws}", flush=True)
    if skipped == draws:
        raise ValueError("No usable bootstrap draws")

    def interval(sample: list[float] | np.ndarray) -> dict:
        return {"lower": float(np.percentile(sample, 2.5)),
                "upper": float(np.percentile(sample, 97.5)), "replicates": len(sample)}

    points = {key: metrics(labels, probability) for key, probability in values.items()}
    intervals = {key: {metric: {"point": points[key][metric], **interval(sample)}
                       for metric, sample in model.items()} for key, model in samples.items()}
    differences = {}
    for name, (first, second) in comparisons().items():
        differences[name] = {metric: {
            "point": points[first][metric] - points[second][metric],
            **interval(np.asarray(samples[first][metric]) - np.asarray(samples[second][metric])),
        } for metric in METRICS}
    return {"draws_requested": draws, "draws_used": draws - skipped,
            "single_class_draws_skipped": skipped, "seed": 42,
            "pairing": "Identical held-out borrower indices across all models and conditions within each draw",
            "intervals": intervals, "differences": differences}


def controls(plan: dict, main: Path) -> dict[str, Path]:
    paths = {"headline": main}
    execution = {"status": "running", "headline": main.name, "controls": []}
    for control in ("label_permutation", "order_shuffle"):
        path = main.with_name(main.name + "-" + control.replace("_", "-"))
        paths[control] = path
        if path.exists():
            verify_completed(path, plan, control)
            if not (path / "bootstrap_ci.json").is_file():
                raise ValueError("Incomplete control must be diagnosed before reuse")
            execution["controls"].append({"control": control, "run": path.name, "status": "passed"})
            continue
        verify_accepted_dataset(DATA_FILE)
        if file_sha256(ROOT / "src/scripts/competition.py") != plan["training_source_sha256"]:
            raise ValueError("Training implementation changed after the comparison plan")
        path.mkdir()
        log_path = path.with_name(path.name + ".log")
        print(f"Starting full-data {control}; log: {log_path}", flush=True)
        code = f"""import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'src/scripts')!r})
from competition import run, write_json
from validation import validate
from bootstrap_predictions import bootstrap
destination = Path({str(path)!r})
try:
    run(destination, control={control!r})
    write_json(destination / 'validation.json', validate(destination))
    write_json(destination / 'bootstrap_ci.json', bootstrap(destination, draws=1000, seed=42))
except Exception as error:
    write_json(destination / 'failure.json', {{'status':'failed','type':type(error).__name__,'error':str(error)}})
    raise
print('Control completed: {control}', flush=True)
"""
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            execution["status"] = "failed"
            execution["controls"].append({"control": control, "run": path.name, "status": "failed"})
            write_json(ROOT / "artifacts/comparison_execution.json", execution)
            raise RuntimeError(f"Control failed; inspect {log_path}")
        verify_completed(path, plan, control)
        execution["controls"].append({"control": control, "run": path.name, "status": "passed"})
        write_json(ROOT / "artifacts/comparison_execution.json", execution)
        print(f"Validated full-data {control}", flush=True)
    execution["status"] = "passed"
    write_json(ROOT / "artifacts/comparison_execution.json", execution)
    return paths


def report(runs: dict[str, Path], plan: dict) -> dict:
    ids, labels, values = align_predictions(runs)
    for model in ("logistic_regression", "lightgbm"):
        np.testing.assert_allclose(values[f"headline/{model}"], values[f"order_shuffle/{model}"], rtol=0, atol=1e-12)
    with np.load(runs["headline"] / "model_inputs.npz", allow_pickle=False) as archive:
        ordered_ids, training_labels = archive["ids"], archive["training_labels"]
    fit_ids = joblib.load(runs["headline"] / "preprocessors.joblib")["fit_ids"]
    lookup = {uid: i for i, uid in enumerate(ordered_ids)}
    fit_prevalence = float(training_labels[[lookup[uid] for uid in fit_ids]].mean())
    values["reference/constant_fit_prevalence"] = np.full(len(labels), fit_prevalence)
    boot = paired_bootstrap(labels, values, draws=plan["uncertainty"]["replicates"])
    points = {condition: json.loads((path / "metrics.json").read_text())["test"]
              for condition, path in runs.items()}
    order_ap = boot["differences"]["gru_headline_minus_order_shuffle"]["average_precision"]
    order_auc = boot["differences"]["gru_headline_minus_order_shuffle"]["roc_auc"]
    findings = {
        "highest_point_roc_auc": max(MODELS, key=lambda model: points["headline"][model]["roc_auc"]),
        "highest_point_average_precision": max(MODELS, key=lambda model: points["headline"][model]["average_precision"]),
        "static_order_shuffle_predictions_unchanged": True,
        "gru_order_average_precision_difference_interval_above_zero": order_ap["lower"] > 0,
        "gru_order_roc_auc_difference_interval_above_zero": order_auc["lower"] > 0,
        "gru_vs_static_intervals": {model: boot["differences"][f"gru_minus_{model}"]
                                    for model in ("logistic_regression", "lightgbm")},
        "constant_probability_from_inner_train": fit_prevalence,
        "label_permutation_vs_chance": {
            model: boot["differences"][f"{model}_label_permutation_minus_constant_reference"]
            for model in MODELS},
    }
    result = {"status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
              "plan": "comparison_plan.json", "dataset_sha256": plan["dataset_sha256"],
              "accepted_snapshot_sha256": plan["accepted_snapshot_sha256"],
              "test_borrowers": len(ids), "test_positives": int(labels.sum()),
              "test_prevalence": float(labels.mean()), "runs": {key: path.name for key, path in runs.items()},
              "prediction_sha256": {key: file_sha256(path / "predictions.csv") for key, path in runs.items()},
              "test_metrics": points, "uncertainty": boot, "findings": findings,
              "analysis_source_sha256": file_sha256(Path(__file__)),
              "resources": {condition: json.loads((path / "timing_memory.json").read_text())
                            for condition, path in runs.items()},
              "epochs_completed": {condition: len(json.loads((path / "training_history.json").read_text())["loss"])
                                   for condition, path in runs.items()},
              "limitations": ["One accepted synthetic population and one training seed",
                              "Intervals condition on these fitted models; they exclude training-run and generator uncertainty",
                              "Multiple nominal 95% intervals are descriptive; no familywise significance claim",
                              "No generator or model settings changed from held-out scores",
                              "Simulation comparison, not deployment validation"]}
    write_json(ROOT / "artifacts/comparison_results.json", result)

    def formatted(value: dict) -> str:
        return f"{value['point']:.4f} [{value['lower']:.4f}, {value['upper']:.4f}]"

    lines = ["# Full accepted-dataset comparison", "", f"Held-out borrowers: {len(ids):,}; positives: {int(labels.sum()):,} ({labels.mean():.2%}).", "",
             "All three experiments passed independent artifact/label/preprocessing/metric validation. Parameters, dataset and model settings stayed fixed.", "",
             "## Ranking metrics", "", "Point estimates with paired borrower 95% bootstrap intervals.", "",
             "| Experiment | Model | ROC-AUC | Average precision |", "| --- | --- | --- | --- |"]
    for condition in CONDITIONS:
        for model in MODELS:
            interval = boot["intervals"][f"{condition}/{model}"]
            lines.append(f"| {condition} | {model} | {formatted(interval['roc_auc'])} | {formatted(interval['average_precision'])} |")
    lines.extend(["", "## Paired model and control differences", "", "Positive ranking-metric differences favour the first model/experiment.", "",
                  "| Difference | ROC-AUC | Average precision |", "| --- | --- | --- |"])
    for name, value in boot["differences"].items():
        lines.append(f"| {name} | {formatted(value['roc_auc'])} | {formatted(value['average_precision'])} |")
    lines.extend(["", "## Probability quality", "", "Lower Brier score and log loss are better. The reference uses only the inner-training prevalence.", "",
                  "| Headline model | Brier score | Log loss |", "| --- | --- | --- |"])
    for model in MODELS:
        interval = boot["intervals"][f"headline/{model}"]
        lines.append(f"| {model} | {formatted(interval['brier'])} | {formatted(interval['log_loss'])} |")
    ref = boot["intervals"]["reference/constant_fit_prevalence"]
    lines.append(f"| constant fit-prevalence reference | {formatted(ref['brier'])} | {formatted(ref['log_loss'])} |")
    lines.extend(["", "## Interpretation", "",
                  f"Highest headline ROC-AUC point score: {findings['highest_point_roc_auc']}. Highest average-precision point score: {findings['highest_point_average_precision']}.", "",
                  f"GRU headline minus order-shuffle: ROC-AUC {formatted(order_auc)}; average precision {formatted(order_ap)}. Static predictions were unchanged by the order intervention.", "",
                  "An interval spanning zero does not establish a clear paired difference. The temporal-order hypothesis should be judged from the full-minus-shuffle contrast alongside the GRU-versus-static contrasts, with average precision primary and ROC-AUC secondary.", "",
                  "Label permutation is a diagnostic negative control; one point score need not equal chance exactly. Probability quality is reported separately from ranking discrimination; no test-set recalibration was performed.", "",
                  "These are nominal, conditional bootstrap intervals from one synthetic dataset and one seed. They do not quantify model-training or real-population uncertainty.", "",
                  "## Evidence", ""])
    for condition, path in runs.items():
        lines.append(f"- {condition}: [{path.name}]({path.name}/metrics.json), validation.json, bootstrap_ci.json, predictions.csv and fitted models.")
    lines.extend(["", "Machine-readable report: comparison_results.json. Evaluation plan: comparison_plan.json.", ""])
    (ROOT / "artifacts/comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    plan = json.loads((ROOT / "artifacts/comparison_plan.json").read_text())
    verify_accepted_dataset(DATA_FILE)
    headline = ROOT / "artifacts" / plan["headline_run"]
    verify_completed(headline, plan, "none")
    if not (headline / "bootstrap_ci.json").is_file():
        raise ValueError("Headline uncertainty calculation is not complete")
    paths = controls(plan, headline)
    result = report(paths, plan)
    print(json.dumps({"status": result["status"], "findings": result["findings"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
