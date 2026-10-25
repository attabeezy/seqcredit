"""Verify the recovered fixed comparison and produce research figures/evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset_io import DATA_FILE, load
from experiment_v2 import ROOT, load_contract
from input_freeze import file_sha256, verify_accepted_dataset
import validation

ART = ROOT / "artifacts"
MODELS = ("logistic_regression", "lightgbm", "gru")
NAMES = {"logistic_regression": "Logistic regression", "lightgbm": "LightGBM", "gru": "GRU"}


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluator():
    """Load the exact analysis implementation preserved with the experiment."""
    spec = importlib.util.spec_from_file_location("locked_comparison", ART / "evaluate_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify() -> tuple[dict, dict, np.ndarray, dict[str, np.ndarray]]:
    """Rederive data locally and verify all three remote results without refitting."""
    accepted = verify_accepted_dataset(DATA_FILE)
    plan = json.loads((ART / "comparison_plan.json").read_text())
    result = json.loads((ART / "comparison_results.json").read_text())
    if result["status"] != "passed" or accepted["dataset"]["sha256"] != result["dataset_sha256"]:
        raise ValueError("The comparison does not match the accepted dataset")
    analysis = evaluator()
    if file_sha256(ART / "evaluate_comparison.py") != result["analysis_source_sha256"]:
        raise ValueError("The preserved analysis source changed")
    runs = {name: ART / value for name, value in result["runs"].items()}
    checks = {}
    contract = load_contract()
    raw, split, local_reference = load(DATA_FILE, contract)
    original_loader = validation.load
    try:
        for condition, path in runs.items():
            print("Locally verifying", condition, flush=True)
            analysis.verify_completed(path, plan, "none" if condition == "headline" else condition)
            hashes = json.loads((path / "source_and_input_hashes.json").read_text())
            for relative, expected in hashes.items():
                if relative == "event_file_sha256":
                    continue
                source = (ROOT / relative).resolve()
                if not source.is_relative_to(ROOT.resolve()) or file_sha256(source) != expected:
                    raise ValueError("Local source differs from executed source: " + relative)
            remote_reference = json.loads((path / "data_reference.json").read_text())
            if {k: v for k, v in local_reference.items() if k != "path"} != {
                    k: v for k, v in remote_reference.items() if k != "path"}:
                raise ValueError("Local/remote data identities differ")

            def relocated_loader(requested: Path, requested_contract: dict) -> tuple:
                if str(requested).replace("\\", "/") != remote_reference["path"] or requested_contract != contract:
                    raise ValueError("Unexpected validation data or contract")
                return raw, split, {**local_reference, "path": remote_reference["path"]}

            validation.load = relocated_loader
            checks[condition] = validation.validate(path)
            if file_sha256(path / "predictions.csv") != result["prediction_sha256"][condition]:
                raise ValueError("Report prediction checksum mismatch")
            if json.loads((path / "metrics.json").read_text())["test"] != result["test_metrics"][condition]:
                raise ValueError("Report metrics differ from run metrics")
    finally:
        validation.load = original_loader
    ids, labels, values = analysis.align_predictions(runs)
    for model in MODELS[:2]:
        np.testing.assert_array_equal(values[f"headline/{model}"], values[f"order_shuffle/{model}"])
    if len(ids) != result["test_borrowers"] or int(labels.sum()) != result["test_positives"]:
        raise ValueError("Report test population mismatch")
    with np.load(runs["label_permutation"] / "model_inputs.npz", allow_pickle=False) as inputs:
        original = inputs["labels"][split["fit"]]
        permuted = inputs["training_labels"][split["fit"]]
        x_fit = inputs["static"][split["fit"]]
    n, positives = len(original), int(original.sum())
    overlap = int(np.sum((original == 1) & (permuted == 1)))
    static_models = {condition: joblib.load(path / "models/logistic_regression.joblib")
                     for condition, path in runs.items()}
    coef_a = static_models["headline"].coef_.ravel()
    coef_b = static_models["label_permutation"].coef_.ravel()
    diagnostics = {
        "status": "audited_without_new_fits",
        "fit_borrowers": n, "original_fit_positives": positives,
        "permuted_fit_positives": int(permuted.sum()),
        "positive_label_overlap": overlap,
        "expected_overlap_under_uniform_permutation": positives ** 2 / n,
        "original_vs_permuted_fit_label_correlation": float(np.corrcoef(original, permuted)[0, 1]),
        "headline_vs_permuted_logistic_coefficient_cosine": float(np.dot(coef_a, coef_b) / (np.linalg.norm(coef_a) * np.linalg.norm(coef_b))),
        "headline_vs_permuted_logistic_test_score_correlation": float(np.corrcoef(values["headline/logistic_regression"], values["label_permutation/logistic_regression"])[0, 1]),
        "feature_correlations_with_fit_labels": {
            feature: {"original": float(np.corrcoef(x_fit[:, i], original)[0, 1]),
                      "permuted": float(np.corrcoef(x_fit[:, i], permuted)[0, 1])}
            for i, feature in enumerate(contract["features"]["static"])},
        "permutation_rule": "Exact fixed inner-fit permutation independently rederived by validation; genuine inner-stop and evaluation labels retained",
        "interpretation": "Above-chance negative-control ranking is reported, not certified as resolved. A single permutation can align with predictive feature directions; genuine-label early stopping is an additional supervisory path for LightGBM/GRU. Conditional test bootstraps do not quantify permutation/training randomness. No new fits or seed selection performed."}
    write(ART / "negative_control_audit.json", diagnostics)
    # Independent ranking-only calculation: same locked predictions, RNG and
    # borrower pairing, without calling the experiment's metric/bootstrap code.
    points, samples = {}, {}
    for key, probability in values.items():
        points[key] = {"roc_auc": roc_auc_score(labels, probability),
                       "average_precision": average_precision_score(labels, probability)}
        samples[key] = {"roc_auc": [], "average_precision": []}
    rng = np.random.default_rng(result["uncertainty"]["seed"])
    used = 0
    for draw in range(result["uncertainty"]["draws_requested"]):
        indices = rng.integers(0, len(labels), len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        used += 1
        for key, probability in values.items():
            samples[key]["roc_auc"].append(roc_auc_score(y, probability[indices]))
            samples[key]["average_precision"].append(average_precision_score(y, probability[indices]))
    assert used == result["uncertainty"]["draws_used"]
    for key in values:
        for metric in ("roc_auc", "average_precision"):
            saved = result["uncertainty"]["intervals"][key][metric]
            np.testing.assert_allclose([points[key][metric], *np.quantile(samples[key][metric], [.025, .975])],
                                       [saved["point"], saved["lower"], saved["upper"]], rtol=1e-12, atol=1e-12)
    verified_contrasts = []
    for name, (first, second) in analysis.comparisons().items():
        if first not in values or second not in values:
            continue
        for metric in ("roc_auc", "average_precision"):
            delta = np.asarray(samples[first][metric]) - np.asarray(samples[second][metric])
            saved = result["uncertainty"]["differences"][name][metric]
            np.testing.assert_allclose([points[first][metric] - points[second][metric], *np.quantile(delta, [.025, .975])],
                                       [saved["point"], saved["lower"], saved["upper"]], rtol=1e-12, atol=1e-12)
        verified_contrasts.append(name)
    evidence = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": accepted["dataset"]["sha256"],
                "runs": checks, "source_hashes_match_executed_code": True,
                "test_ids_labels_and_predictions_aligned": True,
                "static_predictions_exactly_unchanged_by_order_shuffle": True,
                "ranking_bootstrap_independently_recomputed": True,
                "bootstrap_draws_used": used, "verified_paired_contrasts": verified_contrasts,
                "path_translation": "Only /content/seqcredit/data/... is mapped to the identical local accepted file; recorded artifacts remain unmodified",
                "comparison_results_sha256": file_sha256(ART / "comparison_results.json"),
                "verification_source_sha256": file_sha256(Path(__file__))}
    write(ART / "research_integrity.json", evidence)
    return result, evidence, labels, values


def figures(result: dict, labels: np.ndarray, values: dict[str, np.ndarray]) -> None:
    """Export standard scientific figures from the locked predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    directory = ART / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for model in MODELS:
        probability = values[f"headline/{model}"]
        fpr, tpr, _ = roc_curve(labels, probability)
        precision, recall, _ = precision_recall_curve(labels, probability)
        point = result["test_metrics"]["headline"][model]
        axes[0].plot(fpr, tpr, label=f"{NAMES[model]} ({point['roc_auc']:.3f})")
        axes[1].plot(recall, precision, label=f"{NAMES[model]} (AP {point['average_precision']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance ranking")
    axes[1].axhline(labels.mean(), color="black", linestyle="--", linewidth=1, label=f"Test prevalence {labels.mean():.3f}")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision–recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("Frozen simulation: 5,125 test borrowers, 210 nonrepayments")
    for extension in ("png", "svg"):
        fig.savefig(directory / f"ranking.{extension}")
    plt.close(fig)
    names = ["gru_minus_logistic_regression", "gru_minus_lightgbm", "gru_headline_minus_order_shuffle"]
    labels_plot = ["GRU − logistic regression", "GRU − LightGBM", "GRU − order-shuffled GRU"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", sharey=True)
    for axis, metric, title in zip(axes, ("average_precision", "roc_auc"), ("Average precision (primary)", "ROC-AUC (secondary)")):
        for i, name in enumerate(names):
            item = result["uncertainty"]["differences"][name][metric]
            axis.errorbar(item["point"], i, xerr=[[item["point"]-item["lower"]], [item["upper"]-item["point"]]], fmt="o", capsize=4, color="C0")
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Paired difference; positive favours full GRU", yticks=range(3), yticklabels=labels_plot)
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Nominal 95% paired borrower bootstrap intervals (1,000 draws)", fontsize=11)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"paired_differences.{extension}")
    plt.close(fig)
    write(ART / "research_figure_manifest.json", {
        "source_predictions": result["prediction_sha256"],
        "files": {path.relative_to(ART).as_posix(): file_sha256(path) for path in sorted(directory.iterdir()) if path.is_file()},
        "matplotlib_version": matplotlib.__version__,
        "analysis_source_sha256": file_sha256(Path(__file__))})


def main() -> None:
    result, evidence, labels, values = verify()
    figures(result, labels, values)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
