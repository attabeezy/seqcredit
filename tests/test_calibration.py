"""Calibration must generalize without fitting held-out borrowers or labels."""

import csv
import json

import numpy as np
import pytest
from scipy.special import expit

from calibration import MODELS, analyze, apply, fit, probability_metrics, validate_analysis, write_analysis


@pytest.fixture
def scores():
    rng = np.random.default_rng(42)
    rows = []
    for partition, size in (("validation", 300), ("test", 150)):
        risk = rng.normal(size=size)
        labels = (rng.random(size) < expit(-2 + risk)).astype(int)
        for i in range(size):
            rows.append({"borrower_id": f"{partition}-{i}", "partition": partition,
                         "target": int(labels[i]), **{m: float(expit(risk[i] + j * 0.1)) for j, m in enumerate(MODELS)}})
    return rows


def test_test_scores_and_labels_cannot_change_fitted_calibrators_or_oof(scores):
    report, parameters, predictions = analyze(scores, 0.12, draws=20)
    changed = [{**r, **({"target": 1 - r["target"], **{m: 1 - r[m] for m in MODELS}}
                        if r["partition"] == "test" else {})} for r in scores]
    other, fitted, outputs = analyze(changed, 0.12, draws=20)
    assert parameters == fitted
    assert report["folds"] == other["folds"]
    assert [r for r in predictions if r["partition"] == "validation"] == [
        r for r in outputs if r["partition"] == "validation"]
    for model in MODELS:
        assert abs(report["models"][model]["ranking"]["roc_auc_delta"]) < 1e-12
        assert abs(report["models"][model]["ranking"]["average_precision_delta"]) < 1e-12


def test_reliability_includes_endpoints_and_empty_bins():
    result = probability_metrics(np.array([0, 1]), np.array([0., 1.]))
    assert result["brier"] == result["ece"] == 0
    assert [b["count"] for b in result["reliability_bins"]] == [1] + [0] * 8 + [1]
    assert result["reliability_bins"][1]["observed_event_rate"] is None
    coefficients = fit(np.array([0., .2, .8, 1.]), np.array([0, 0, 1, 1]))
    assert np.isfinite(apply(coefficients, np.array([0., 1.]))).all()


def test_constant_scores_are_reported_without_requiring_discrimination(scores):
    constant = [{**row, **{m: .5 for m in MODELS}} for row in scores]
    report, parameters, _ = analyze(constant, .12, draws=20)
    for model in MODELS:
        assert parameters[model]["slope"] == 0
        assert not report["models"][model]["ranking"]["positive_slope"]
        assert report["models"][model]["test"]["calibrated"]["roc_auc"] == .5


def test_overlap_invalid_scores_and_insufficient_validation_fail(scores):
    overlap = [dict(r) for r in scores]
    overlap[-1]["borrower_id"] = overlap[0]["borrower_id"]
    with pytest.raises(ValueError, match="overlap"):
        analyze(overlap, .12, draws=20)
    for bad in (np.nan, np.inf, -0.01, 1.01):
        with pytest.raises(ValueError, match="probabilities"):
            fit(np.array([bad, .4]), np.array([0, 1]))
    insufficient = [{**r, "target": int(i == 0)} if r["partition"] == "validation" else r
                    for i, r in enumerate(scores)]
    with pytest.raises(ValueError, match="at least five"):
        analyze(insufficient, .12, draws=20)


def test_saved_calibration_reconstructs_and_detects_tampering(scores, tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    before = path.read_bytes()
    write_analysis(tmp_path, .12)
    assert path.read_bytes() == before
    assert validate_analysis(tmp_path, .12)["status"] == "passed"
    coefficients = tmp_path / "calibration/gru.json"
    payload = json.loads(coefficients.read_text())
    payload["intercept"] += .1
    coefficients.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation-only"):
        validate_analysis(tmp_path, .12)
