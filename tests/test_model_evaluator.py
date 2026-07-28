"""Tests for seqcredit_model.credit_model.ModelEvaluator."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend before pyplot is touched

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from seqcredit_model.credit_model import ModelEvaluator


@pytest.fixture
def y_test():
    rng = np.random.RandomState(42)
    # 30 samples, imbalanced-ish: 20 negatives, 10 positives
    y = np.array([0] * 20 + [1] * 10)
    rng.shuffle(y)
    return y


@pytest.fixture
def fake_model_predictions(y_test):
    """Two fake models' predicted probabilities: one good, one near-random."""
    rng = np.random.RandomState(0)
    n = len(y_test)

    # "Good" model: probabilities correlated with true label.
    good_proba = np.clip(
        y_test * 0.6 + rng.uniform(0, 0.4, size=n), 0.0, 1.0
    )
    # "Bad" model: pure random noise, uncorrelated with label.
    bad_proba = rng.uniform(0, 1, size=n)

    return {"GoodModel": good_proba, "BadModel": bad_proba}


@pytest.fixture
def evaluator(y_test, fake_model_predictions):
    ev = ModelEvaluator(y_test)
    for name, proba in fake_model_predictions.items():
        ev.add_model(name, proba)
    return ev


class TestAddModel:
    def test_add_model_populates_results(self, evaluator):
        assert set(evaluator.results.keys()) == {"GoodModel", "BadModel"}

    def test_add_model_computes_expected_fields(self, evaluator):
        res = evaluator.results["GoodModel"]
        expected_keys = {
            "y_pred_proba",
            "y_pred",
            "auc_roc",
            "auc_pr",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "confusion_matrix",
            "fpr",
            "tpr",
            "precision_arr",
            "recall_arr",
            "bootstrap_ci",
        }
        assert expected_keys == set(res.keys())

    def test_metrics_in_valid_range(self, evaluator):
        for name, res in evaluator.results.items():
            for metric in ["auc_roc", "auc_pr", "f1", "precision", "recall", "accuracy"]:
                assert 0.0 <= res[metric] <= 1.0

    def test_good_model_outperforms_random_model_auc(self, evaluator):
        assert evaluator.results["GoodModel"]["auc_roc"] > evaluator.results["BadModel"]["auc_roc"]

    def test_y_pred_uses_half_threshold(self, y_test, fake_model_predictions):
        ev = ModelEvaluator(y_test)
        ev.add_model("GoodModel", fake_model_predictions["GoodModel"])
        expected = (fake_model_predictions["GoodModel"] >= 0.5).astype(int)
        assert np.array_equal(ev.results["GoodModel"]["y_pred"], expected)

    def test_bootstrap_ci_none_by_default(self, evaluator):
        assert evaluator.results["GoodModel"]["bootstrap_ci"] is None

    def test_bootstrap_ci_computed_when_requested(self, y_test, fake_model_predictions):
        ev = ModelEvaluator(y_test)
        ev.add_model(
            "GoodModel",
            fake_model_predictions["GoodModel"],
            compute_ci=True,
            n_bootstrap=50,
        )
        ci = ev.results["GoodModel"]["bootstrap_ci"]
        assert ci is not None
        assert set(ci.keys()) == {
            "AUC-ROC",
            "AUC-PR",
            "F1",
            "Precision",
            "Recall",
            "Accuracy",
        }
        for metric_stats in ci.values():
            assert set(metric_stats.keys()) == {"mean", "std", "ci_lower", "ci_upper"}
            assert metric_stats["ci_lower"] <= metric_stats["mean"] <= metric_stats["ci_upper"]


class TestGetComparisonTable:
    def test_comparison_table_has_right_models_and_columns(self, evaluator):
        table = evaluator.get_comparison_table()
        assert isinstance(table, pd.DataFrame)
        assert table.index.name == "Model"
        assert set(table.index) == {"GoodModel", "BadModel"}
        assert list(table.columns) == [
            "AUC-ROC",
            "AUC-PR",
            "F1",
            "Precision",
            "Recall",
            "Accuracy",
        ]

    def test_comparison_table_values_match_results(self, evaluator):
        table = evaluator.get_comparison_table()
        assert table.loc["GoodModel", "AUC-ROC"] == pytest.approx(
            evaluator.results["GoodModel"]["auc_roc"]
        )


class TestGetComparisonTableWithCI:
    def test_without_ci_data_columns_are_none(self, evaluator):
        table = evaluator.get_comparison_table_with_ci()
        assert set(table.index) == {"GoodModel", "BadModel"}
        assert "AUC-ROC_CI" in table.columns
        assert table.loc["GoodModel", "AUC-ROC_CI"] is None

    def test_with_ci_data_columns_formatted(self, y_test, fake_model_predictions):
        ev = ModelEvaluator(y_test)
        ev.add_model(
            "GoodModel",
            fake_model_predictions["GoodModel"],
            compute_ci=True,
            n_bootstrap=50,
        )
        table = ev.get_comparison_table_with_ci()
        ci_str = table.loc["GoodModel", "AUC-ROC_CI"]
        assert ci_str is not None
        assert ci_str.startswith("(") and ci_str.endswith(")")


class TestGetComparisonCsvWithCI:
    def test_csv_table_has_lower_upper_columns(self, y_test, fake_model_predictions):
        ev = ModelEvaluator(y_test)
        ev.add_model(
            "GoodModel",
            fake_model_predictions["GoodModel"],
            compute_ci=True,
            n_bootstrap=50,
        )
        table = ev.get_comparison_csv_with_ci()
        assert "AUC-ROC_lower" in table.columns
        assert "AUC-ROC_upper" in table.columns
        assert table.loc["GoodModel", "AUC-ROC_lower"] <= table.loc["GoodModel", "AUC-ROC_upper"]

    def test_csv_table_none_without_ci(self, evaluator):
        table = evaluator.get_comparison_csv_with_ci()
        assert table.loc["GoodModel", "AUC-ROC_lower"] is None


class TestFormatResultsWithCI:
    def test_formats_as_strings(self, evaluator):
        table = evaluator.format_results_with_ci()
        assert isinstance(table.loc["GoodModel", "AUC-ROC"], str)


class TestPlottingMethodsRunWithoutError:
    """These call the plotting methods with the Agg backend; we only assert
    they execute and return the expected object type, not pixel content."""

    def test_plot_roc_curves(self, evaluator):
        ax = evaluator.plot_roc_curves()
        assert ax is not None
        plt.close("all")

    def test_plot_pr_curves(self, evaluator):
        ax = evaluator.plot_pr_curves()
        assert ax is not None
        plt.close("all")

    def test_plot_confusion_matrices(self, evaluator):
        fig = evaluator.plot_confusion_matrices()
        assert fig is not None
        plt.close("all")

    def test_plot_threshold_analysis(self, evaluator):
        ax = evaluator.plot_threshold_analysis("GoodModel")
        assert ax is not None
        plt.close("all")

    def test_print_classification_reports_no_exception(self, evaluator, capsys):
        evaluator.print_classification_reports()
        captured = capsys.readouterr()
        assert "GoodModel" in captured.out
        assert "BadModel" in captured.out
