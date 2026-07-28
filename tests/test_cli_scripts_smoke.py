"""Lightweight smoke tests for the CLI orchestration / benchmark scripts.

These scripts (run_cv_benchmark, run_ablation_study, run_hyperparameter_tuning,
run_full_benchmark, compute_bootstrap_ci, build_poster) are expensive to run
for real (the full CV benchmark alone takes ~75 minutes per CLAUDE.md). This
file intentionally does NOT validate benchmark numbers or research
conclusions. It only checks:

- the module imports cleanly (no import-time errors)
- where a script exposes an importable core function that does real work,
  that function runs correctly end-to-end on a tiny/fast synthetic input
- where a script's logic is not easily separable from main() / hardcoded
  paths, only an import-only smoke check is done (documented per-script)

Keep this whole file fast (well under a few minutes to run).
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared tiny synthetic data for CV-style functions (independent of the
# tiny_synthetic_dataset fixture, to keep these tests fast and self-contained)
# ---------------------------------------------------------------------------
def _make_tiny_xy(n=60, n_features=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Make y weakly dependent on X so models have something to learn but
    # AUC-ROC / metrics computations stay well-defined (both classes present).
    y = (X[:, 0] + rng.randn(n) * 0.5 > 0).astype(int)
    if y.sum() < 2:
        y[:2] = 1
    if (y == 0).sum() < 2:
        y[:2] = 0
    return X, y


# ---------------------------------------------------------------------------
# run_cv_benchmark.py
# ---------------------------------------------------------------------------
class TestRunCvBenchmark:
    def test_imports_cleanly(self):
        mod = importlib.import_module("seqcredit_model.run_cv_benchmark")
        assert hasattr(mod, "main")

    def test_compute_metrics_shape(self):
        from seqcredit_model.run_cv_benchmark import compute_metrics

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.4])
        metrics = compute_metrics(y_true, y_proba)
        expected_keys = {
            "auc_roc",
            "auc_pr",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "brier",
            "ece",
        }
        assert set(metrics.keys()) == expected_keys
        for v in metrics.values():
            assert isinstance(v, float)

    def test_compute_ece_bounds(self):
        from seqcredit_model.run_cv_benchmark import compute_ece

        y_true = np.array([0, 1, 0, 1] * 10)
        y_proba = np.linspace(0.01, 0.99, 40)
        ece = compute_ece(y_true, y_proba)
        assert 0.0 <= ece <= 1.0

    def test_run_static_model_cv_logistic_regression(self):
        from seqcredit_model.credit_model import LogisticRegressionModel
        from seqcredit_model.run_cv_benchmark import run_static_model_cv

        X, y = _make_tiny_xy()
        fold_df, summary, oof = run_static_model_cv(
            LogisticRegressionModel,
            {"class_weight": "balanced", "random_state": 42},
            X,
            y,
            n_splits=2,
            seed=42,
        )
        assert isinstance(fold_df, pd.DataFrame)
        assert len(fold_df) == 2
        assert "auc_roc" in fold_df.columns
        assert "auc_roc" in summary and "auc_roc_std" in summary
        assert oof.shape == (len(y),)
        # every OOF slot should have been filled by exactly one fold
        assert not np.any(np.isnan(oof))

    def test_run_static_model_cv_random_forest(self):
        from seqcredit_model.credit_model import RandomForestModel
        from seqcredit_model.run_cv_benchmark import run_static_model_cv

        X, y = _make_tiny_xy()
        fold_df, summary, oof = run_static_model_cv(
            RandomForestModel,
            {"n_estimators": 10, "max_depth": 3, "class_weight": "balanced", "random_state": 42},
            X,
            y,
            n_splits=2,
            seed=42,
        )
        assert len(fold_df) == 2
        assert 0.0 <= summary["auc_roc"] <= 1.0

    def test_run_static_model_cv_unknown_model_raises(self):
        from seqcredit_model.run_cv_benchmark import run_static_model_cv

        X, y = _make_tiny_xy()

        class NotAModel:
            pass

        with pytest.raises(ValueError):
            run_static_model_cv(NotAModel, {}, X, y, n_splits=2)

    def test_bootstrap_delta_test(self):
        from seqcredit_model.run_cv_benchmark import bootstrap_delta_test
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(0)
        n = 50
        y_true = rng.randint(0, 2, size=n)
        y_proba_1 = np.clip(y_true * 0.6 + rng.rand(n) * 0.3, 0, 1)
        y_proba_2 = rng.rand(n)

        result = bootstrap_delta_test(
            y_true, y_proba_1, y_proba_2, roc_auc_score, n_bootstrap=50, seed=1
        )
        expected_keys = {"delta_mean", "delta_std", "ci_lower", "ci_upper", "p_value"}
        assert set(result.keys()) == expected_keys
        assert 0.0 <= result["p_value"] <= 1.0

    def test_run_significance_tests(self):
        from seqcredit_model.run_cv_benchmark import (
            bootstrap_delta_test,
            run_significance_tests,
        )
        import seqcredit_model.run_cv_benchmark as cvb

        # keep bootstrap cheap for this test by shrinking the module default
        rng = np.random.RandomState(0)
        n = 50
        y_true = rng.randint(0, 2, size=n)
        oof_preds = {
            "ModelA": np.clip(y_true * 0.6 + rng.rand(n) * 0.3, 0, 1),
            "ModelB": rng.rand(n),
        }
        old_n_bootstrap = cvb.N_BOOTSTRAP
        cvb.N_BOOTSTRAP = 30
        try:
            df = run_significance_tests(y_true, oof_preds, [("ModelA", "ModelB")])
        finally:
            cvb.N_BOOTSTRAP = old_n_bootstrap

        assert isinstance(df, pd.DataFrame)
        assert set(df["metric"]) == {"AUC-ROC", "AUC-PR"}
        assert len(df) == 2

    def test_is_ephemeral_run_env_toggle(self, monkeypatch):
        from seqcredit_model.run_cv_benchmark import is_ephemeral_run

        monkeypatch.delenv("SEQCREDIT_EPHEMERAL", raising=False)
        assert is_ephemeral_run() is False
        monkeypatch.setenv("SEQCREDIT_EPHEMERAL", "1")
        assert is_ephemeral_run() is True

    def test_cleanup_ephemeral_sequence_artifacts_noop_when_not_ephemeral(
        self, monkeypatch
    ):
        from seqcredit_model.run_cv_benchmark import (
            cleanup_ephemeral_sequence_artifacts,
        )

        monkeypatch.delenv("SEQCREDIT_EPHEMERAL", raising=False)
        # Should return immediately without touching any files or raising.
        cleanup_ephemeral_sequence_artifacts()

    def test_has_sequences_and_get_runtime_data_dir_callable(self):
        from seqcredit_model.run_cv_benchmark import get_runtime_data_dir, has_sequences

        # Just check these don't raise and return sane types; actual truthiness
        # depends on the real repo data/ dir contents, which is out of scope.
        assert isinstance(has_sequences(), bool)
        assert get_runtime_data_dir() is not None


# ---------------------------------------------------------------------------
# run_ablation_study.py
# ---------------------------------------------------------------------------
class TestRunAblationStudy:
    def test_imports_cleanly(self):
        mod = importlib.import_module("seqcredit_model.run_ablation_study")
        assert hasattr(mod, "main")
        assert hasattr(mod, "FEATURE_GROUPS")
        assert isinstance(mod.FEATURE_GROUPS, dict) and len(mod.FEATURE_GROUPS) > 0

    def test_run_condition_tiny(self):
        import seqcredit_model.run_ablation_study as ablation

        X, y = _make_tiny_xy(n=60, n_features=5)
        # run_condition hardcodes RandomForestModel + N_SPLITS (module const);
        # shrink N_SPLITS so the test stays fast.
        old_n_splits = ablation.N_SPLITS
        ablation.N_SPLITS = 2
        try:
            row = ablation.run_condition("TEST_CONDITION", X, y)
        finally:
            ablation.N_SPLITS = old_n_splits

        assert row["condition"] == "TEST_CONDITION"
        assert row["n_features"] == 5
        assert 0.0 <= row["auc_roc"] <= 1.0
        assert "elapsed_s" in row


# ---------------------------------------------------------------------------
# run_hyperparameter_tuning.py
# ---------------------------------------------------------------------------
class TestRunHyperparameterTuning:
    def test_imports_cleanly(self):
        mod = importlib.import_module("seqcredit_model.run_hyperparameter_tuning")
        assert hasattr(mod, "main")
        assert "RandomForest" in mod.SEARCH_SPACES

    def test_search_and_eval_best_params_tiny(self):
        import seqcredit_model.run_hyperparameter_tuning as tuning

        X, y = _make_tiny_xy(n=80, n_features=5)

        # Shrink module-level search budget so RandomizedSearchCV stays fast;
        # search_best_params/eval_best_params read these as globals at call time.
        old_n_iter = tuning.N_ITER
        old_cv_search = tuning.CV_SEARCH
        old_cv_final = tuning.CV_FINAL
        tuning.N_ITER = 2
        tuning.CV_SEARCH = 2
        tuning.CV_FINAL = 2
        try:
            config = tuning.SEARCH_SPACES["RandomForest"]
            best_params, search_auc = tuning.search_best_params(
                "RandomForest", config, X, y
            )
            assert isinstance(best_params, dict) and len(best_params) > 0
            assert 0.0 <= search_auc <= 1.0

            final_summary, final_params = tuning.eval_best_params(
                "RandomForest", config, best_params, X, y
            )
            assert "auc_roc" in final_summary
            assert isinstance(final_params, dict)
        finally:
            tuning.N_ITER = old_n_iter
            tuning.CV_SEARCH = old_cv_search
            tuning.CV_FINAL = old_cv_final


# ---------------------------------------------------------------------------
# run_full_benchmark.py
# ---------------------------------------------------------------------------
class TestRunFullBenchmark:
    def test_imports_cleanly(self):
        mod = importlib.import_module("seqcredit_model.run_full_benchmark")
        assert hasattr(mod, "main")
        assert hasattr(mod, "clear_caches")
        assert hasattr(mod, "set_global_seed")

    def test_clear_caches_removes_only_relative_cache_files(self, tmp_path, monkeypatch):
        """clear_caches() operates on hardcoded relative paths like
        "data/cv_manifest.json" (run_full_benchmark.py:25-31), so we run it
        for real but from a throwaway cwd to avoid touching the repo's data/.
        """
        from seqcredit_model.run_full_benchmark import clear_caches

        fake_data_dir = tmp_path / "data"
        fake_data_dir.mkdir()
        cache_files = [
            "lstm_sequences.npz",
            "cv_results_y_default.csv",
            "cv_results_bad.csv",
            "significance_tests.csv",
            "cv_manifest.json",
        ]
        for fname in cache_files:
            (fake_data_dir / fname).write_text("placeholder")
        other_file = fake_data_dir / "user_features.csv"
        other_file.write_text("keep me")

        monkeypatch.chdir(tmp_path)
        clear_caches()

        for fname in cache_files:
            assert not (fake_data_dir / fname).exists()
        assert other_file.exists()

    def test_set_global_seed_runs(self):
        from seqcredit_model.run_full_benchmark import set_global_seed

        # Just verify it executes without raising (sets numpy/random/tf seeds).
        set_global_seed()


# ---------------------------------------------------------------------------
# compute_bootstrap_ci.py
#
# main() is not easily separable from hardcoded MODELS_DIR / DATA_DIR paths:
# it loads real pickled/keras models and real lstm_test_arrays.npz from the
# repo's data/ and models/ directories with no override hooks, and running it
# for real would mean scoring real trained models (slow, and out of scope per
# task instructions). Kept to import-only.
# ---------------------------------------------------------------------------
class TestComputeBootstrapCi:
    def test_imports_cleanly(self):
        mod = importlib.import_module("seqcredit_model.compute_bootstrap_ci")
        assert hasattr(mod, "main")
        assert mod.N_BOOTSTRAP == 1000
        assert mod.CI_LEVEL == 0.95


# ---------------------------------------------------------------------------
# build_poster.py
# ---------------------------------------------------------------------------
class TestBuildPoster:
    def test_imports_cleanly(self):
        mod = importlib.import_module("seqcredit_model.build_poster")
        assert hasattr(mod, "main")
        assert hasattr(mod, "set_shape_text")

    def test_main_handles_missing_template_gracefully(self, capsys):
        """main() hardcodes an absolute Windows path to a pptx template
        (build_poster.py:99) that does not exist in this checkout/environment
        (it's under a sibling project directory, not this repo). main() checks
        os.path.exists() first and returns early with an error message rather
        than raising, so we can safely call it for real here.
        """
        from seqcredit_model.build_poster import main

        result = main()
        assert result is None
        captured = capsys.readouterr()
        assert "Error: Template not found" in captured.out
