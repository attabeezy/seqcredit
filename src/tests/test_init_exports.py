"""Tests for the public API surface re-exported by seqcredit/__init__.py."""
import inspect

import seqcredit


class TestAllListIntegrity:
    def test_all_defined(self):
        assert hasattr(seqcredit, "__all__")
        assert isinstance(seqcredit.__all__, list)
        assert len(seqcredit.__all__) > 0

    def test_no_duplicate_names_in_all(self):
        names = seqcredit.__all__
        assert len(names) == len(set(names))

    def test_every_name_in_all_is_importable_attribute(self):
        for name in seqcredit.__all__:
            assert hasattr(seqcredit, name), f"{name!r} listed in __all__ but not an attribute"

    def test_every_name_in_all_importable_via_from_import(self):
        # Exercises the actual `from seqcredit import X` path, not just getattr.
        import importlib

        mod = importlib.import_module("seqcredit")
        for name in seqcredit.__all__:
            value = getattr(mod, name)
            assert value is not None


class TestExpectedExportSet:
    """Cross-check against the export list documented in CLAUDE.md."""

    EXPECTED_NAMES = {
        "config",
        "TemporalTransactionFeatureEngineer",
        "CalibratedMoMoDataGenerator",
        "CreditRiskDataLoader",
        "LogisticRegressionModel",
        "XGBoostModel",
        "RandomForestModel",
        "LightGBMModel",
        "LSTMModel",
        "GRUModel",
        "HybridLSTMModel",
        "HybridGRUModel",
        "ModelEvaluator",
        "set_random_seeds",
        "bootstrap_evaluate",
    }

    def test_all_matches_expected_documented_set(self):
        assert set(seqcredit.__all__) == self.EXPECTED_NAMES


class TestExportTypes:
    """Verify each export is the expected kind of object (class vs function/module)."""

    def test_config_is_module(self):
        import types

        assert isinstance(seqcredit.config, types.ModuleType)

    def test_feature_engineer_is_class(self):
        assert inspect.isclass(seqcredit.TemporalTransactionFeatureEngineer)

    def test_data_generator_is_class(self):
        assert inspect.isclass(seqcredit.CalibratedMoMoDataGenerator)

    def test_data_loader_is_class(self):
        assert inspect.isclass(seqcredit.CreditRiskDataLoader)

    def test_model_classes_are_classes(self):
        model_classes = [
            "LogisticRegressionModel",
            "XGBoostModel",
            "RandomForestModel",
            "LightGBMModel",
            "LSTMModel",
            "GRUModel",
            "HybridLSTMModel",
            "HybridGRUModel",
        ]
        for name in model_classes:
            obj = getattr(seqcredit, name)
            assert inspect.isclass(obj), f"{name} should be a class"

    def test_model_evaluator_is_class(self):
        assert inspect.isclass(seqcredit.ModelEvaluator)

    def test_set_random_seeds_is_callable_function(self):
        assert inspect.isfunction(seqcredit.set_random_seeds)

    def test_bootstrap_evaluate_is_callable_function(self):
        assert inspect.isfunction(seqcredit.bootstrap_evaluate)


class TestVersion:
    def test_version_string_present(self):
        assert hasattr(seqcredit, "__version__")
        assert isinstance(seqcredit.__version__, str)
        assert seqcredit.__version__  # non-empty


class TestModelsNotInAllButPresentInCreditModel:
    """CLAUDE.md documents TransformerModel/HybridTransformerModel as present in
    credit_model.py but NOT part of the package's public __init__ exports; confirm
    that documented asymmetry actually holds."""

    def test_transformer_models_not_exported_from_package_root(self):
        assert "TransformerModel" not in seqcredit.__all__
        assert "HybridTransformerModel" not in seqcredit.__all__
        assert not hasattr(seqcredit, "TransformerModel")
        assert not hasattr(seqcredit, "HybridTransformerModel")

    def test_transformer_models_importable_directly_from_credit_model(self):
        from seqcredit.credit_model import (
            TransformerModel,
            HybridTransformerModel,
        )

        assert inspect.isclass(TransformerModel)
        assert inspect.isclass(HybridTransformerModel)
