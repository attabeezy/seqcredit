"""Tests for the public API surface re-exported by seqcredit_model/__init__.py."""
import inspect

import seqcredit_model


class TestAllListIntegrity:
    def test_all_defined(self):
        assert hasattr(seqcredit_model, "__all__")
        assert isinstance(seqcredit_model.__all__, list)
        assert len(seqcredit_model.__all__) > 0

    def test_no_duplicate_names_in_all(self):
        names = seqcredit_model.__all__
        assert len(names) == len(set(names))

    def test_every_name_in_all_is_importable_attribute(self):
        for name in seqcredit_model.__all__:
            assert hasattr(seqcredit_model, name), f"{name!r} listed in __all__ but not an attribute"

    def test_every_name_in_all_importable_via_from_import(self):
        # Exercises the actual `from seqcredit_model import X` path, not just getattr.
        import importlib

        mod = importlib.import_module("seqcredit_model")
        for name in seqcredit_model.__all__:
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
        assert set(seqcredit_model.__all__) == self.EXPECTED_NAMES


class TestExportTypes:
    """Verify each export is the expected kind of object (class vs function/module)."""

    def test_config_is_module(self):
        import types

        assert isinstance(seqcredit_model.config, types.ModuleType)

    def test_feature_engineer_is_class(self):
        assert inspect.isclass(seqcredit_model.TemporalTransactionFeatureEngineer)

    def test_data_generator_is_class(self):
        assert inspect.isclass(seqcredit_model.CalibratedMoMoDataGenerator)

    def test_data_loader_is_class(self):
        assert inspect.isclass(seqcredit_model.CreditRiskDataLoader)

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
            obj = getattr(seqcredit_model, name)
            assert inspect.isclass(obj), f"{name} should be a class"

    def test_model_evaluator_is_class(self):
        assert inspect.isclass(seqcredit_model.ModelEvaluator)

    def test_set_random_seeds_is_callable_function(self):
        assert inspect.isfunction(seqcredit_model.set_random_seeds)

    def test_bootstrap_evaluate_is_callable_function(self):
        assert inspect.isfunction(seqcredit_model.bootstrap_evaluate)


class TestVersion:
    def test_version_string_present(self):
        assert hasattr(seqcredit_model, "__version__")
        assert isinstance(seqcredit_model.__version__, str)
        assert seqcredit_model.__version__  # non-empty


class TestModelsNotInAllButPresentInCreditModel:
    """CLAUDE.md documents TransformerModel/HybridTransformerModel as present in
    credit_model.py but NOT part of the package's public __init__ exports; confirm
    that documented asymmetry actually holds."""

    def test_transformer_models_not_exported_from_package_root(self):
        assert "TransformerModel" not in seqcredit_model.__all__
        assert "HybridTransformerModel" not in seqcredit_model.__all__
        assert not hasattr(seqcredit_model, "TransformerModel")
        assert not hasattr(seqcredit_model, "HybridTransformerModel")

    def test_transformer_models_importable_directly_from_credit_model(self):
        from seqcredit_model.credit_model import (
            TransformerModel,
            HybridTransformerModel,
        )

        assert inspect.isclass(TransformerModel)
        assert inspect.isclass(HybridTransformerModel)
