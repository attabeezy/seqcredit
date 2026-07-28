"""Tests for the static/tabular model wrappers in seqcredit_model.credit_model:

LogisticRegressionModel, XGBoostModel, RandomForestModel, LightGBMModel.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from seqcredit_model.credit_model import (
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)

MODEL_CLASSES = [
    LogisticRegressionModel,
    XGBoostModel,
    RandomForestModel,
    LightGBMModel,
]


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
class TestStaticModelsCommonBehavior:
    def test_fit_returns_self(self, model_cls, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = model_cls()
        result = model.fit(X_train, y_train)
        assert result is model

    def test_predict_proba_shape_and_range(self, model_cls, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = model_cls().fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_returns_binary_labels(self, model_cls, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = model_cls().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_threshold_behavior(self, model_cls, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = model_cls().fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        # threshold=0 -> everything predicted positive
        preds_low = model.predict(X_test, threshold=0.0)
        assert np.all(preds_low == 1)

        # threshold=1.0001 -> nothing predicted positive (proba is always < that)
        preds_high = model.predict(X_test, threshold=1.0001)
        assert np.all(preds_high == 0)

        # default threshold matches manual computation
        preds_default = model.predict(X_test)
        assert np.array_equal(preds_default, (proba >= 0.5).astype(int))

    def test_cross_validate_returns_expected_keys(self, model_cls, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = model_cls()
        results = model.cross_validate(X_train, y_train, n_splits=3)
        assert set(results.keys()) == {"auc_roc", "auc_pr", "f1", "accuracy"}
        for key, values in results.items():
            assert len(values) == 3
            for v in values:
                assert 0.0 <= v <= 1.0

    def test_save_load_round_trip_preserves_predictions(
        self, model_cls, classification_data, tmp_path
    ):
        X_train, X_test, y_train, y_test = classification_data
        model = model_cls().fit(X_train, y_train)
        proba_before = model.predict_proba(X_test)

        save_path = tmp_path / f"{model_cls.__name__}.joblib"
        model.save(str(save_path))
        assert save_path.exists()

        loaded = model_cls.load(str(save_path))
        proba_after = loaded.predict_proba(X_test)

        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-6, atol=1e-8)


class TestLogisticRegressionSpecific:
    def test_get_coefficients(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticRegressionModel().fit(X_train, y_train)
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
        coefs = model.get_coefficients(feature_names)
        assert set(coefs.columns) == {"feature", "coefficient", "abs_coefficient"}
        assert len(coefs) == X_train.shape[1]
        # sorted descending by absolute coefficient
        assert list(coefs["abs_coefficient"]) == sorted(
            coefs["abs_coefficient"], reverse=True
        )

    def test_class_weight_default_is_balanced(self):
        model = LogisticRegressionModel()
        assert model.model.class_weight == "balanced"


class TestXGBoostSpecific:
    def test_get_feature_importance(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = XGBoostModel().fit(X_train, y_train)
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
        importance = model.get_feature_importance(feature_names)
        assert set(importance.columns) == {"feature", "importance"}
        assert len(importance) == X_train.shape[1]

    def test_scale_pos_weight_applied(self):
        model = XGBoostModel(scale_pos_weight=3.5)
        assert model.params["scale_pos_weight"] == 3.5

    def test_scale_pos_weight_defaults_to_one(self):
        model = XGBoostModel()
        assert model.params["scale_pos_weight"] == 1.0

    def test_fit_with_eval_set(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = XGBoostModel(n_estimators=20)
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)


class TestRandomForestSpecific:
    def test_get_feature_importance(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = RandomForestModel(n_estimators=20).fit(X_train, y_train)
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
        importance = model.get_feature_importance(feature_names)
        assert set(importance.columns) == {"feature", "importance"}
        assert len(importance) == X_train.shape[1]
        # importances sorted descending
        assert list(importance["importance"]) == sorted(
            importance["importance"], reverse=True
        )

    def test_class_weight_default_is_balanced(self):
        model = RandomForestModel()
        assert model.class_weight == "balanced"


class TestLightGBMSpecific:
    def test_get_feature_importance(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LightGBMModel(n_estimators=20).fit(X_train, y_train)
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
        importance = model.get_feature_importance(feature_names)
        assert set(importance.columns) == {"feature", "importance"}
        assert len(importance) == X_train.shape[1]

    def test_fit_with_eval_set_and_early_stopping(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = LightGBMModel(n_estimators=50)
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test, early_stopping_rounds=5)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
