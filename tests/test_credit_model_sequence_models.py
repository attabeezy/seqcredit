"""Tests for the sequence / deep-learning model classes in credit_model.py:

LSTMModel, GRUModel, HybridLSTMModel, HybridGRUModel, TransformerModel,
HybridTransformerModel.

These are real Keras models, so every test here is kept intentionally tiny:
small random arrays, 1 training epoch, and small hidden-unit counts wherever
the constructor exposes them, to keep the whole file fast.
"""

import os

# Must be set before tensorflow gets imported (via seqcredit_model.credit_model
# below) to keep TF logging quiet.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from seqcredit_model.credit_model import (
    GRUModel,
    HybridGRUModel,
    HybridLSTMModel,
    HybridTransformerModel,
    LSTMModel,
    TransformerModel,
)

N_SAMPLES = 30
SEQ_LEN = 10
N_SEQ_FEATURES = 8
N_STATIC_FEATURES = 6

# Tiny hyperparameters shared by the plain recurrent models.
TINY_RNN_KWARGS = dict(
    lstm_units_1=4,
    lstm_units_2=4,
    dense_units=4,
    dropout_rate=0.1,
    learning_rate=0.01,
)

# Tiny hyperparameters for the transformer models (d_model must be divisible
# by num_heads).
TINY_TRANSFORMER_KWARGS = dict(
    d_model=4,
    num_heads=2,
    ff_dim=8,
    num_blocks=1,
    dropout_rate=0.1,
    learning_rate=0.01,
)


# ---------------------------------------------------------------------------
# Fixtures (local to this file — conftest.py is intentionally left untouched)
# ---------------------------------------------------------------------------


@pytest.fixture
def seq_arrays():
    """Small deterministic (X_seq, y) arrays shaped like real model input."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_SAMPLES, SEQ_LEN, N_SEQ_FEATURES)).astype(np.float32)
    y = np.array([0, 1] * (N_SAMPLES // 2), dtype=np.int64)
    perm = rng.permutation(N_SAMPLES)
    X = X[perm]
    y = y[perm]
    return X, y


@pytest.fixture
def static_arrays():
    rng = np.random.default_rng(43)
    return rng.standard_normal((N_SAMPLES, N_STATIC_FEATURES)).astype(np.float32)


@pytest.fixture
def seq_train_val(seq_arrays):
    """Stratified train/val split of the sequence-only data."""
    X, y = seq_arrays
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def hybrid_train_val(seq_arrays, static_arrays):
    """Stratified train/val split of the sequence + static hybrid data."""
    X_seq, y = seq_arrays
    idx = np.arange(N_SAMPLES)
    idx_train, idx_val = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=42
    )
    return (
        X_seq[idx_train],
        X_seq[idx_val],
        static_arrays[idx_train],
        static_arrays[idx_val],
        y[idx_train],
        y[idx_val],
    )


# ---------------------------------------------------------------------------
# Plain sequence models: LSTMModel, GRUModel, TransformerModel
# ---------------------------------------------------------------------------

SEQ_MODEL_CASES = [
    (LSTMModel, TINY_RNN_KWARGS),
    (GRUModel, TINY_RNN_KWARGS),
    (TransformerModel, TINY_TRANSFORMER_KWARGS),
]


@pytest.mark.parametrize(
    "model_cls,kwargs", SEQ_MODEL_CASES, ids=[c.__name__ for c, _ in SEQ_MODEL_CASES]
)
def test_sequence_model_build_fit_predict_save_load(
    model_cls, kwargs, seq_train_val, tmp_path
):
    X_train, X_val, y_train, y_val = seq_train_val

    model = model_cls(**kwargs)
    model.build_model((SEQ_LEN, N_SEQ_FEATURES))

    assert model.model is not None
    assert tuple(model.model.output_shape) == (None, 1)

    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=1,
        batch_size=64,
    )
    assert history is not None
    assert "loss" in history.history

    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert set(np.unique(preds)).issubset({0, 1})
    np.testing.assert_array_equal(preds, (proba >= 0.5).astype(int))

    # Threshold behaviour: extreme thresholds must force all-0 / all-1 output.
    assert np.all(model.predict(X_val, threshold=-1.0) == 1)
    assert np.all(model.predict(X_val, threshold=2.0) == 0)

    save_path = str(tmp_path / f"{model_cls.__name__}_model")
    model.save(save_path)
    assert Path(f"{save_path}.keras").exists()
    assert Path(f"{save_path}.json").exists()

    loaded = model_cls.load(save_path)
    loaded_proba = loaded.predict_proba(X_val)
    np.testing.assert_allclose(proba, loaded_proba, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Hybrid models: HybridLSTMModel, HybridGRUModel, HybridTransformerModel
# ---------------------------------------------------------------------------

HYBRID_MODEL_CASES = [
    (HybridLSTMModel, TINY_RNN_KWARGS),
    (HybridGRUModel, TINY_RNN_KWARGS),
    (
        HybridTransformerModel,
        dict(TINY_TRANSFORMER_KWARGS, dense_units=4),
    ),
]


@pytest.mark.parametrize(
    "model_cls,kwargs",
    HYBRID_MODEL_CASES,
    ids=[c.__name__ for c, _ in HYBRID_MODEL_CASES],
)
def test_hybrid_model_build_fit_predict_save_load(
    model_cls, kwargs, hybrid_train_val, tmp_path
):
    X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = (
        hybrid_train_val
    )

    model = model_cls(**kwargs)
    model.build_model((SEQ_LEN, N_SEQ_FEATURES), N_STATIC_FEATURES)

    assert model.model is not None
    assert tuple(model.model.output_shape) == (None, 1)

    history = model.fit(
        X_seq_train,
        X_static_train,
        y_train,
        X_val_seq=X_seq_val,
        X_val_static=X_static_val,
        y_val=y_val,
        epochs=1,
        batch_size=64,
    )
    assert history is not None
    assert "loss" in history.history

    proba = model.predict_proba(X_seq_val, X_static_val)
    assert proba.shape == (len(X_seq_val),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    preds = model.predict(X_seq_val, X_static_val)
    assert preds.shape == (len(X_seq_val),)
    assert set(np.unique(preds)).issubset({0, 1})
    np.testing.assert_array_equal(preds, (proba >= 0.5).astype(int))

    assert np.all(model.predict(X_seq_val, X_static_val, threshold=-1.0) == 1)
    assert np.all(model.predict(X_seq_val, X_static_val, threshold=2.0) == 0)

    save_path = str(tmp_path / f"{model_cls.__name__}_model")
    model.save(save_path)
    assert Path(f"{save_path}.keras").exists()
    assert Path(f"{save_path}.json").exists()

    loaded = model_cls.load(save_path)
    loaded_proba = loaded.predict_proba(X_seq_val, X_static_val)
    np.testing.assert_allclose(proba, loaded_proba, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# cross_validate: end-to-end check on a single (cheapest) model class only.
# ---------------------------------------------------------------------------


def test_lstm_cross_validate(seq_arrays):
    X, y = seq_arrays
    model = LSTMModel(**TINY_RNN_KWARGS)

    results = model.cross_validate(X, y, n_splits=2, epochs=1, batch_size=64)

    assert set(results.keys()) == {"auc_roc", "auc_pr", "f1", "accuracy"}
    for key, values in results.items():
        assert len(values) == 2, f"expected 2 folds worth of {key}"
        for v in values:
            assert np.isfinite(v)
