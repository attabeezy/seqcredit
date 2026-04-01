"""
Sequential CRM for DCE - Source Package

This package contains the core modules for the Sequential Deep Learning
for Credit Risk Modeling in Data-Constrained Environments project.

Modules:
    - feature_engineering: Temporal feature extraction from transaction data
    - synthetic_data: Calibrated synthetic data generation
    - models: Credit risk prediction models (LR, XGBoost, LSTM)
"""

from . import config
from .feature_engineering import TemporalTransactionFeatureEngineer
from .synthetic_data import CalibratedMoMoDataGenerator
from .credit_model import (
    CreditRiskDataLoader,
    LogisticRegressionModel,
    XGBoostModel,
    RandomForestModel,
    LightGBMModel,
    LSTMModel,
    HybridLSTMModel,
    ModelEvaluator,
    set_random_seeds,
    bootstrap_evaluate,
)

__version__ = "0.1.0"
__all__ = [
    "config",
    "TemporalTransactionFeatureEngineer",
    "CalibratedMoMoDataGenerator",
    "CreditRiskDataLoader",
    "LogisticRegressionModel",
    "XGBoostModel",
    "RandomForestModel",
    "LightGBMModel",
    "LSTMModel",
    "HybridLSTMModel",
    "ModelEvaluator",
    "set_random_seeds",
    "bootstrap_evaluate",
]
