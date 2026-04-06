"""
Sequential CRM for DCE - Source Package

This package contains the core modules for the Sequential Deep Learning
for Credit Risk Modeling in Data-Constrained Environments project.

Modules:
    - pipeline: Temporal feature extraction from transaction data
    - synthesize: Calibrated synthetic data generation
    - models: Credit risk prediction models (LR, XGBoost, LSTM)
"""

from . import config
from .pipeline import TemporalTransactionFeatureEngineer
from .synthesize import CalibratedMoMoDataGenerator
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
