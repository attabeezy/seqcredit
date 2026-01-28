"""
Sequential CRM for DCE - Source Package

This package contains the core modules for the Sequential Deep Learning
for Credit Risk Modeling in Data-Constrained Environments project.

Modules:
    - feature_engineering: Temporal feature extraction from transaction data
    - synthetic_data: Calibrated synthetic data generation
"""

from .feature_engineering import TemporalTransactionFeatureEngineer
from .synthetic_data import CalibratedMoMoDataGenerator

__version__ = "0.1.0"
__all__ = ['TemporalTransactionFeatureEngineer', 'CalibratedMoMoDataGenerator']
