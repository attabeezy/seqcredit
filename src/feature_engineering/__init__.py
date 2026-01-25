"""
Feature Engineering Module

Provides temporal feature extraction from mobile money transaction data.

Classes:
    - TemporalTransactionFeatureEngineer: Extract 113 temporal features
      from sequential transaction data.

Usage:
    from src.feature_engineering.real_temporal_feature_engineering import (
        TemporalTransactionFeatureEngineer
    )

    engineer = TemporalTransactionFeatureEngineer()
    features = engineer.extract_all_features(transaction_df)
    user_summary = engineer.create_user_level_summary(transaction_df)
"""

from .real_temporal_feature_engineering import TemporalTransactionFeatureEngineer

__all__ = ["TemporalTransactionFeatureEngineer"]
