"""
Data Generation Module

Provides calibrated synthetic mobile money transaction data generation.

Classes:
    - CalibratedMoMoDataGenerator: Generate realistic multi-user transaction
      datasets calibrated to real Ghanaian mobile money patterns.

Usage:
    from src.data_generation.calibrated_synthetic_generator import (
        CalibratedMoMoDataGenerator
    )

    generator = CalibratedMoMoDataGenerator(
        n_users=2000,
        avg_transactions_per_user=15,
        fraud_rate=0.05
    )
    df, users = generator.generate_dataset()
"""

from .calibrated_synthetic_generator import CalibratedMoMoDataGenerator

__all__ = ["CalibratedMoMoDataGenerator"]
