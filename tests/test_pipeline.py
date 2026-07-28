"""Tests for seqcredit_model.pipeline (TemporalTransactionFeatureEngineer,
build_user_feature_dataset)."""
import numpy as np
import pandas as pd
import pytest

from seqcredit_model.pipeline import (
    TemporalTransactionFeatureEngineer,
    build_user_feature_dataset,
)
from seqcredit_model.synthesize import CalibratedMoMoDataGenerator
from seqcredit_model.config import SYNTHETIC_PARAMS_FILE


def _make_raw_transactions(n=25, start="2024-01-01", freq_hours=6):
    """Hand-crafted transaction DataFrame with the exact columns pipeline.py expects."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start=start, periods=n, freq=f"{freq_hours}h")

    types = rng.choice(
        ["TRANSFER", "DEBIT", "PAYMENT", "PAYMENT_SEND", "CASH_OUT", "CASH_IN"],
        size=n,
        p=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1],
    )
    amounts = np.round(rng.uniform(5, 30, size=n), 2)
    fees = np.round(rng.choice([0, 0.5, 1.0], size=n, p=[0.7, 0.2, 0.1]), 2)
    elevy = np.round(rng.choice([0, 1.5], size=n, p=[0.85, 0.15]), 2)

    balances_before = np.zeros(n)
    balances_after = np.zeros(n)
    bal = 5000.0
    for i in range(n):
        balances_before[i] = bal
        if types[i] == "CASH_IN":
            bal += amounts[i]
        else:
            bal -= amounts[i] + fees[i] + elevy[i]
        balances_after[i] = bal

    to_names = rng.choice(["Alice", "Bob", "Carol", "MTN", "hubtel.sp"], size=n)

    df = pd.DataFrame(
        {
            "TRANSACTION DATE": dates,
            "FROM ACCT": "12345678",
            "FROM NAME": "User TEST01",
            "FROM NO.": "233200000000",
            "TRANS. TYPE": types,
            "AMOUNT": amounts,
            "FEES": fees,
            "E-LEVY": elevy,
            "BAL BEFORE": balances_before,
            "BAL AFTER": balances_after,
            "TO NO.": "0",
            "TO NAME": to_names,
            "TO ACCT": "ACCT_0001",
        }
    )
    return df


class TestExtractAllFeatures:
    def test_returns_same_row_count_as_input(self):
        df = _make_raw_transactions(n=30)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)
        assert len(result) == len(df)

    def test_cyclical_encodings_present_and_bounded(self):
        df = _make_raw_transactions(n=30)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)

        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert col in result.columns
            assert (result[col] >= -1.0001).all()
            assert (result[col] <= 1.0001).all()

    def test_categorical_one_hot_columns_present(self):
        df = _make_raw_transactions(n=30)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)

        one_hots = [
            "is_transfer",
            "is_debit",
            "is_payment",
            "is_payment_send",
            "is_cash_out",
            "is_cash_in",
            "is_adjustment",
        ]
        for col in one_hots:
            assert col in result.columns
            assert set(result[col].unique()) <= {0, 1}

        # exactly one type flag active per row (types used are all covered)
        row_sum = result[one_hots].sum(axis=1)
        assert (row_sum == 1).all()

    def test_rolling_window_columns_present(self):
        df = _make_raw_transactions(n=30)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df, windows=[3, 7, 14, 30])

        for n in [3, 5, 10]:
            assert f"last_{n}_avg_amount" in result.columns
            assert f"last_{n}_max_amount" in result.columns
            assert f"last_{n}_min_amount" in result.columns

        for window_days in [3, 7, 14, 30]:
            key = f"{window_days}d"
            assert f"rolling_{key}_count" in result.columns
            assert f"rolling_{key}_sum" in result.columns
            assert f"rolling_{key}_mean" in result.columns

    def test_balance_dynamics_columns(self):
        df = _make_raw_transactions(n=20)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)

        np.testing.assert_allclose(
            result["balance_change"].to_numpy(),
            (result["BAL AFTER"] - result["BAL BEFORE"]).to_numpy(),
        )
        assert "log_balance_before" in result.columns
        assert "log_balance_after" in result.columns

    def test_risk_indicator_columns_present_and_numeric(self):
        df = _make_raw_transactions(n=20)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)

        for col in [
            "unusual_hour",
            "rapid_transaction",
            "unusual_amount_high",
            "unusual_amount_low",
            "rapid_balance_drop",
            "consecutive_withdrawals",
            "high_frequency_period",
            "risk_score",
        ]:
            assert col in result.columns
            assert result[col].notna().all()

    def test_sorts_by_transaction_date(self):
        df = _make_raw_transactions(n=15)
        shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)

        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(shuffled)

        dates = pd.to_datetime(result["TRANSACTION DATE"])
        assert dates.is_monotonic_increasing

    def test_txn_number_sequence(self):
        df = _make_raw_transactions(n=12)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)

        assert list(result["txn_number"]) == list(range(1, len(df) + 1))
        assert list(result["reverse_txn_number"]) == [
            len(df) - n for n in result["txn_number"]
        ]

    def test_first_row_time_since_last_txn_is_zero(self):
        df = _make_raw_transactions(n=10)
        engineer = TemporalTransactionFeatureEngineer()
        result = engineer.extract_all_features(df)
        assert result["time_since_last_txn_hours"].iloc[0] == 0


class TestCreateUserLevelSummary:
    def test_summary_has_expected_fields(self):
        df = _make_raw_transactions(n=20)
        engineer = TemporalTransactionFeatureEngineer()
        features = engineer.extract_all_features(df)
        summary = engineer.create_user_level_summary(features)

        expected_fields = {
            "obs_txn_count",
            "total_volume",
            "avg_transaction_amount",
            "pct_transfers",
            "pct_debits",
            "pct_cashouts",
            "pct_payments",
            "avg_balance",
            "min_balance",
            "max_balance",
            "unique_recipients",
            "unique_txn_types",
            "account_age_days",
            "transactions_per_day",
        }
        assert expected_fields.issubset(set(summary.index))
        assert summary["obs_txn_count"] == len(df)

    def test_summary_has_no_nans_for_well_formed_input(self):
        df = _make_raw_transactions(n=25)
        engineer = TemporalTransactionFeatureEngineer()
        features = engineer.extract_all_features(df)
        summary = engineer.create_user_level_summary(features)

        assert not summary.isna().any(), summary[summary.isna()]

    def test_pct_fields_are_fractions(self):
        df = _make_raw_transactions(n=25)
        engineer = TemporalTransactionFeatureEngineer()
        features = engineer.extract_all_features(df)
        summary = engineer.create_user_level_summary(features)

        for col in ["pct_transfers", "pct_debits", "pct_cashouts", "pct_payments"]:
            assert 0.0 <= summary[col] <= 1.0


class TestBuildUserFeatureDataset:
    @pytest.fixture
    def small_transactions_dir(self, tmp_path):
        """Generate a small real synthetic dataset (not the shared fixture,
        to keep this test file self-contained per agent-scope rules)."""
        transactions_dir = tmp_path / "user_transactions"
        generator = CalibratedMoMoDataGenerator(
            n_users=25,
            avg_transactions_per_user=20,
            output_dir=str(transactions_dir),
            calibration_file=str(SYNTHETIC_PARAMS_FILE),
            seed=42,
        )
        generator.generate_dataset()
        return transactions_dir

    def test_one_row_per_user(self, small_transactions_dir, tmp_path):
        output_path = tmp_path / "user_features.csv"
        result = build_user_feature_dataset(
            transactions_dir=str(small_transactions_dir),
            output_path=str(output_path),
        )

        csv_files = list(small_transactions_dir.glob("USER_*.csv"))
        assert len(result) <= len(csv_files)
        assert len(result) > 0
        assert result.index.name == "user_id"
        assert result.index.is_unique

    def test_no_nans_in_critical_columns(self, small_transactions_dir, tmp_path):
        output_path = tmp_path / "user_features.csv"
        result = build_user_feature_dataset(
            transactions_dir=str(small_transactions_dir),
            output_path=str(output_path),
        )

        critical_columns = [
            "obs_txn_count",
            "total_volume",
            "avg_transaction_amount",
            "pct_transfers",
            "pct_debits",
            "avg_balance",
            "unique_recipients",
            "unique_txn_types",
        ]
        for col in critical_columns:
            assert col in result.columns
            assert result[col].notna().all(), f"NaNs found in {col}"

    def test_output_written_to_disk(self, small_transactions_dir, tmp_path):
        output_path = tmp_path / "user_features.csv"
        build_user_feature_dataset(
            transactions_dir=str(small_transactions_dir),
            output_path=str(output_path),
        )
        assert output_path.exists()
        reloaded = pd.read_csv(output_path, index_col=0)
        assert len(reloaded) > 0

    def test_empty_directory_returns_empty_dataframe(self, tmp_path):
        empty_dir = tmp_path / "empty_transactions"
        empty_dir.mkdir()
        output_path = tmp_path / "user_features.csv"

        result = build_user_feature_dataset(
            transactions_dir=str(empty_dir), output_path=str(output_path)
        )
        assert len(result) == 0

    def test_skips_files_with_fewer_than_two_rows(self, tmp_path):
        transactions_dir = tmp_path / "user_transactions"
        transactions_dir.mkdir()
        output_path = tmp_path / "user_features.csv"

        # A one-row file should be skipped (pipeline requires len(df) >= 2).
        one_row_df = _make_raw_transactions(n=1)
        one_row_df.to_csv(transactions_dir / "USER_000001.csv", index=False)

        # A well-formed file should be included.
        good_df = _make_raw_transactions(n=10)
        good_df.to_csv(transactions_dir / "USER_000002.csv", index=False)

        result = build_user_feature_dataset(
            transactions_dir=str(transactions_dir), output_path=str(output_path)
        )
        assert len(result) == 1
        assert "USER_000002" in result.index
        assert "USER_000001" not in result.index

    def test_only_reads_user_prefixed_csv_files(self, tmp_path):
        transactions_dir = tmp_path / "user_transactions"
        transactions_dir.mkdir()
        output_path = tmp_path / "user_features.csv"

        good_df = _make_raw_transactions(n=10)
        good_df.to_csv(transactions_dir / "USER_000001.csv", index=False)
        # Non-matching filename should be ignored by the USER_*.csv glob.
        good_df.to_csv(transactions_dir / "summary_stats.csv", index=False)

        result = build_user_feature_dataset(
            transactions_dir=str(transactions_dir), output_path=str(output_path)
        )
        assert len(result) == 1
        assert "USER_000001" in result.index
