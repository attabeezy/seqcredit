"""Tests for seqcredit_model.synthesize.CalibratedMoMoDataGenerator."""
import json

import numpy as np
import pandas as pd
import pytest

from seqcredit_model.synthesize import CalibratedMoMoDataGenerator
from seqcredit_model.config import SYNTHETIC_PARAMS_FILE

EXPECTED_COLUMNS = {
    "TRANSACTION DATE",
    "FROM ACCT",
    "FROM NAME",
    "FROM NO.",
    "TRANS. TYPE",
    "AMOUNT",
    "FEES",
    "E-LEVY",
    "BAL BEFORE",
    "BAL AFTER",
    "TO NO.",
    "TO NAME",
    "TO ACCT",
}


def _make_generator(tmp_path, n_users=20, avg_txns=20, seed=42, **kwargs):
    return CalibratedMoMoDataGenerator(
        n_users=n_users,
        avg_transactions_per_user=avg_txns,
        output_dir=str(tmp_path / "user_transactions"),
        calibration_file=str(SYNTHETIC_PARAMS_FILE),
        seed=seed,
        **kwargs,
    )


class TestDeterminism:
    def test_same_seed_produces_identical_summaries(self, tmp_path):
        gen1 = _make_generator(tmp_path / "run1", n_users=15, avg_txns=15, seed=123)
        summary1 = gen1.generate_dataset()

        gen2 = _make_generator(tmp_path / "run2", n_users=15, avg_txns=15, seed=123)
        summary2 = gen2.generate_dataset()

        pd.testing.assert_frame_equal(
            summary1.reset_index(drop=True), summary2.reset_index(drop=True)
        )

    def test_same_seed_produces_identical_transaction_csvs(self, tmp_path):
        gen1 = _make_generator(tmp_path / "run1", n_users=10, avg_txns=15, seed=7)
        gen1.generate_dataset()

        gen2 = _make_generator(tmp_path / "run2", n_users=10, avg_txns=15, seed=7)
        gen2.generate_dataset()

        csvs1 = sorted((tmp_path / "run1" / "user_transactions").glob("*.csv"))
        csvs2 = sorted((tmp_path / "run2" / "user_transactions").glob("*.csv"))
        assert len(csvs1) == len(csvs2) > 0

        for f1, f2 in zip(csvs1, csvs2):
            df1 = pd.read_csv(f1)
            df2 = pd.read_csv(f2)
            pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_summaries(self, tmp_path):
        gen1 = _make_generator(tmp_path / "run1", n_users=15, avg_txns=15, seed=1)
        summary1 = gen1.generate_dataset()

        gen2 = _make_generator(tmp_path / "run2", n_users=15, avg_txns=15, seed=2)
        summary2 = gen2.generate_dataset()

        # Extremely unlikely that two different seeds produce identical
        # per-user credit archetypes/labels across 15 users.
        assert not summary1["credit_archetype"].equals(summary2["credit_archetype"])


class TestArchetypeAndLabelDistribution:
    def test_archetype_distribution_roughly_matches_weights(self, tmp_path):
        n_users = 150
        gen = _make_generator(tmp_path, n_users=n_users, avg_txns=15, seed=42)
        summary_df = gen.generate_dataset()

        assert len(summary_df) == n_users

        counts = summary_df["credit_archetype"].value_counts(normalize=True)
        for archetype, config in gen.CREDIT_ARCHETYPES.items():
            expected = config["weight"]
            observed = counts.get(archetype, 0.0)
            # Generous absolute tolerance given n=150 and rare archetypes.
            assert abs(observed - expected) < 0.15, (
                f"{archetype}: expected~{expected}, got {observed}"
            )

    def test_credit_risk_labels_are_valid(self, tmp_path):
        gen = _make_generator(tmp_path, n_users=30, avg_txns=15, seed=42)
        summary_df = gen.generate_dataset()

        assert set(summary_df["credit_risk_label"].unique()) <= {-1, 0, 1, 2}

    def test_non_borrowers_have_label_minus_one(self, tmp_path):
        gen = _make_generator(tmp_path, n_users=60, avg_txns=15, seed=42)
        summary_df = gen.generate_dataset()

        non_borrowers = summary_df[summary_df["credit_archetype"] == "non_borrower"]
        assert (non_borrowers["credit_risk_label"] == -1).all()
        assert (non_borrowers["loans_taken"] == 0).all()


class TestUserCsvOutput:
    def test_csv_has_expected_columns_and_sane_values(self, tmp_path):
        gen = _make_generator(tmp_path, n_users=20, avg_txns=20, seed=42)
        gen.generate_dataset()

        csv_files = sorted((tmp_path / "user_transactions").glob("USER_*.csv"))
        assert len(csv_files) > 0

        for filepath in csv_files:
            df = pd.read_csv(filepath)
            assert EXPECTED_COLUMNS.issubset(set(df.columns))
            assert len(df) > 0

            # No negative amounts/fees/e-levy anywhere.
            assert (df["AMOUNT"] >= 0).all()
            assert (df["FEES"] >= 0).all()
            assert (df["E-LEVY"] >= 0).all()

            # BAL AFTER must be explained by BAL BEFORE, AMOUNT, FEES, E-LEVY
            # depending on transaction direction.
            is_inflow = df["TRANS. TYPE"].isin(["CASH_IN", "CREDIT"])
            outflow_expected = df["BAL BEFORE"] - (
                df["AMOUNT"] + df["FEES"].fillna(0) + df["E-LEVY"].fillna(0)
            )
            inflow_expected = df["BAL BEFORE"] + df["AMOUNT"]

            expected_after = np.where(is_inflow, inflow_expected, outflow_expected)
            np.testing.assert_allclose(
                df["BAL AFTER"].to_numpy(), expected_after, atol=0.05
            )

    def test_user_labels_csv_written_with_all_users(self, tmp_path):
        n_users = 25
        gen = _make_generator(tmp_path, n_users=n_users, avg_txns=15, seed=42)
        gen.generate_dataset()

        labels_path = tmp_path / "user_labels.csv"
        assert labels_path.exists()

        labels_df = pd.read_csv(labels_path)
        assert len(labels_df) == n_users
        assert set(labels_df.columns) >= {
            "user_id",
            "credit_archetype",
            "loans_taken",
            "credit_risk_label",
            "final_credit_limit",
            "gen_txn_count",
        }
        assert labels_df["user_id"].is_unique

    def test_no_transactions_written_outside_output_dir(self, tmp_path):
        gen = _make_generator(tmp_path, n_users=10, avg_txns=10, seed=1)
        gen.generate_dataset()

        # output_dir.mkdir happened during __init__; ensure it's the tmp_path
        # location and not the real repo data/ directory.
        assert str(gen.output_dir).startswith(str(tmp_path))


class TestGeneratorUnitBehavior:
    def test_generate_transaction_amount_is_positive(self):
        gen_kwargs = dict(
            n_users=1,
            avg_transactions_per_user=1,
            calibration_file=str(SYNTHETIC_PARAMS_FILE),
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gen = CalibratedMoMoDataGenerator(
                output_dir=tmp, seed=42, **gen_kwargs
            )
            profile = gen.generate_user_profile(0)
            for txn_type in ["TRANSFER", "DEBIT", "PAYMENT", "CASH_OUT", "CASH_IN"]:
                amount = gen.generate_transaction_amount(profile, txn_type)
                assert amount >= 0.5

    def test_calculate_fees_zero_when_user_does_not_accept_fees(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gen = CalibratedMoMoDataGenerator(
                n_users=1,
                avg_transactions_per_user=1,
                output_dir=tmp,
                calibration_file=str(SYNTHETIC_PARAMS_FILE),
                seed=42,
            )
            fees, elevy = gen.calculate_fees(500, "CASH_OUT", user_accepts_fees=False)
            assert fees == 0.0
            assert elevy == 0.0

    def test_save_user_transactions_returns_none_for_empty_list(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gen = CalibratedMoMoDataGenerator(
                n_users=1,
                avg_transactions_per_user=1,
                output_dir=tmp,
                calibration_file=str(SYNTHETIC_PARAMS_FILE),
                seed=42,
            )
            result = gen.save_user_transactions("USER_000000", [])
            assert result is None

    def test_assign_credit_archetype_only_returns_known_archetypes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gen = CalibratedMoMoDataGenerator(
                n_users=1,
                avg_transactions_per_user=1,
                output_dir=tmp,
                calibration_file=str(SYNTHETIC_PARAMS_FILE),
                seed=42,
            )
            for _ in range(100):
                archetype = gen._assign_credit_archetype()
                assert archetype in gen.CREDIT_ARCHETYPES


class TestCalibrationFileHandling:
    def test_missing_calibration_file_raises(self, tmp_path):
        missing_file = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            CalibratedMoMoDataGenerator(
                n_users=1,
                avg_transactions_per_user=1,
                output_dir=str(tmp_path / "out"),
                calibration_file=str(missing_file),
                seed=42,
            )

    def test_loads_real_calibration_file(self, tmp_path):
        gen = _make_generator(tmp_path, n_users=1, avg_txns=1, seed=42)
        with open(SYNTHETIC_PARAMS_FILE) as f:
            expected = json.load(f)
        assert gen.calibration == expected
