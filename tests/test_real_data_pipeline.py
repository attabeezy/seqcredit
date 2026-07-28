"""Tests for the Spark-based real-data pipeline (real_data_pipeline.py).

There is no access to the real Databricks table
(``melodatabricks616.default.yara_dump_table``), so these tests build small
synthetic Spark DataFrames that match the raw-table schema inferred from the
pipeline code itself: TRANSACTION_TIMESTAMP (Oracle-style
'DD-MON-YY HH.MI.SS.FFFFFFFFF' string), TRANSACTION_TYPE, TRANSACTION_AMOUNT,
DEBIT_PARTY_ID/TYPE/ACCOUNT_TYPE, CREDIT_PARTY_ID/TYPE/ACCOUNT_TYPE.

The most safety-critical logic in this module is the leakage-prevention
scheme: labels are derived only from events strictly AFTER a borrower's index
loan (most recent disbursement), and features only from events strictly
BEFORE it. Several tests below exist specifically to pin that behaviour down.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqcredit_model.real_data_pipeline import (
    CUSTOMER_ACCOUNT_TYPE,
    LENDER_ID,
    LOAN_DISBURSEMENT_TYPE,
    LOAN_INTEREST_TYPE,
    LOAN_PENALTY_TYPE,
    LOAN_PRINCIPAL_TYPE,
    RAW_SEQUENCE_VERSION,
    SEQ_FEATURE_NAMES,
    TXTYPE_CATS,
    _categorize_txtype,
    _derive_loan_cutoffs,
    _hash_user_ids,
    _parse_timestamp,
    _resolve_storage_path,
    _write_npz_with_metadata,
    build_pipeline,
    build_sequences_spark,
    derive_labels,
    engineer_features,
)

RAW_COLUMNS = [
    "TRANSACTION_TIMESTAMP",
    "TRANSACTION_TYPE",
    "TRANSACTION_AMOUNT",
    "DEBIT_PARTY_ID",
    "DEBIT_PARTY_TYPE",
    "DEBIT_ACCOUNT_TYPE",
    "CREDIT_PARTY_ID",
    "CREDIT_PARTY_TYPE",
    "CREDIT_ACCOUNT_TYPE",
]

MERCHANT_ACCOUNT_TYPE = "Merchant Wallet"
AGENT_ACCOUNT_TYPE = "Agent Wallet"
LENDER_ACCOUNT_TYPE = "Lender Settlement Account"


def _ora_ts(dt: datetime, one_digit_second: bool = False) -> str:
    """Format a datetime as the Oracle-style string the pipeline parses."""
    date_part = dt.strftime("%d-%b-%y").upper()
    if one_digit_second:
        time_part = f"{dt.hour:02d}.{dt.minute:02d}.{dt.second}"
    else:
        time_part = dt.strftime("%H.%M.%S")
    return f"{date_part} {time_part}.000000000"


def loan_disbursement(user_id, dt, amount=500.0):
    return (
        _ora_ts(dt), LOAN_DISBURSEMENT_TYPE, amount,
        LENDER_ID, "Lender", LENDER_ACCOUNT_TYPE,
        user_id, "Customer", CUSTOMER_ACCOUNT_TYPE,
    )


def repayment(user_id, dt, amount=50.0, kind="principal"):
    txtype = LOAN_PRINCIPAL_TYPE if kind == "principal" else LOAN_INTEREST_TYPE
    return (
        _ora_ts(dt), txtype, amount,
        user_id, "Customer", CUSTOMER_ACCOUNT_TYPE,
        LENDER_ID, "Lender", LENDER_ACCOUNT_TYPE,
    )


def penalty(user_id, dt, amount=10.0):
    return (
        _ora_ts(dt), LOAN_PENALTY_TYPE, amount,
        user_id, "Customer", CUSTOMER_ACCOUNT_TYPE,
        LENDER_ID, "Lender", LENDER_ACCOUNT_TYPE,
    )


def outgoing_spend(user_id, dt, amount, txn_type="Pay Bill Payment", counterparty="merchant_1"):
    return (
        _ora_ts(dt), txn_type, amount,
        user_id, "Customer", CUSTOMER_ACCOUNT_TYPE,
        counterparty, "Merchant", MERCHANT_ACCOUNT_TYPE,
    )


def incoming_deposit(user_id, dt, amount, txn_type="Cash Deposit at Agent", counterparty="agent_1"):
    return (
        _ora_ts(dt), txn_type, amount,
        counterparty, "Agent", AGENT_ACCOUNT_TYPE,
        user_id, "Customer", CUSTOMER_ACCOUNT_TYPE,
    )


D = lambda *a: datetime(*a)  # noqa: E731 - small date helper


# ── Section 1: pure-Python helpers, no Spark needed ────────────────────────

class TestResolveStoragePath:
    def test_local_path_passthrough(self):
        assert _resolve_storage_path("data/out.csv") == Path("data/out.csv")

    def test_local_path_from_path_object(self, tmp_path):
        p = tmp_path / "foo.csv"
        assert _resolve_storage_path(p) == Path(str(p))

    def test_dbfs_path_rewritten_to_local_mount(self):
        result = _resolve_storage_path("dbfs:/mnt/data/out.csv")
        assert result == Path("/dbfs/mnt/data/out.csv")

    def test_dbfs_path_with_extra_leading_slash(self):
        result = _resolve_storage_path("dbfs:///mnt/data/out.csv")
        assert result == Path("/dbfs/mnt/data/out.csv")


class TestHashUserIds:
    def test_deterministic_for_same_input(self):
        ids = ["user_a", "user_b", "user_c"]
        assert _hash_user_ids(ids) == _hash_user_ids(list(ids))

    def test_order_sensitive(self):
        assert _hash_user_ids(["a", "b"]) != _hash_user_ids(["b", "a"])

    def test_content_sensitive(self):
        assert _hash_user_ids(["a", "b"]) != _hash_user_ids(["a", "c"])

    def test_returns_hex_digest(self):
        digest = _hash_user_ids(["x"])
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)  # raises if not valid hex


class TestWriteNpzWithMetadata:
    def test_round_trips_arrays_and_metadata(self, tmp_path):
        out_path = tmp_path / "payload.npz"
        metadata = {"a": 1, "nested": {"b": [1, 2, 3]}, "s": "hello"}
        arr1 = np.arange(12).reshape(3, 4).astype(np.float32)
        arr2 = np.array(["u1", "u2", "u3"], dtype=object)

        _write_npz_with_metadata(out_path, metadata, sequences=arr1, user_ids=arr2)

        assert out_path.exists()
        with np.load(out_path, allow_pickle=True) as data:
            np.testing.assert_array_equal(data["sequences"], arr1)
            np.testing.assert_array_equal(data["user_ids"], arr2)
            loaded_metadata = json.loads(str(data["metadata_json"]))
            assert loaded_metadata == metadata


# ── Section 2: _parse_timestamp ─────────────────────────────────────────────

class TestParseTimestamp:
    def test_two_digit_second_format(self, spark_session):
        df = spark_session.createDataFrame(
            [(_ora_ts(D(2023, 1, 1, 10, 15, 30)),)], ["TRANSACTION_TIMESTAMP"]
        )
        result = _parse_timestamp(df).collect()[0]["ts"]
        assert result == datetime(2023, 1, 1, 10, 15, 30)

    def test_one_digit_second_format(self, spark_session):
        df = spark_session.createDataFrame(
            [(_ora_ts(D(2023, 3, 15, 9, 5, 9), one_digit_second=True),)],
            ["TRANSACTION_TIMESTAMP"],
        )
        result = _parse_timestamp(df).collect()[0]["ts"]
        assert result == datetime(2023, 3, 15, 9, 5, 9)

    def test_mixed_batch_both_formats_parse(self, spark_session):
        rows = [
            (_ora_ts(D(2023, 6, 1, 23, 59, 58)),),
            (_ora_ts(D(2023, 6, 2, 0, 0, 5), one_digit_second=True),),
        ]
        df = spark_session.createDataFrame(rows, ["TRANSACTION_TIMESTAMP"])
        parsed = [r["ts"] for r in _parse_timestamp(df).orderBy("TRANSACTION_TIMESTAMP").collect()]
        assert None not in parsed
        assert datetime(2023, 6, 1, 23, 59, 58) in parsed
        assert datetime(2023, 6, 2, 0, 0, 5) in parsed


# ── Section 3: _categorize_txtype ───────────────────────────────────────────

class TestCategorizeTxtype:
    REPRESENTATIVE = {
        LOAN_DISBURSEMENT_TYPE: "loan_disbursement",
        LOAN_PRINCIPAL_TYPE: "loan_repayment_principal",
        LOAN_INTEREST_TYPE: "loan_repayment_interest",
        LOAN_PENALTY_TYPE: "loan_penalty",
        "Cash Withdrawal at Agent": "cash_out",
        "ATM Withdrawal": "cash_out",
        "Cash Deposit at Agent": "cash_in",
        "Airtime Purchase": "airtime_data",
        "Data Purchase Bundle": "airtime_data",
        "EVD Top Up": "airtime_data",
        "GHIPSS Instant Pay": "transfer",
        "P2P Transfer": "transfer",
        "Pay Bill Payment": "payment",
        "Online Payment": "payment",
        "Direct Debit Collection": "payment",
        "Sports Betting Deposit": "betting",
        "FSI Settlement": "fsi",
        "Some Unrecognised Type XYZ": "other",
    }

    def test_all_representative_types_map_correctly(self, spark_session):
        rows = [(txtype, 1.0) for txtype in self.REPRESENTATIVE]
        df = spark_session.createDataFrame(rows, ["TRANSACTION_TYPE", "TRANSACTION_AMOUNT"])
        result = {
            r["TRANSACTION_TYPE"]: r["txtype_cat"]
            for r in _categorize_txtype(df).collect()
        }
        for txtype, expected_cat in self.REPRESENTATIVE.items():
            assert result[txtype] == expected_cat, f"{txtype!r} -> {result[txtype]!r}, expected {expected_cat!r}"

    def test_all_twelve_categories_covered_by_fixture(self):
        assert set(self.REPRESENTATIVE.values()) == set(TXTYPE_CATS)


# ── Shared synthetic multi-user dataset used by cutoff/label/feature tests ──
#
# Timeline:
#   alice: loan on 2023-01-10, clean repayment after -> label 0 (good)
#   bob:   loan on 2023-01-12, penalty AND repayment after -> label 1 (risky)
#   carol: loan on 2023-01-14, never repays after -> label 2 (default)
#   dave:  loan on 2023-05-20, too close to the dataset's global max
#          timestamp (2023-06-01) to satisfy the 60-day followup window ->
#          excluded entirely from cutoffs/labels (right-censoring guard)
#
# An anchor transaction on 2023-06-01 establishes the dataset's global max
# timestamp independent of any borrower's own activity.

def _build_core_rows():
    rows = []
    # Pre-loan activity (used by engineer_features)
    rows.append(outgoing_spend("alice", D(2023, 1, 1), 20.0, "Pay Bill Payment"))
    rows.append(incoming_deposit("alice", D(2023, 1, 3), 100.0))
    rows.append(outgoing_spend("alice", D(2023, 1, 5), 15.0, "Airtime Purchase"))

    rows.append(outgoing_spend("bob", D(2023, 1, 2), 30.0, "P2P Transfer"))
    rows.append(incoming_deposit("bob", D(2023, 1, 4), 80.0))

    rows.append(outgoing_spend("carol", D(2023, 1, 1), 25.0, "Cash Withdrawal at Agent"))
    rows.append(incoming_deposit("carol", D(2023, 1, 3), 60.0))

    rows.append(outgoing_spend("dave", D(2023, 1, 1), 10.0))

    # Index loan disbursements
    rows.append(loan_disbursement("alice", D(2023, 1, 10)))
    rows.append(loan_disbursement("bob", D(2023, 1, 12)))
    rows.append(loan_disbursement("carol", D(2023, 1, 14)))
    rows.append(loan_disbursement("dave", D(2023, 5, 20)))

    # Post-loan events used for label derivation
    rows.append(repayment("alice", D(2023, 1, 15), kind="principal"))

    rows.append(penalty("bob", D(2023, 1, 20)))
    rows.append(repayment("bob", D(2023, 1, 25), kind="principal"))

    # carol: no post-loan repayment or penalty at all -> default

    # Anchor transaction fixing the dataset's global max ts far enough past
    # alice/bob/carol's loans but NOT far enough past dave's loan.
    rows.append(outgoing_spend("zzz_anchor", D(2023, 6, 1), 5.0, "Cash Withdrawal at Agent"))

    return rows


@pytest.fixture
def core_df(spark_session):
    return spark_session.createDataFrame(_build_core_rows(), RAW_COLUMNS)


@pytest.fixture
def core_df_ts(core_df):
    df = _parse_timestamp(core_df)
    df = _categorize_txtype(df)
    return df


# ── Section 4: _derive_loan_cutoffs + derive_labels (leakage-safety core) ───

class TestDeriveLoanCutoffs:
    def test_includes_borrowers_with_sufficient_followup(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60).toPandas()
        included = set(cutoffs["user_id"])
        assert {"alice", "bob", "carol"}.issubset(included)

    def test_excludes_right_censored_borrower(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60).toPandas()
        assert "dave" not in set(cutoffs["user_id"])

    def test_last_loan_ts_matches_disbursement_date(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60).toPandas()
        row = cutoffs.set_index("user_id").loc["alice"]
        assert pd.Timestamp(row["last_loan_ts"]) == pd.Timestamp(2023, 1, 10)

    def test_uses_most_recent_disbursement_when_multiple_loans(self, spark_session):
        rows = [
            loan_disbursement("eve", D(2023, 1, 1)),
            loan_disbursement("eve", D(2023, 1, 20)),  # index loan: most recent
            outgoing_spend("zzz_anchor", D(2023, 6, 1), 5.0, "Cash Withdrawal at Agent"),
        ]
        df = _categorize_txtype(_parse_timestamp(spark_session.createDataFrame(rows, RAW_COLUMNS)))
        cutoffs = _derive_loan_cutoffs(df, min_followup_days=60).toPandas()
        row = cutoffs.set_index("user_id").loc["eve"]
        assert pd.Timestamp(row["last_loan_ts"]) == pd.Timestamp(2023, 1, 20)


class TestDeriveLabels:
    def test_clean_repayer_labelled_good(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60)
        labels = derive_labels(core_df_ts, cutoffs).set_index("user_id")
        assert labels.loc["alice", "credit_risk_label"] == 0

    def test_penalised_but_repaid_labelled_risky(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60)
        labels = derive_labels(core_df_ts, cutoffs).set_index("user_id")
        assert labels.loc["bob", "credit_risk_label"] == 1

    def test_never_repaid_labelled_default(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60)
        labels = derive_labels(core_df_ts, cutoffs).set_index("user_id")
        assert labels.loc["carol", "credit_risk_label"] == 2

    def test_right_censored_borrower_absent_from_labels(self, core_df_ts):
        cutoffs = _derive_loan_cutoffs(core_df_ts, min_followup_days=60)
        labels = derive_labels(core_df_ts, cutoffs)
        assert "dave" not in set(labels["user_id"])

    def test_pre_loan_repayment_signal_does_not_count(self, spark_session):
        """A repayment-shaped transaction BEFORE the index loan must not be
        interpreted as a post-loan repayment (it would belong to a prior
        loan cycle, not the index loan being labelled)."""
        rows = [
            repayment("frank", D(2023, 1, 1)),  # before the index loan
            loan_disbursement("frank", D(2023, 1, 10)),
            # no repayment or penalty AFTER 2023-01-10
            outgoing_spend("zzz_anchor", D(2023, 6, 1), 5.0, "Cash Withdrawal at Agent"),
        ]
        df = _categorize_txtype(_parse_timestamp(spark_session.createDataFrame(rows, RAW_COLUMNS)))
        cutoffs = _derive_loan_cutoffs(df, min_followup_days=60)
        labels = derive_labels(df, cutoffs).set_index("user_id")
        assert labels.loc["frank", "credit_risk_label"] == 2


# ── Section 5: engineer_features + leakage test ─────────────────────────────

class TestEngineerFeatures:
    def _run(self, spark_session, df_ts):
        cutoffs = _derive_loan_cutoffs(df_ts, min_followup_days=60)
        borrower_ids_spark = spark_session.createDataFrame(
            cutoffs.select("user_id").toPandas(), schema="user_id string"
        )
        return engineer_features(df_ts, borrower_ids_spark, cutoffs), cutoffs

    def test_runs_without_error_and_has_expected_columns(self, spark_session, core_df_ts):
        features, _ = self._run(spark_session, core_df_ts)
        assert "user_id" in features.columns
        assert "obs_txn_count" in features.columns
        assert set(features["user_id"]) == {"alice", "bob", "carol"}

    def test_no_leakage_from_post_cutoff_transaction(self, spark_session):
        """Adding a transaction strictly AFTER a borrower's index loan must
        not change that borrower's engineered features at all."""
        base_rows = _build_core_rows()
        df_base = _categorize_txtype(
            _parse_timestamp(spark_session.createDataFrame(base_rows, RAW_COLUMNS))
        )
        features_base, _ = self._run(spark_session, df_base)

        leaky_rows = base_rows + [
            # A large, distinctive post-cutoff transaction for alice
            # (loan on 2023-01-10) that should be fully excluded.
            outgoing_spend("alice", D(2023, 1, 16), 999_999.0, "Pay Bill Payment"),
        ]
        df_leaky = _categorize_txtype(
            _parse_timestamp(spark_session.createDataFrame(leaky_rows, RAW_COLUMNS))
        )
        features_leaky, _ = self._run(spark_session, df_leaky)

        base_alice = features_base.set_index("user_id").loc["alice"].sort_index()
        leaky_alice = features_leaky.set_index("user_id").loc["alice"].sort_index()
        pd.testing.assert_series_equal(base_alice, leaky_alice, check_names=False)

    def test_other_users_unaffected_by_one_users_extra_transaction(self, spark_session):
        base_rows = _build_core_rows()
        df_base = _categorize_txtype(
            _parse_timestamp(spark_session.createDataFrame(base_rows, RAW_COLUMNS))
        )
        features_base, _ = self._run(spark_session, df_base)

        leaky_rows = base_rows + [
            outgoing_spend("alice", D(2023, 1, 16), 999_999.0, "Pay Bill Payment"),
        ]
        df_leaky = _categorize_txtype(
            _parse_timestamp(spark_session.createDataFrame(leaky_rows, RAW_COLUMNS))
        )
        features_leaky, _ = self._run(spark_session, df_leaky)

        base_bob = features_base.set_index("user_id").loc["bob"].sort_index()
        leaky_bob = features_leaky.set_index("user_id").loc["bob"].sort_index()
        pd.testing.assert_series_equal(base_bob, leaky_bob, check_names=False)

    def test_pre_cutoff_transaction_does_affect_features(self, spark_session):
        """Sanity check for the leakage test above: a PRE-cutoff addition
        should actually change the count, proving the filter isn't simply
        dropping everything."""
        base_rows = _build_core_rows()
        df_base = _categorize_txtype(
            _parse_timestamp(spark_session.createDataFrame(base_rows, RAW_COLUMNS))
        )
        features_base, _ = self._run(spark_session, df_base)

        more_rows = base_rows + [
            outgoing_spend("alice", D(2023, 1, 6), 40.0, "Pay Bill Payment"),
        ]
        df_more = _categorize_txtype(
            _parse_timestamp(spark_session.createDataFrame(more_rows, RAW_COLUMNS))
        )
        features_more, _ = self._run(spark_session, df_more)

        base_count = features_base.set_index("user_id").loc["alice", "obs_txn_count"]
        more_count = features_more.set_index("user_id").loc["alice", "obs_txn_count"]
        assert more_count == base_count + 1


# ── Section 6: build_sequences_spark and build_pipeline (integration) ───────

class TestBuildPipelineIntegration:
    def test_build_pipeline_writes_expected_csvs(self, spark_session, core_df, tmp_path):
        out_dir = tmp_path / "pipeline_out"
        features_pd, labels_pd = build_pipeline(core_df, output_dir=out_dir, min_followup_days=60)

        features_path = out_dir / "user_features.csv"
        labels_path = out_dir / "user_labels.csv"
        assert features_path.exists()
        assert labels_path.exists()

        # Columns expected by CreditRiskDataLoader (credit_model.py)
        assert "user_id" in features_pd.columns
        assert "obs_txn_count" in features_pd.columns
        assert "user_id" in labels_pd.columns
        assert "credit_risk_label" in labels_pd.columns

        assert set(labels_pd["user_id"]) == {"alice", "bob", "carol"}
        labels_by_user = labels_pd.set_index("user_id")["credit_risk_label"]
        assert labels_by_user["alice"] == 0
        assert labels_by_user["bob"] == 1
        assert labels_by_user["carol"] == 2

        # Round-trip from disk matches the in-memory frames
        on_disk_features = pd.read_csv(features_path)
        on_disk_labels = pd.read_csv(labels_path)
        assert set(on_disk_features["user_id"]) == set(features_pd["user_id"])
        assert set(on_disk_labels["user_id"]) == set(labels_pd["user_id"])

    def test_build_sequences_spark_writes_consumable_npz(self, spark_session, core_df, tmp_path):
        output_path = tmp_path / "sequences" / "lstm_sequences_raw.npz"
        returned_path = build_sequences_spark(
            core_df, min_followup_days=60, max_seq_len=10, output_path=output_path
        )

        assert Path(returned_path) == output_path
        assert output_path.exists()

        with np.load(output_path, allow_pickle=True) as data:
            assert "user_ids" in data
            assert "sequences" in data
            assert "feature_names" in data
            assert "metadata_json" in data

            user_ids = list(data["user_ids"])
            assert set(user_ids) == {"alice", "bob", "carol"}

            sequences = data["sequences"]
            assert sequences.shape == (len(user_ids), 10, len(SEQ_FEATURE_NAMES))

            feature_names = list(data["feature_names"])
            assert feature_names == list(SEQ_FEATURE_NAMES)

            metadata = json.loads(str(data["metadata_json"]))
            assert metadata["sequence_version"] == RAW_SEQUENCE_VERSION
            assert metadata["borrower_count"] == len(user_ids)
            assert metadata["max_seq_len"] == 10
            assert metadata["feature_names"] == list(SEQ_FEATURE_NAMES)
            assert metadata["user_ids_sha256"] == _hash_user_ids(user_ids)
