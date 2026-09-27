import pandas as pd
import pytest

from seqcredit_mvp.features import validate_transactions


@pytest.fixture
def valid_transaction_row():
    return {
        "borrower_id": "BORROWER_1",
        "transaction_id": "TXN_001",
        "timestamp": "2026-03-01T10:00:00",
        "transaction_type": "TRANSFER",
        "amount": 100.0,
        "balance_before": 500.0,
        "balance_after": 400.0,
        "fee": 1.5,
        "recipient_id": "REC_99",
    }


def test_validation_accepts_valid_batch(valid_transaction_row):
    df = pd.DataFrame([valid_transaction_row])
    cleaned = validate_transactions(df)
    assert len(cleaned) == 1
    assert cleaned["borrower_id"].iloc[0] == "BORROWER_1"
    assert cleaned["amount"].iloc[0] == 100.0


def test_validation_rejects_empty_dataframe():
    empty_df = pd.DataFrame(columns=[
        "borrower_id", "transaction_id", "timestamp", "transaction_type",
        "amount", "balance_before", "balance_after", "fee", "recipient_id"
    ])
    with pytest.raises(ValueError, match="The transaction file is empty"):
        validate_transactions(empty_df)


def test_validation_rejects_missing_required_columns(valid_transaction_row):
    del valid_transaction_row["fee"]
    with pytest.raises(ValueError, match="Missing required columns.*fee"):
        validate_transactions(pd.DataFrame([valid_transaction_row]))


def test_validation_rejects_unparseable_timestamps(valid_transaction_row):
    valid_transaction_row["timestamp"] = "not-a-valid-timestamp"
    with pytest.raises(ValueError, match="timestamp values could not be parsed"):
        validate_transactions(pd.DataFrame([valid_transaction_row]))


@pytest.mark.parametrize("col", ["amount", "balance_before", "balance_after", "fee"])
def test_validation_rejects_negative_numeric_fields(valid_transaction_row, col):
    valid_transaction_row[col] = -10.0
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_transactions(pd.DataFrame([valid_transaction_row]))


@pytest.mark.parametrize("col", ["amount", "balance_before", "balance_after", "fee"])
def test_validation_rejects_non_numeric_fields(valid_transaction_row, col):
    valid_transaction_row[col] = "abc"
    with pytest.raises(ValueError, match=f"Column '{col}' contains non-numeric or missing values"):
        validate_transactions(pd.DataFrame([valid_transaction_row]))


@pytest.mark.parametrize("col", ["borrower_id", "transaction_id", "transaction_type", "recipient_id"])
def test_validation_rejects_blank_or_whitespace_identifiers(valid_transaction_row, col):
    valid_transaction_row[col] = "   "
    with pytest.raises(ValueError, match=f"Column '{col}' cannot contain blank values"):
        validate_transactions(pd.DataFrame([valid_transaction_row]))


def test_validation_rejects_duplicate_transaction_ids(valid_transaction_row):
    row2 = dict(valid_transaction_row)
    row2["amount"] = 200.0
    with pytest.raises(ValueError, match="transaction_id values must be unique"):
        validate_transactions(pd.DataFrame([valid_transaction_row, row2]))


def test_validation_preserves_and_ignores_extra_columns(valid_transaction_row):
    valid_transaction_row["extra_metadata"] = "arbitrary_info"
    valid_transaction_row["another_column"] = 12345
    cleaned = validate_transactions(pd.DataFrame([valid_transaction_row]))
    assert "extra_metadata" in cleaned.columns
    assert cleaned["extra_metadata"].iloc[0] == "arbitrary_info"
