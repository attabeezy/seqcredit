import pandas as pd
import pytest

from seqcredit_mvp.features import build_feature_tables, validate_transactions
from seqcredit_mvp.synthetic import SyntheticConfig, generate_synthetic_cohort


def test_generator_is_reproducible_and_has_expected_shape():
    config = SyntheticConfig(n_borrowers=40, transactions_per_borrower=12, seed=7)
    tx_a, labels_a = generate_synthetic_cohort(config)
    tx_b, labels_b = generate_synthetic_cohort(config)
    pd.testing.assert_frame_equal(tx_a, tx_b)
    pd.testing.assert_frame_equal(labels_a, labels_b)
    assert len(tx_a) == 480
    assert labels_a["borrower_id"].nunique() == 40
    assert tx_a["synthetic_demo"].all()


def test_feature_contract_keeps_sequence_features_separate():
    transactions, _ = generate_synthetic_cohort(
        SyntheticConfig(n_borrowers=30, transactions_per_borrower=12, seed=9)
    )
    static, sequence = build_feature_tables(transactions)
    assert static.index.equals(sequence.index)
    assert set(static.columns) < set(sequence.columns)
    assert "amount_slope" not in static.columns
    assert "amount_slope" in sequence.columns


def test_validation_explains_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_transactions(pd.DataFrame({"borrower_id": ["DEMO_1"]}))


def test_validation_normalizes_numeric_identifiers_for_dashboard_keys():
    transactions, _ = generate_synthetic_cohort(
        SyntheticConfig(n_borrowers=2, transactions_per_borrower=4, seed=17)
    )
    transactions["borrower_id"] = transactions["borrower_id"].map({
        borrower_id: index for index, borrower_id in enumerate(transactions["borrower_id"].unique(), start=1)
    })
    clean = validate_transactions(transactions)
    assert clean["borrower_id"].map(type).eq(str).all()
