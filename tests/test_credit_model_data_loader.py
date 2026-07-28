"""Tests for seqcredit_model.credit_model.CreditRiskDataLoader."""
import numpy as np
import pandas as pd
import pytest

from seqcredit_model.credit_model import CreditRiskDataLoader
from seqcredit_model.synthesize import CalibratedMoMoDataGenerator
from seqcredit_model.pipeline import build_user_feature_dataset


@pytest.fixture
def loader(tiny_synthetic_dataset):
    return CreditRiskDataLoader(
        features_path=str(tiny_synthetic_dataset["user_features_file"]),
        summaries_path=str(tiny_synthetic_dataset["user_labels_file"]),
        transactions_dir=str(tiny_synthetic_dataset["transactions_dir"]),
    )


@pytest.fixture(scope="module")
def larger_synthetic_dataset(tmp_path_factory):
    """A somewhat larger synthetic dataset than tiny_synthetic_dataset.

    tiny_synthetic_dataset (40 users, seed=42) produces only 1 "default" (class 2)
    user, which is too few for stratified train/test splitting (sklearn requires
    >= 2 members per class). Splitting-related tests need at least 2 members in
    the minority class, so we generate a bigger pool here (150 users), locally,
    without touching the shared conftest.py fixture.
    """
    tmp_path = tmp_path_factory.mktemp("larger_dataset")
    data_dir = tmp_path / "data"
    transactions_dir = data_dir / "user_transactions"
    user_features_file = data_dir / "user_features.csv"
    user_labels_file = data_dir / "user_labels.csv"

    generator = CalibratedMoMoDataGenerator(
        n_users=150,
        avg_transactions_per_user=20,
        output_dir=str(transactions_dir),
        seed=42,
    )
    generator.generate_dataset()

    build_user_feature_dataset(
        transactions_dir=str(transactions_dir),
        output_path=str(user_features_file),
    )

    labels_df = pd.read_csv(data_dir / "user_labels.csv")
    labels_df["is_bad"] = (labels_df["credit_risk_label"] > 0).astype(int)
    labels_df.set_index("user_id", inplace=True)
    labels_df[["credit_risk_label", "is_bad"]].to_csv(user_labels_file)

    return {
        "data_dir": data_dir,
        "transactions_dir": transactions_dir,
        "user_features_file": user_features_file,
        "user_labels_file": user_labels_file,
    }


@pytest.fixture
def loader_large(larger_synthetic_dataset):
    return CreditRiskDataLoader(
        features_path=str(larger_synthetic_dataset["user_features_file"]),
        summaries_path=str(larger_synthetic_dataset["user_labels_file"]),
        transactions_dir=str(larger_synthetic_dataset["transactions_dir"]),
    )


class TestLoadStaticData:
    def test_returns_dataframe_and_series(self, loader):
        X, y = loader.load_static_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_no_leakage_columns_in_X(self, loader):
        X, y = loader.load_static_data()
        leaky_cols = {
            "user_id",
            "credit_risk_label",
            "credit_archetype",
            "default",
            "gen_txn_count",
            "final_credit_limit",
            "loans_taken",
        }
        assert leaky_cols.isdisjoint(set(X.columns))

    def test_y_is_binary_default_target(self, loader):
        X, y = loader.load_static_data()
        assert set(np.unique(y.values)).issubset({0, 1})

    def test_non_borrowers_filtered_out(self, loader, tiny_synthetic_dataset):
        labels = pd.read_csv(tiny_synthetic_dataset["user_labels_file"])
        n_non_borrowers = (labels["credit_risk_label"] == -1).sum()
        X, y = loader.load_static_data()
        assert len(X) == len(labels) - n_non_borrowers

    def test_X_is_numeric(self, loader):
        X, y = loader.load_static_data()
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        assert non_numeric == [], f"Non-numeric feature columns found: {non_numeric}"

    def test_caches_static_data(self, loader):
        X1, y1 = loader.load_static_data()
        assert loader._static_data is not None
        X2, y2 = loader.load_static_data()
        pd.testing.assert_frame_equal(X1, X2)


class TestPrepareStaticSplits:
    def test_returns_expected_keys(self, loader_large):
        splits = loader_large.prepare_static_splits()
        expected_keys = {
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "X_train_scaled",
            "X_test_scaled",
            "scaler",
            "feature_names",
        }
        assert expected_keys == set(splits.keys())

    def test_shapes_are_consistent(self, loader_large):
        splits = loader_large.prepare_static_splits()
        assert len(splits["X_train"]) == len(splits["y_train"])
        assert len(splits["X_test"]) == len(splits["y_test"])
        assert splits["X_train_scaled"].shape == splits["X_train"].shape
        assert splits["X_test_scaled"].shape == splits["X_test"].shape

    def test_train_test_split_ratio_close_to_test_size(self, larger_synthetic_dataset):
        loader = CreditRiskDataLoader(
            features_path=str(larger_synthetic_dataset["user_features_file"]),
            summaries_path=str(larger_synthetic_dataset["user_labels_file"]),
            transactions_dir=str(larger_synthetic_dataset["transactions_dir"]),
            test_size=0.25,
        )
        splits = loader.prepare_static_splits()
        total = len(splits["X_train"]) + len(splits["X_test"])
        actual_ratio = len(splits["X_test"]) / total
        assert abs(actual_ratio - 0.25) < 0.1

    def test_train_and_test_user_ids_disjoint(self, loader_large):
        loader_large.prepare_static_splits()
        assert loader_large._train_user_ids.isdisjoint(loader_large._test_user_ids)
        assert len(loader_large._train_user_ids) > 0
        assert len(loader_large._test_user_ids) > 0

    def test_ordered_user_ids_match_row_counts(self, loader_large):
        splits = loader_large.prepare_static_splits()
        assert len(loader_large._train_user_ids_ordered) == len(splits["X_train"])
        assert len(loader_large._test_user_ids_ordered) == len(splits["X_test"])

    def test_feature_names_match_X_columns(self, loader_large):
        splits = loader_large.prepare_static_splits()
        assert splits["feature_names"] == list(splits["X_train"].columns)

    def test_calls_load_static_data_implicitly(self, loader_large):
        assert loader_large._static_data is None
        loader_large.prepare_static_splits()
        assert loader_large._static_data is not None

    def test_scaler_fit_on_train_only(self, loader_large):
        splits = loader_large.prepare_static_splits()
        # Scaled train data should have ~zero mean since scaler was fit on it.
        means = splits["X_train_scaled"].mean(axis=0)
        assert np.allclose(means, 0, atol=1e-6)


class TestLoadSequencesConvention:
    @pytest.fixture(autouse=True)
    def _isolated_sequence_cache_paths(self, tmp_path, monkeypatch):
        """load_sequences() reads/writes LSTM sequence caches at paths resolved
        via SEQCREDIT_LSTM_CACHE_FILE / SEQCREDIT_RAW_SEQ_FILE (defaulting to
        the real repo's data/ directory). Without this override, tests would
        read a stale pre-existing repo cache (data/lstm_sequences.npz, ~90MB,
        built by a previous real run with a different max_seq_len/user set)
        and would also pollute the real repo data/ directory with test-generated
        files.

        NOTE: even with these env vars set, load_sequences()'s own `cache_path`
        parameter default is `str(LSTM_CACHE_FILE)` -- a module-level constant
        frozen to the real repo path *at import time* (see credit_model.py
        line 518). The function's un-cached "cache_file = Path(cache_path)"
        fallback branch (credit_model.py line 640) keys off that frozen literal
        rather than calling get_runtime_lstm_cache_file() again, so it is NOT
        redirected by SEQCREDIT_LSTM_CACHE_FILE and would still hit the real
        repo file if reached. We avoid ever reaching that branch below by
        passing our own explicit tmp cache_path to load_sequences().
        """
        monkeypatch.setenv(
            "SEQCREDIT_LSTM_CACHE_FILE", str(tmp_path / "lstm_sequences.npz")
        )
        monkeypatch.setenv(
            "SEQCREDIT_RAW_SEQ_FILE", str(tmp_path / "lstm_sequences_raw.npz")
        )
        self._tmp_cache_path = str(tmp_path / "lstm_sequences_explicit.npz")

    def test_load_sequences_without_prepare_static_splits_still_works(self, loader_large):
        """The docstring says call prepare_static_splits() first, but the code
        itself falls back to calling it implicitly (self._train_user_ids is None
        triggers prepare_static_splits() inside load_sequences()). Document that
        actual (permissive) behavior rather than assuming it raises."""
        assert loader_large._train_user_ids is None
        result = loader_large.load_sequences(
            max_seq_len=10, cache_path=self._tmp_cache_path
        )
        assert loader_large._train_user_ids is not None
        assert set(result.keys()) == {
            "X_train_seq",
            "X_test_seq",
            "y_train",
            "y_test",
            "feature_names",
        }

    def test_load_sequences_shapes(self, loader_large):
        loader_large.prepare_static_splits()
        result = loader_large.load_sequences(
            max_seq_len=10, cache_path=self._tmp_cache_path
        )
        assert result["X_train_seq"].shape[0] == len(result["y_train"])
        assert result["X_test_seq"].shape[0] == len(result["y_test"])
        assert result["X_train_seq"].shape[1] == 10
        assert result["X_test_seq"].shape[1] == 10


class TestClassWeightsAndScalePosWeight:
    @pytest.fixture
    def deterministic_loader(self, tmp_path):
        """A fully deterministic labels/features pair: 8 good, 2 bad (default)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        transactions_dir = data_dir / "user_transactions"
        transactions_dir.mkdir()

        n_good = 8
        n_bad = 2
        user_ids = [f"user_{i}" for i in range(n_good + n_bad)]
        labels = [0] * n_good + [2] * n_bad  # 0=good, 2=default

        features_df = pd.DataFrame(
            {
                "user_id": user_ids,
                "obs_txn_count": [10] * len(user_ids),
                "feature_a": np.arange(len(user_ids), dtype=float),
                "feature_b": np.arange(len(user_ids), dtype=float) * 2,
                "total_loan_volume": np.zeros(len(user_ids)),
                "avg_loan_amount": np.zeros(len(user_ids)),
                "max_loan_amount": np.zeros(len(user_ids)),
                "loan_to_total_volume_ratio": np.zeros(len(user_ids)),
                "pct_credit_transactions": np.zeros(len(user_ids)),
                "loan_timing_in_sequence": np.zeros(len(user_ids)),
                "avg_balance_at_loan": np.zeros(len(user_ids)),
                "min_balance_at_loan": np.zeros(len(user_ids)),
                "balance_to_loan_ratio_at_disbursement": np.zeros(len(user_ids)),
            }
        )
        labels_df = pd.DataFrame(
            {
                "user_id": user_ids,
                "credit_risk_label": labels,
                "gen_txn_count": [10] * len(user_ids),
            }
        )

        features_path = data_dir / "user_features.csv"
        labels_path = data_dir / "user_labels.csv"
        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)

        return CreditRiskDataLoader(
            features_path=str(features_path),
            summaries_path=str(labels_path),
            transactions_dir=str(transactions_dir),
        )

    def test_get_class_weights_known_distribution(self, deterministic_loader):
        # 8 good (y=0), 2 default (y=1) -> balanced weights favor the minority class.
        weights = deterministic_loader.get_class_weights()
        assert set(weights.keys()) == {0, 1}
        assert weights[1] > weights[0]
        # sklearn balanced formula: n_samples / (n_classes * n_samples_per_class)
        assert weights[0] == pytest.approx(10 / (2 * 8))
        assert weights[1] == pytest.approx(10 / (2 * 2))

    def test_get_scale_pos_weight_known_distribution(self, deterministic_loader):
        scale_pos_weight = deterministic_loader.get_scale_pos_weight()
        # n_neg / n_pos = 8 / 2 = 4.0
        assert scale_pos_weight == pytest.approx(4.0)

    def test_get_class_weights_triggers_load_static_data(self, deterministic_loader):
        assert deterministic_loader._static_data is None
        deterministic_loader.get_class_weights()
        assert deterministic_loader._static_data is not None

    def test_get_scale_pos_weight_triggers_load_static_data(self, deterministic_loader):
        assert deterministic_loader._static_data is None
        deterministic_loader.get_scale_pos_weight()
        assert deterministic_loader._static_data is not None


class TestValidateData:
    def test_raises_on_no_overlap(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        transactions_dir = data_dir / "user_transactions"
        transactions_dir.mkdir()

        features_df = pd.DataFrame(
            {"user_id": ["a", "b"], "obs_txn_count": [1, 2]}
        )
        labels_df = pd.DataFrame(
            {
                "user_id": ["c", "d"],
                "credit_risk_label": [0, 1],
                "gen_txn_count": [1, 2],
            }
        )
        features_path = data_dir / "user_features.csv"
        labels_path = data_dir / "user_labels.csv"
        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)

        loader = CreditRiskDataLoader(
            features_path=str(features_path),
            summaries_path=str(labels_path),
            transactions_dir=str(transactions_dir),
        )
        with pytest.raises(ValueError, match="No users overlap"):
            loader.load_static_data()

    def test_raises_on_invalid_label_values(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        transactions_dir = data_dir / "user_transactions"
        transactions_dir.mkdir()

        features_df = pd.DataFrame(
            {"user_id": ["a", "b"], "obs_txn_count": [1, 2]}
        )
        labels_df = pd.DataFrame(
            {
                "user_id": ["a", "b"],
                "credit_risk_label": [0, 99],  # 99 is invalid
                "gen_txn_count": [1, 2],
            }
        )
        features_path = data_dir / "user_features.csv"
        labels_path = data_dir / "user_labels.csv"
        features_df.to_csv(features_path, index=False)
        labels_df.to_csv(labels_path, index=False)

        loader = CreditRiskDataLoader(
            features_path=str(features_path),
            summaries_path=str(labels_path),
            transactions_dir=str(transactions_dir),
        )
        with pytest.raises(ValueError, match="Unexpected credit_risk_label"):
            loader.load_static_data()
