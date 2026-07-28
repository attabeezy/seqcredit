"""Tests for seqcredit_model.config path resolution."""
import importlib
from pathlib import Path

import pytest

from seqcredit_model import config


def reload_config():
    """Reload config.py fresh so module-level constants pick up env changes."""
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore_config():
    """Ensure config module state (and relevant env vars) are restored after each test."""
    yield
    for var in [
        "SEQCREDIT_DATA_DIR",
        "SEQCREDIT_LSTM_CACHE_FILE",
        "SEQCREDIT_RAW_SEQ_FILE",
        "SEQCREDIT_USER_FEATURES_FILE",
        "SEQCREDIT_USER_LABELS_FILE",
    ]:
        __import__("os").environ.pop(var, None)
    reload_config()


class TestDefaultPaths:
    def test_project_root_is_repo_root(self):
        # config.py lives at <root>/src/seqcredit_model/config.py
        assert (config.PROJECT_ROOT / "src" / "seqcredit_model" / "config.py").exists()

    def test_data_dir_defaults_under_project_root(self):
        assert config.DATA_DIR == config.PROJECT_ROOT / "data"

    def test_derived_file_paths_default_locations(self):
        assert config.USER_FEATURES_FILE == config.DATA_DIR / "user_features.csv"
        assert config.USER_LABELS_FILE == config.DATA_DIR / "user_labels.csv"
        assert config.LSTM_CACHE_FILE == config.DATA_DIR / "lstm_sequences.npz"
        assert config.RAW_SEQ_FILE == config.DATA_DIR / "lstm_sequences_raw.npz"
        assert config.TRANSACTIONS_DIR == config.DATA_DIR / "user_transactions"
        assert config.LEGACY_DIR == config.DATA_DIR / "legacy"
        assert config.MODELS_DIR == config.PROJECT_ROOT / "models"
        assert (
            config.SYNTHETIC_PARAMS_FILE
            == config.PROJECT_ROOT / "src" / "synthetic_params.json"
        )

    def test_random_seed_constant(self):
        assert config.RANDOM_SEED == 42


class TestResolveStoragePath:
    def test_plain_local_path_returned_as_is(self):
        assert config._resolve_storage_path("relative/path") == Path("relative/path")

    def test_absolute_local_path_returned_as_is(self):
        assert config._resolve_storage_path("/abs/path/file.csv") == Path(
            "/abs/path/file.csv"
        )

    def test_dbfs_path_translated_to_dbfs_mount(self):
        result = config._resolve_storage_path("dbfs:/mnt/data/file.csv")
        assert result == Path("/dbfs/mnt/data/file.csv")

    def test_dbfs_path_with_extra_leading_slash(self):
        # removeprefix("dbfs:/") on "dbfs://mnt/x" leaves "/mnt/x"; lstrip("/") strips it.
        result = config._resolve_storage_path("dbfs://mnt/x")
        assert result == Path("/dbfs/mnt/x")


class TestResolveRuntimePath:
    def test_no_override_returns_default(self, monkeypatch):
        monkeypatch.delenv("SEQCREDIT_DATA_DIR", raising=False)
        default = Path("/default/data")
        assert config._resolve_runtime_path("SEQCREDIT_DATA_DIR", default) == default

    def test_override_replaces_default(self, monkeypatch, tmp_path):
        override_dir = tmp_path / "custom_data"
        monkeypatch.setenv("SEQCREDIT_DATA_DIR", str(override_dir))
        result = config._resolve_runtime_path("SEQCREDIT_DATA_DIR", Path("/default"))
        assert result == Path(str(override_dir))

    def test_empty_string_override_falls_back_to_default(self, monkeypatch):
        # `if override:` treats "" as falsy, so an explicitly-empty env var
        # should NOT be treated as an override.
        monkeypatch.setenv("SEQCREDIT_DATA_DIR", "")
        default = Path("/default/data")
        assert config._resolve_runtime_path("SEQCREDIT_DATA_DIR", default) == default

    def test_dbfs_override_is_translated(self, monkeypatch):
        monkeypatch.setenv("SEQCREDIT_DATA_DIR", "dbfs:/mnt/team/data")
        result = config._resolve_runtime_path("SEQCREDIT_DATA_DIR", Path("/default"))
        assert result == Path("/dbfs/mnt/team/data")


class TestRuntimeGetterFunctions:
    def test_get_runtime_lstm_cache_file_default(self, monkeypatch):
        monkeypatch.delenv("SEQCREDIT_LSTM_CACHE_FILE", raising=False)
        assert config.get_runtime_lstm_cache_file() == config.LSTM_CACHE_FILE

    def test_get_runtime_lstm_cache_file_override(self, monkeypatch):
        monkeypatch.setenv("SEQCREDIT_LSTM_CACHE_FILE", "dbfs:/mnt/cache/seq.npz")
        assert config.get_runtime_lstm_cache_file() == Path("/dbfs/mnt/cache/seq.npz")

    def test_get_runtime_raw_seq_file_default(self, monkeypatch):
        monkeypatch.delenv("SEQCREDIT_RAW_SEQ_FILE", raising=False)
        assert config.get_runtime_raw_seq_file() == config.RAW_SEQ_FILE

    def test_get_runtime_raw_seq_file_override(self, monkeypatch, tmp_path):
        override = tmp_path / "raw.npz"
        monkeypatch.setenv("SEQCREDIT_RAW_SEQ_FILE", str(override))
        assert config.get_runtime_raw_seq_file() == Path(str(override))

    def test_get_runtime_user_features_file_default(self, monkeypatch):
        monkeypatch.delenv("SEQCREDIT_USER_FEATURES_FILE", raising=False)
        assert config.get_runtime_user_features_file() == config.USER_FEATURES_FILE

    def test_get_runtime_user_features_file_override(self, monkeypatch, tmp_path):
        override = tmp_path / "features.csv"
        monkeypatch.setenv("SEQCREDIT_USER_FEATURES_FILE", str(override))
        assert config.get_runtime_user_features_file() == Path(str(override))

    def test_get_runtime_user_labels_file_default(self, monkeypatch):
        monkeypatch.delenv("SEQCREDIT_USER_LABELS_FILE", raising=False)
        assert config.get_runtime_user_labels_file() == config.USER_LABELS_FILE

    def test_get_runtime_user_labels_file_override(self, monkeypatch, tmp_path):
        override = tmp_path / "labels.csv"
        monkeypatch.setenv("SEQCREDIT_USER_LABELS_FILE", str(override))
        assert config.get_runtime_user_labels_file() == Path(str(override))

    def test_get_runtime_user_labels_file_empty_override_falls_back(self, monkeypatch):
        # Empty string override should behave like "unset" (falsy check).
        monkeypatch.setenv("SEQCREDIT_USER_LABELS_FILE", "")
        assert config.get_runtime_user_labels_file() == config.USER_LABELS_FILE


class TestModuleReloadPicksUpDataDirOverride:
    def test_data_dir_env_var_affects_module_level_default_on_reload(
        self, monkeypatch, tmp_path
    ):
        override_dir = tmp_path / "override_data"
        monkeypatch.setenv("SEQCREDIT_DATA_DIR", str(override_dir))
        reloaded = reload_config()
        try:
            assert reloaded.DATA_DIR == Path(str(override_dir))
            assert reloaded.USER_FEATURES_FILE == Path(str(override_dir)) / "user_features.csv"
        finally:
            monkeypatch.delenv("SEQCREDIT_DATA_DIR", raising=False)
            reload_config()
