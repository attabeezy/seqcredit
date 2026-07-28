"""Tests for seqcredit_model.reset artifact cleanup helpers.

These tests exercise remove_files / remove_dirs / collect_existing as pure
functions against tmp_path fixtures only -- never against real repo data.
Because reset.py's print statements call path.relative_to(reset.PROJECT_ROOT),
we monkeypatch reset.PROJECT_ROOT to tmp_path for the duration of each test so
that arbitrary tmp_path-rooted paths can be printed without raising.
"""
from pathlib import Path

import pytest

from seqcredit_model import reset


@pytest.fixture(autouse=True)
def _project_root_is_tmp(tmp_path, monkeypatch):
    """Point reset.PROJECT_ROOT at tmp_path so relative_to() calls succeed
    for paths we create under tmp_path, and so nothing under the real repo
    root can ever be touched by these tests."""
    monkeypatch.setattr(reset, "PROJECT_ROOT", tmp_path)
    yield


class TestRemoveFiles:
    def test_removes_existing_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("x")
        f2.write_text("y")

        reset.remove_files([f1, f2])

        assert not f1.exists()
        assert not f2.exists()

    def test_skips_nonexistent_files_without_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        assert not missing.exists()

        # Should not raise.
        reset.remove_files([missing])

        assert not missing.exists()

    def test_leaves_untouched_files_alone(self, tmp_path):
        keep = tmp_path / "keep.txt"
        keep.write_text("keep me")
        target = tmp_path / "remove_me.txt"
        target.write_text("bye")

        reset.remove_files([target])

        assert not target.exists()
        assert keep.exists()
        assert keep.read_text() == "keep me"

    def test_empty_list_is_noop(self, tmp_path):
        reset.remove_files([])  # should not raise


class TestRemoveDirs:
    def test_removes_existing_dir_and_contents(self, tmp_path):
        d = tmp_path / "some_dir"
        d.mkdir()
        (d / "file1.csv").write_text("1")
        (d / "file2.csv").write_text("2")

        reset.remove_dirs([d])

        assert not d.exists()

    def test_skips_nonexistent_dir_without_error(self, tmp_path):
        missing_dir = tmp_path / "nope"
        assert not missing_dir.exists()

        reset.remove_dirs([missing_dir])  # should not raise

        assert not missing_dir.exists()

    def test_counts_only_files_not_subdirs_but_still_removes_all(self, tmp_path):
        d = tmp_path / "nested"
        d.mkdir()
        (d / "top.txt").write_text("1")
        sub = d / "subdir"
        sub.mkdir()
        (sub / "inner.txt").write_text("2")

        # remove_dirs uses path.glob("*") (non-recursive) purely for the
        # printed count, but shutil.rmtree removes everything regardless.
        reset.remove_dirs([d])

        assert not d.exists()


class TestCollectExisting:
    def test_lists_only_existing_files(self, tmp_path):
        present = tmp_path / "present.csv"
        present.write_text("data")
        missing = tmp_path / "missing.csv"

        labels = reset.collect_existing([present, missing], [])

        assert len(labels) == 1
        assert "present.csv" in labels[0]
        assert "missing.csv" not in "".join(labels)

    def test_lists_existing_dirs_with_file_count(self, tmp_path):
        d = tmp_path / "transactions"
        d.mkdir()
        (d / "u1.csv").write_text("1")
        (d / "u2.csv").write_text("2")

        labels = reset.collect_existing([], [d])

        assert len(labels) == 1
        assert "transactions" in labels[0]
        assert "2 files" in labels[0]

    def test_dir_file_count_is_recursive(self, tmp_path):
        d = tmp_path / "transactions"
        d.mkdir()
        (d / "u1.csv").write_text("1")
        sub = d / "sub"
        sub.mkdir()
        (sub / "u2.csv").write_text("2")

        labels = reset.collect_existing([], [d])

        assert "2 files" in labels[0]

    def test_returns_empty_list_when_nothing_exists(self, tmp_path):
        labels = reset.collect_existing(
            [tmp_path / "a.csv", tmp_path / "b.csv"], [tmp_path / "some_dir"]
        )
        assert labels == []

    def test_combines_files_and_dirs(self, tmp_path):
        f = tmp_path / "features.csv"
        f.write_text("data")
        d = tmp_path / "raw"
        d.mkdir()
        (d / "x.csv").write_text("1")

        labels = reset.collect_existing([f], [d])

        assert len(labels) == 2

    def test_does_not_delete_anything(self, tmp_path):
        f = tmp_path / "features.csv"
        f.write_text("data")
        d = tmp_path / "raw"
        d.mkdir()
        (d / "x.csv").write_text("1")

        reset.collect_existing([f], [d])

        # collect_existing must be read-only.
        assert f.exists()
        assert d.exists()


class TestModuleLevelConstants:
    def test_target_lists_are_defined_and_nonempty(self):
        assert len(reset.MODELS) > 0
        assert len(reset.DATA_FILES) > 0
        assert len(reset.DATA_FULL_FILES) > 0
        assert len(reset.DATA_FULL_DIRS) > 0

    def test_all_targets_are_path_objects(self):
        for p in (
            reset.MODELS
            + reset.DATA_FILES
            + reset.DATA_FULL_FILES
            + reset.DATA_FULL_DIRS
        ):
            assert isinstance(p, Path)
