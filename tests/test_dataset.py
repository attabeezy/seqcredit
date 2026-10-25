"""Single-file round trips and fail-closed provenance checks."""

from copy import deepcopy
import gzip
from pathlib import Path

import numpy as np
import pytest

from dataset_io import build, load
from experiment_v2 import load_contract


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    contract = deepcopy(load_contract())
    contract["dataset"] = {"borrowers": 1000}
    path = tmp_path_factory.mktemp("dataset") / "yara_telecel_dump.csv.gz"
    build(contract, path)
    return path, contract


def test_one_file_contains_all_events_and_reproduces_splits(example):
    path, contract = example
    first, splits, reference = load(path, contract)
    second, again, _ = load(path, contract)
    assert list(path.parent.iterdir()) == [path]
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [600, 200, 200]
    for key in splits:
        np.testing.assert_array_equal(splits[key], again[key])
    np.testing.assert_array_equal(first["labels"], second["labels"])
    assert reference["borrowers"] == 1000


def test_changed_contract_is_rejected(example):
    path, contract = example
    changed = deepcopy(contract)
    changed["history_process"]["trend_log_scale"] += 0.1
    with pytest.raises(ValueError, match="provenance mismatch"):
        load(path, changed)


def test_repeated_build_keeps_identical_bytes(example):
    path, contract = example
    before = path.read_bytes()
    build(contract, path)
    assert path.read_bytes() == before


def test_fresh_build_is_byte_deterministic(example, tmp_path):
    path, contract = example
    rebuilt = tmp_path / path.name
    build(contract, rebuilt)
    assert rebuilt.read_bytes() == path.read_bytes()


def test_truncated_or_tampered_file_is_rejected(example, tmp_path):
    path, contract = example
    bad = tmp_path / path.name
    text = gzip.decompress(path.read_bytes()).decode()
    header, body = text.split("\n", 1)
    with gzip.open(bad, "wt", encoding="utf-8") as handle:
        handle.write(header + "\n" + body.split("\n")[0] + "\n")
    with pytest.raises(ValueError):
        load(bad, contract)
