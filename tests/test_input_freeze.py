"""Accepted data must fail closed when its bytes or parameter baseline change."""

import json

import pytest

import input_freeze as freeze


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(freeze, "LOCK_PATH", tmp_path / "src/configs/frozen_inputs.json")
    monkeypatch.setattr(freeze, "ACCEPTED_PATH", tmp_path / "src/configs/accepted_dataset.json")
    config = {"seed": 42, "dataset": {"borrowers": 3}}
    for name, relative in freeze.INPUTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config if name == "contract" else {"evidence": name}))
    for relative in freeze.GENERATOR_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original generator\n")
    data = tmp_path / freeze.DATASET_RELATIVE_PATH
    data.parent.mkdir(parents=True)
    data.write_bytes(b"accepted sample bytes")
    lock = freeze.write_lock()
    audit = {"status": "consistent_with_configured_parameters", "diagnostic_flags": [],
             "exact_checks": {"fixture": {"status": "passed", "violations": 0}},
             "data_sha256": freeze.file_sha256(data), "configuration_hashes": freeze.dataset_identity(config, lock),
             "borrowers": 3, "event_rows": 9, "phase_counts": {"background": 6, "index_loan": 3},
             "activity": {"mean": 2, "n": 3}}
    report = tmp_path / freeze.AUDIT_RELATIVE_PATH
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(audit))
    return data, config, report


def test_records_user_acceptance_and_verifies_exact_bytes(sample):
    data, _, _ = sample
    accepted = freeze.accept_dataset()
    assert accepted["accepted_by"] == "user"
    assert accepted["dataset"]["sha256"] == freeze.file_sha256(data)
    assert set(accepted["inputs"]) == {"contract", "ledger", "mapping"}
    assert freeze.verify_accepted_dataset(data) == accepted


def test_same_size_data_mutation_is_rejected_before_csv_loading(sample, monkeypatch):
    data, config, _ = sample
    freeze.accept_dataset()
    before = data.read_bytes()
    data.write_bytes(b"X" + before[1:])
    import dataset_io

    monkeypatch.setattr(dataset_io, "DATA_FILE", data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset_io.load(data, config)


def test_truncated_frozen_file_is_rejected(sample):
    data, _, _ = sample
    freeze.accept_dataset()
    data.write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        freeze.verify_accepted_dataset(data)


def test_changed_parameters_are_rejected_even_after_refreshing_input_lock(sample):
    data, config, _ = sample
    freeze.accept_dataset()
    config["dataset"]["borrowers"] = 4
    (freeze.ROOT / freeze.INPUTS["contract"]).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Frozen input hash mismatch"):
        freeze.verify_accepted_dataset(data)
    freeze.write_lock(force=True)
    with pytest.raises(ValueError, match="Accepted parameter freeze mismatch"):
        freeze.verify_accepted_dataset(data)


def test_changed_generator_is_rejected(sample):
    data, _, _ = sample
    freeze.accept_dataset()
    (freeze.ROOT / freeze.GENERATOR_SOURCES[0]).write_text("# different generator\n")
    with pytest.raises(ValueError, match="generator source mismatch"):
        freeze.verify_accepted_dataset(data)


def test_acceptance_cannot_be_silently_replaced(sample):
    freeze.accept_dataset()
    with pytest.raises(FileExistsError, match="already exists"):
        freeze.accept_dataset()


def test_missing_acceptance_is_rejected_before_csv_loading(sample, monkeypatch):
    data, config, _ = sample
    import dataset_io

    monkeypatch.setattr(dataset_io, "DATA_FILE", data)
    with pytest.raises(FileNotFoundError, match="Missing accepted dataset freeze"):
        dataset_io.load(data, config)


def test_failed_audit_cannot_be_accepted(sample):
    _, _, report = sample
    audit = json.loads(report.read_text())
    audit["exact_checks"]["fixture"]["violations"] = 1
    report.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="passing sample parameter audit"):
        freeze.accept_dataset()
    assert not freeze.ACCEPTED_PATH.exists()


def test_stale_audit_cannot_freeze_a_different_dataset(sample):
    data, _, _ = sample
    data.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="audit does not match"):
        freeze.accept_dataset()
    assert not freeze.ACCEPTED_PATH.exists()


def test_existing_accepted_build_verifies_hashes_without_rebuilding_features(sample, monkeypatch):
    data, config, _ = sample
    freeze.accept_dataset()
    import dataset_io

    monkeypatch.setattr(dataset_io, "DATA_FILE", data)
    monkeypatch.setattr(dataset_io, "ACCEPTED_PATH", freeze.ACCEPTED_PATH)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Accepted bytes should not need another feature reconstruction")

    monkeypatch.setattr(dataset_io, "load", unexpected_load)
    assert dataset_io.build(config, data) == data
