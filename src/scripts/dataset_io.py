"""One compressed event table; labels, splits and features are derived on load."""

from __future__ import annotations

import csv
import gzip
import io
from itertools import groupby
import os
from pathlib import Path
import uuid

import numpy as np

from contract_checks import derive_label, validate_events
from experiment_v2 import ROOT, features, generate, load_contract, partitions, prevalence_intercept
from input_freeze import ACCEPTED_PATH, dataset_identity, file_sha256, verify, verify_accepted_dataset


DATA_FILE = ROOT / "data/yara_telecel_dump.csv.gz"
EVENT_COLUMNS = ["borrower_id", "event_id", "day", "phase", "amount", "loan_id",
                 "category", "raw_type", "is_incoming", "counterparty"]
PROVENANCE_COLUMNS = ["contract_hash", "input_hash", "generator_hash"]


def identity(contract: dict) -> dict[str, str]:
    return dataset_identity(contract, verify())


def build(contract: dict | None = None, path: Path = DATA_FILE) -> Path:
    """Atomically build a missing file, or validate and reuse an existing one."""
    contract = load_contract() if contract is None else contract
    expected = identity(contract)
    if path.exists():
        if path.resolve() == DATA_FILE.resolve() and ACCEPTED_PATH.exists():
            accepted = verify_accepted_dataset(path)
            if expected != accepted["dataset_identity"]:
                raise ValueError("Requested parameters disagree with the accepted dataset")
        else:
            load(path, contract)
        print(f"Ready: {path}", flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building {contract['dataset']['borrowers']:,} borrowers into {path}", flush=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as binary:
            with gzip.GzipFile(filename="", fileobj=binary, mode="wb", compresslevel=6, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS + PROVENANCE_COLUMNS,
                                            lineterminator="\n")
                    writer.writeheader()
                    completed = 0

                    def save(events: list[dict]) -> None:
                        nonlocal completed
                        writer.writerows({**event, **expected} for event in events)
                        completed += 1
                        if completed % 5000 == 0:
                            print(f"Generated {completed:,}/{contract['dataset']['borrowers']:,} borrowers", flush=True)

                    generated = generate(contract, event_sink=save)
        # Read the actual serialized file before committing it.
        print("Checking the saved event file before making it available", flush=True)
        restored, _, _ = load(temporary, contract)
        for key in ("ids", "labels", "static"):
            np.testing.assert_array_equal(generated[key], restored[key])
        for before, after in zip(generated["sequences"], restored["sequences"]):
            np.testing.assert_array_equal(before, after)
        if path.resolve() == DATA_FILE.resolve() and ACCEPTED_PATH.exists():
            verify_accepted_dataset(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Built: {path} ({len(generated['ids']):,} borrowers)", flush=True)
    counts = generated["static"][:, 0]
    print(f"Background events per borrower: mean {counts.mean():.2f}, "
          f"range {int(counts.min())}-{int(counts.max())}; all retained", flush=True)
    return path


def load(path: Path = DATA_FILE, contract: dict | None = None) -> tuple[dict, dict, dict]:
    """Validate provenance and lifecycle rows; reconstruct all model inputs."""
    contract = load_contract() if contract is None else contract
    expected = identity(contract)
    accepted = verify_accepted_dataset(path) if path.resolve() == DATA_FILE.resolve() else None
    if accepted is not None and expected != accepted["dataset_identity"]:
        raise ValueError("Requested parameters disagree with the accepted dataset")
    categories = contract["features"]["categories"]
    ids, labels, static, sequences = [], [], [], []
    seen = set()
    count = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EVENT_COLUMNS + PROVENANCE_COLUMNS:
            raise ValueError("Dataset columns do not match the current design; rebuild data/yara_telecel_dump.csv.gz")
        for uid, group in groupby(reader, key=lambda row: row["borrower_id"]):
            events = []
            if uid in seen:
                raise ValueError("Dataset borrower rows must be contiguous and unique")
            seen.add(uid)
            for row in group:
                if any(row[key] != value for key, value in expected.items()):
                    raise ValueError("Dataset provenance mismatch; remove the stale file and run dataset.py")
                events.append({key: (float(row[key]) if key in {"amount", "day"} else
                                     int(row[key]) if key == "is_incoming" else row[key])
                               for key in EVENT_COLUMNS})
            validate_events(events, [uid])
            loan = next(event["loan_id"] for event in events if event["phase"] == "index_loan")
            aggregate, sequence = features(events, categories)
            ids.append(uid)
            labels.append(derive_label(events, uid, loan))
            static.append(aggregate)
            sequences.append(sequence)
            count += len(events)
    if len(ids) != contract["dataset"]["borrowers"]:
        raise ValueError("Dataset borrower count mismatch")
    if ids != [f"SIM_{i:06d}" for i in range(len(ids))]:
        raise ValueError("Dataset borrower order does not match generated population")
    counts = contract["outcome"]["reference_counts"]
    _, calibration = prevalence_intercept(counts["no_repayment"] / contract["outcome"]["reference_denominator"],
                                          contract["history_process"]["risk_beta"])
    data = dict(ids=np.array(ids), labels=np.array(labels, dtype=np.int32), static=np.stack(static),
                sequences=sequences, event_count=count, calibration=calibration)
    split = partitions(data["ids"], data["labels"])
    reference = {"path": str(path.resolve()), "sha256": file_sha256(path), **expected,
                 "borrowers": len(ids), "events": count,
                 "history_lengths": {"min": int(data["static"][:, 0].min()),
                                     "median": float(np.median(data["static"][:, 0])),
                                     "mean": float(data["static"][:, 0].mean()),
                                     "max": int(data["static"][:, 0].max()),
                                     "total": int(data["static"][:, 0].sum())}}
    if accepted is not None:
        reference["accepted_snapshot_sha256"] = file_sha256(ACCEPTED_PATH)
    return data, split, reference
