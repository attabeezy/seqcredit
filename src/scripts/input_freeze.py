"""Freeze and verify the immutable calibration inputs used by v2 generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "src/configs/frozen_inputs.json"
ACCEPTED_PATH = ROOT / "src/configs/accepted_dataset.json"
DATASET_RELATIVE_PATH = "data/yara_telecel_dump.csv.gz"
AUDIT_RELATIVE_PATH = "artifacts/sample_validation.json"
GENERATOR_SOURCES = ("src/scripts/experiment_v2.py", "src/scripts/contract_checks.py")
INPUTS = {
    "contract": "src/configs/experiment_design.json",
    "ledger": "docs/calibration/ledger.json",
    "mapping": "docs/calibration/category_mapping.json",
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def current_hashes() -> dict[str, str]:
    return {name: file_sha256(ROOT / relative) for name, relative in INPUTS.items()}


def write_lock(force: bool = False) -> dict:
    if LOCK_PATH.exists() and not force:
        raise FileExistsError(f"{LOCK_PATH} already exists; use --force to replace it")
    hashes = current_hashes()
    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": relative, "sha256": hashes[name]}
            for name, relative in INPUTS.items()
        },
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_lock() -> dict:
    if not LOCK_PATH.exists():
        raise FileNotFoundError(f"Missing frozen input lock: {LOCK_PATH}")
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify() -> dict:
    lock = load_lock()
    expected = {name: item["sha256"] for name, item in lock["inputs"].items()}
    actual = current_hashes()
    if set(expected) != set(actual):
        raise ValueError("Frozen input set mismatch")
    changed = [name for name in actual if actual[name] != expected[name]]
    if changed:
        raise ValueError(f"Frozen input hash mismatch: {', '.join(changed)}")
    for name, relative in INPUTS.items():
        if lock["inputs"][name]["path"] != relative:
            raise ValueError(f"Frozen input path mismatch: {name}")
    return lock


def generator_hashes() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in GENERATOR_SOURCES}


def dataset_identity(contract: dict, lock: dict) -> dict[str, str]:
    from contract_checks import fingerprint

    return {"contract_hash": fingerprint(contract), "input_hash": fingerprint(lock["inputs"]),
            "generator_hash": fingerprint({Path(name).name: digest for name, digest in generator_hashes().items()})}


def accept_dataset(force: bool = False) -> dict:
    """Record explicit user acceptance of the already audited parameter/data bytes."""
    if ACCEPTED_PATH.exists() and not force:
        raise FileExistsError("An accepted dataset already exists; do not overwrite its freeze implicitly")
    lock = verify()
    contract = json.loads((ROOT / INPUTS["contract"]).read_text(encoding="utf-8"))
    audit_path = ROOT / AUDIT_RELATIVE_PATH
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    path = ROOT / DATASET_RELATIVE_PATH
    digest = file_sha256(path)
    expected_identity = dataset_identity(contract, lock)
    if (audit.get("status") != "consistent_with_configured_parameters" or
            audit.get("diagnostic_flags") or not audit.get("exact_checks") or
            any(check.get("status") != "passed" or check.get("violations") != 0
                for check in audit["exact_checks"].values())):
        raise ValueError("Acceptance requires the passing sample parameter audit")
    if (audit.get("data_sha256") != digest or audit.get("configuration_hashes") != expected_identity or
            audit.get("borrowers") != contract["dataset"]["borrowers"]):
        raise ValueError("Sample audit does not match the current parameters and dataset")
    payload = {
        "version": 1, "status": "accepted_frozen",
        "accepted_at": datetime.now(timezone.utc).isoformat(), "accepted_by": "user",
        "acceptance": "User accepted the sample against the parameters and requested freezing both before full training.",
        "seed": contract["seed"], "inputs": lock["inputs"],
        "frozen_inputs_sha256": file_sha256(LOCK_PATH), "generator_sources": generator_hashes(),
        "dataset_identity": expected_identity,
        "dataset": {"path": DATASET_RELATIVE_PATH, "sha256": digest, "bytes": path.stat().st_size,
                    "borrowers": audit["borrowers"], "event_rows": audit["event_rows"],
                    "phase_counts": audit["phase_counts"],
                    "history_lengths": audit["activity"]},
        "audit": {"path": AUDIT_RELATIVE_PATH, "sha256": file_sha256(audit_path),
                  "status": audit["status"], "exact_checks_passed": len(audit["exact_checks"]),
                  "diagnostic_flags": []},
        "policy": "Dataset and parameter bytes are fixed for the competition. Changes require explicit new user acceptance and a new freeze; never select generator changes from test scores.",
    }
    temporary = ACCEPTED_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(ACCEPTED_PATH)
    verify_accepted_dataset(path)
    return payload


def verify_accepted_dataset(path: Path) -> dict:
    """Fail closed if the accepted parameters, generator, or data have changed."""
    if not ACCEPTED_PATH.is_file():
        raise FileNotFoundError("Missing accepted dataset freeze; explicit user acceptance is required")
    accepted = json.loads(ACCEPTED_PATH.read_text(encoding="utf-8"))
    lock = verify()
    if (accepted.get("version") != 1 or accepted.get("status") != "accepted_frozen" or
            accepted.get("accepted_by") != "user"):
        raise ValueError("Invalid accepted dataset freeze")
    if (accepted.get("inputs") != lock["inputs"] or
            accepted.get("frozen_inputs_sha256") != file_sha256(LOCK_PATH)):
        raise ValueError("Accepted parameter freeze mismatch; explicit new acceptance is required")
    if accepted.get("generator_sources") != generator_hashes():
        raise ValueError("Accepted generator source mismatch; explicit new acceptance is required")
    contract = json.loads((ROOT / INPUTS["contract"]).read_text(encoding="utf-8"))
    if (accepted.get("dataset_identity") != dataset_identity(contract, lock) or
            accepted.get("seed") != contract["seed"] or
            accepted["dataset"].get("borrowers") != contract["dataset"]["borrowers"] or
            accepted["dataset"].get("path") != DATASET_RELATIVE_PATH):
        raise ValueError("Accepted dataset identity mismatch")
    if path.stat().st_size != accepted["dataset"]["bytes"]:
        raise ValueError("Accepted dataset size mismatch; do not overwrite the frozen sample")
    if file_sha256(path) != accepted["dataset"]["sha256"]:
        raise ValueError("Accepted dataset checksum mismatch; do not overwrite the frozen sample")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--accept-dataset", action="store_true")
    args = parser.parse_args()
    if args.accept_dataset:
        print(json.dumps(accept_dataset(args.force), indent=2))
    elif args.write:
        payload = write_lock(args.force)
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
