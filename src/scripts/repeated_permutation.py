"""Run the fixed exploratory permutation plan on the accepted dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from importlib.metadata import version

from competition import file_hash, run, write_json
from dataset_io import DATA_FILE
from experiment_v2 import ROOT
from input_freeze import ACCEPTED_PATH, INPUTS, LOCK_PATH, verify_accepted_dataset
from validation import validate


SEEDS = tuple(range(1051, 1061))
PLAN = ROOT / "docs/PERMUTATION_PLAN.md"
EXPECTED_VERSIONS = {"numpy": "2.4.6", "scipy": "1.18.0", "scikit-learn": "1.8.0",
                     "lightgbm": "4.6.0", "tensorflow": "2.21.0", "keras": "3.15.0",
                     "joblib": "1.5.3"}


def conditions() -> list[tuple[str, int | None, int | None]]:
    """Return the complete fixed set of headline and permutation conditions."""
    return [("headline_original", None, None), ("headline_one_epoch", None, 1)] + [
        (f"permutation_{seed}_{policy}", seed, epochs)
        for seed in SEEDS for policy, epochs in (("original", None), ("one_epoch", 1))
    ]


def condition_by_name(name: str) -> tuple[int | None, int | None]:
    """Return fixed (permutation seed, fixed GRU epochs) for one condition."""
    lookup = {item[0]: (item[1], item[2]) for item in conditions()}
    try:
        return lookup[name]
    except KeyError:
        raise ValueError(f"Unknown condition: {name}") from None


def select_conditions(only: list[str] | None) -> list[tuple[str, int | None, int | None]]:
    """Return the planned subset in plan order; None selects the full plan."""
    plan = conditions()
    if only is None:
        return plan
    if not only or not all(isinstance(name, str) for name in only):
        raise ValueError("Batch conditions must be a non-empty list of names")
    if len(set(only)) != len(only):
        raise ValueError("Batch conditions must be unique")
    names = {item[0] for item in plan}
    unknown = [name for name in only if name not in names]
    if unknown:
        raise ValueError(f"Unknown condition: {unknown[0]}")
    wanted = set(only)
    return [item for item in plan if item[0] in wanted]


def parse_only(value: str) -> list[str]:
    """Parse the comma-separated --only argument."""
    return [name.strip() for name in value.split(",") if name.strip()]


def summarize(results: dict) -> dict:
    """Summarize every completed seed without selecting favorable fits."""
    if set(results) != {item[0] for item in conditions()}:
        raise ValueError("Cannot summarize an incomplete experiment")
    summary = {}
    for metric in ("average_precision", "roc_auc"):
        data = []
        for seed in SEEDS:
            original = results[f"permutation_{seed}_original"]["test"]
            fixed = results[f"permutation_{seed}_one_epoch"]["test"]
            data.append({"seed": seed, "logistic_regression": original["logistic_regression"][metric],
                         "gru_original": original["gru"][metric], "gru_one_epoch": fixed["gru"][metric],
                         "gru_original_minus_one_epoch": original["gru"][metric] - fixed["gru"][metric]})
        summary[metric] = {"seeds": data, "reference": 0.5 if metric == "roc_auc" else
                           results["headline_original"]["test"]["gru"]["prevalence"],
                           "range": {name: [min(row[name] for row in data), max(row[name] for row in data)]
                                     for name in ("logistic_regression", "gru_original", "gru_one_epoch")},
                           "original_seed_rank_ascending": {
                               name: 1 + sum(row[name] < data[0][name] for row in data[1:])
                               for name in ("logistic_regression", "gru_original", "gru_one_epoch")}}
    return {"status": "passed", "scope": "exploratory; ten permutations, fixed initialization and inspected test set",
            "metrics": summary,
            "headlines": {key: results[key]["test"] for key in ("headline_original", "headline_one_epoch")}}


def run_one(name: str) -> dict:
    """Run one condition in a fresh process so the peak-memory budget is per-fit."""
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("The specified matched GPU backend is unavailable")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "T4" not in gpu or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("The plan requires a T4 and CUBLAS_WORKSPACE_CONFIG=:4096:8")
    versions = {item: version(item) for item in EXPECTED_VERSIONS}
    if platform.python_version() != "3.13.16" or versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"The original runtime versions are required: Python {platform.python_version()}, {versions}")
    verify_accepted_dataset(DATA_FILE)
    seed, epochs = condition_by_name(name)
    path = ROOT / "artifacts" / "repeated_permutation_20261107" / name
    # Parent pre-creates the directory; tolerate that so --single works standalone too.
    path.mkdir(parents=True, exist_ok=True)
    print(f"Running {name}", flush=True)
    results = run(path, control="none" if seed is None else "label_permutation",
                  permutation_seed=seed, fixed_gru_epochs=epochs)
    write_json(path / "validation.json", validate(path))
    return results


def execute(destination: Path, only: list[str] | None = None) -> None:
    """Execute all (or a batch of) conditions and retain failures rather than skipping seeds."""
    selected = select_conditions(only)
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("The specified matched GPU backend is unavailable")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "T4" not in gpu or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("The plan requires a T4 and CUBLAS_WORKSPACE_CONFIG=:4096:8")
    versions = {name: version(name) for name in EXPECTED_VERSIONS}
    if platform.python_version() != "3.13.16" or versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"The original runtime versions are required: Python {platform.python_version()}, {versions}")
    accepted = verify_accepted_dataset(DATA_FILE)
    if not PLAN.is_file():
        raise FileNotFoundError("Missing fixed exploratory plan")
    destination.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "src/scripts/competition.py", ROOT / "src/scripts/validation.py",
               ROOT / "src/scripts/dataset_io.py", ROOT / "src/scripts/input_freeze.py",
               ROOT / "src/scripts/calibration.py",
               ROOT / "src/scripts/experiment_v2.py", ROOT / "src/scripts/contract_checks.py",
               ROOT / "uv.lock", PLAN, ACCEPTED_PATH, LOCK_PATH, *[ROOT / p for p in INPUTS.values()]]
    source_hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources}
    execution = {"status": "running", "dataset_sha256": accepted["dataset"]["sha256"],
                 "source_hashes": source_hashes, "python": sys.version, "platform": platform.platform(),
                 "versions": versions,
                 "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
                 "gpu_model": gpu, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                 "batch_only": [item[0] for item in selected] if only is not None else None,
                 "conditions": {}}
    write_json(destination / "execution.json", execution)
    results = {}
    for name, seed, epochs in selected:
        path = destination / name
        path.mkdir()
        try:
            print(f"Running {name}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__)), "--single", name], check=True)
            results[name] = json.loads((path / "metrics.json").read_text())
            execution["conditions"][name] = {"status": "passed", "permutation_seed": seed,
                                              "fixed_gru_epochs": epochs}
        except Exception as error:
            execution["conditions"][name] = {"status": "failed", "error": str(error),
                                              "traceback": traceback.format_exc()}
            execution["status"] = "failed"
            write_json(destination / "execution.json", execution)
            raise
        write_json(destination / "execution.json", execution)
    verify_accepted_dataset(DATA_FILE)
    if source_hashes != {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources}:
        raise ValueError("Source or frozen input changed during execution")
    if len(selected) < len(conditions()):
        execution["status"] = "passed"
        write_json(destination / "execution.json", execution)
        return
    # Logistic regression must be identical for the matched stopping policies.
    for seed in SEEDS:
        first = results[f"permutation_{seed}_original"]["test"]["logistic_regression"]
        second = results[f"permutation_{seed}_one_epoch"]["test"]["logistic_regression"]
        if first != second:
            raise ValueError(f"Logistic regression differs between matched conditions: {seed}")
    write_json(destination / "summary.json", summarize(results))
    execution["status"] = "passed"
    write_json(destination / "execution.json", execution)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--single":
        run_one(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--only":
        execute(ROOT / "artifacts" / "repeated_permutation_20261107", parse_only(sys.argv[2]))
    elif len(sys.argv) == 1:
        execute(ROOT / "artifacts" / "repeated_permutation_20261107")
    else:
        raise SystemExit(f"Usage: {Path(__file__).name} [--single <condition>] [--only <condition,...>]")
