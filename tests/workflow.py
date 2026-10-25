"""Small integration workloads isolated from the main data and run outputs."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/scripts"))

from competition import run
from dataset_io import build
from experiment_v2 import load_contract
from validation import validate


def check(borrowers: int, epochs: int, controls: bool = False) -> dict:
    contract = deepcopy(load_contract())
    contract["dataset"] = {"borrowers": borrowers}
    contract["execution"]["early_stopping"]["epoch_cap"] = epochs
    with tempfile.TemporaryDirectory(prefix="yara-test-") as directory:
        root = Path(directory)
        data = root / "yara_telecel_dump.csv.gz"
        build(contract, data)
        output = root / "results"
        output.mkdir()
        scores = run(output, contract=contract, data_path=data)
        report = validate(output)
        assert report["calibration"]["status"] == "passed"
        if controls:
            for control in ("label_permutation", "order_shuffle"):
                control_output = root / control
                control_output.mkdir()
                control_scores = run(control_output, contract=contract, data_path=data, control=control)
                validate(control_output)
                assert not (control_output / "calibration").exists()
                print(control, {name: values["roc_auc"] for name, values in control_scores["validation"].items()})
        print(f"Validated {borrowers:,} borrowers: {report['status']}")
        print({model: values["roc_auc"] for model, values in scores["validation"].items()})
        return report
