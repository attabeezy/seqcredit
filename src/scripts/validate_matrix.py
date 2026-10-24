"""Independently validate every run listed in a matrix execution manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_smoke import ROOT, write_json
from validate_pilots import validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text(encoding="utf-8"))
    if execution.get("status") != "passed":
        raise ValueError("Require a completed matrix execution")
    results = []
    for item in execution["runs"]:
        run = Path(item["run"]).resolve()
        result = validate(run)
        for key in ("profile", "scenario", "seed", "target", "control"):
            if result[key] != item[key]:
                raise ValueError(f"Matrix metadata mismatch for {run}: {key}")
        results.append(result)
    report = {
        "status": "passed",
        "scope": "independent matrix artifact validation",
        "execution": args.execution.name,
        "runs": results,
    }
    output = ROOT / "docs" / "matrix_validation.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
