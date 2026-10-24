"""Run or materialize the preregistered v2 experiment matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

from run_smoke import ROOT, write_json


SCENARIOS = ("no_signal", "aggregate_signal", "order_signal")
SEEDS = (42, 43, 44)
TARGETS = ("primary", "secondary")
MODELS = ("logistic_regression", "lightgbm", "gru")


def matrix() -> list[dict[str, object]]:
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for target in TARGETS:
                rows.append({
                    "profile": "main",
                    "scenario": scenario,
                    "seed": seed,
                    "target": target,
                    "control": "none",
                    "models": list(MODELS),
                })
    for scenario in SCENARIOS:
        for seed in SEEDS:
            rows.append({
                "profile": "main",
                "scenario": scenario,
                "seed": seed,
                "target": "primary",
                "control": "label_permutation",
                "models": list(MODELS),
            })
    for seed in SEEDS:
        rows.append({
            "profile": "main",
            "scenario": "order_signal",
            "seed": seed,
            "target": "primary",
            "control": "order_shuffle",
            "models": list(MODELS),
        })
    return rows


def write_plan(path: Path) -> list[dict[str, object]]:
    plan = matrix()
    write_json(path, {
        "status": "planned",
        "contract": "seqcredit-default-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": plan,
        "counts": {
            "headline_runs": 18,
            "control_runs": 12,
            "total_runs": len(plan),
        },
        "test_policy": "No test predictions are read by this planner for selection.",
    })
    return plan


def execute(plan: list[dict[str, object]], python: Path) -> list[dict[str, object]]:
    results = []
    for number, item in enumerate(plan, start=1):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = ROOT / "data/runs" / f"matrix-{number:03d}-{stamp}-{uuid.uuid4().hex[:8]}"
        command = [
            str(python), str(ROOT / "src/scripts/run_smoke.py"),
            "--profile", str(item["profile"]),
            "--scenario", str(item["scenario"]),
            "--seed", str(item["seed"]),
            "--target", str(item["target"]),
            "--control", str(item["control"]),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        output_lines = [line for line in completed.stdout.splitlines() if line.startswith("PASS: ")]
        if completed.returncode == 0 and len(output_lines) != 1:
            raise RuntimeError(f"Matrix run {number} did not report an artifact path")
        result = {**item, "run_number": number, "returncode": completed.returncode,
                  "command": command, "run": output_lines[0][len("PASS: "):] if output_lines else "",
                  "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
        results.append(result)
        if completed.returncode:
            raise RuntimeError(f"Matrix run {number} failed: {item}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "data/matrix_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    plan = write_plan(args.plan)
    if args.execute:
        results = execute(plan, args.python)
        write_json(args.plan.with_name("matrix_execution.json"), {
            "status": "passed",
            "plan": args.plan.name,
            "runs": results,
        })
    else:
        print(json.dumps({"status": "planned", "runs": len(plan), "plan": str(args.plan)}, indent=2))


if __name__ == "__main__":
    main()
