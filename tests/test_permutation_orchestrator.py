"""Check the orchestrator plan matches the fixed experiment plan exactly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/colab"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/scripts"))

from permutation_one_by_one import PLAN
from repeated_permutation import conditions


def test_orchestrator_plan_matches_experiment():
    assert len(PLAN) == 22
    assert PLAN == [item[0] for item in conditions()]
