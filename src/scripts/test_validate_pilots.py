"""Adversarial checks for independent cross-scenario validation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from validate_pilots import compare_order_inputs, expected_split_sizes, target_values
from bootstrap_predictions import bootstrap
from run_matrix import matrix


class CrossScenarioChecks(unittest.TestCase):
    def test_profile_sizes_and_targets_are_contract_driven(self):
        contract = {"profiles": {"smoke": {"borrowers": 1000}, "main": {"borrowers": 25625}}}
        self.assertEqual(expected_split_sizes("smoke", contract), [600, 200, 200])
        self.assertEqual(expected_split_sizes("main", contract), [15375, 5125, 5125])
        np.testing.assert_array_equal(target_values("primary", np.array([0, 1, 2])), [0, 0, 1])
        np.testing.assert_array_equal(target_values("secondary", np.array([0, 1, 2])), [0, 1, 1])

    def test_matrix_is_preregistered_and_complete(self):
        plan = matrix()
        self.assertEqual(len(plan), 30)
        self.assertEqual(sum(row["control"] == "none" for row in plan), 18)
        self.assertEqual(sum(row["control"] != "none" for row in plan), 12)
        self.assertTrue(all(row["profile"] == "main" for row in plan))

    def test_permuted_payload_passes_but_changed_static_or_payload_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            base, ordered = Path(directory) / "base", Path(directory) / "ordered"
            base.mkdir()
            ordered.mkdir()
            sequence = np.zeros((1, 3, 12), dtype=np.float32)
            sequence[0, :, 0] = [1, 2, 3]
            sequence[0, :, 3] = [-3, -2, -1]
            ids, lengths, static = np.array(["SIM_1"]), np.array([3]), np.array([[6.]])
            np.savez(base / "raw_features.npz", ids=ids, lengths=lengths, static=static, sequence=sequence)
            permuted = sequence.copy()
            permuted[0, :, 0] = [3, 2, 1]

            def save(values, aggregate):
                np.savez(ordered / "raw_features.npz", ids=ids, lengths=lengths, static=aggregate, sequence=values)

            save(permuted, static)
            self.assertEqual(compare_order_inputs(base, ordered)["borrowers_with_changed_order"], 1)
            save(permuted, static + 1)
            with self.assertRaises(AssertionError):
                compare_order_inputs(base, ordered)
            permuted[0, 0, 0] += 10
            save(permuted, static)
            with self.assertRaisesRegex(ValueError, "payload multiset"):
                compare_order_inputs(base, ordered)

    def test_bootstrap_uses_locked_test_rows_and_reports_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with (run / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                handle.write("borrower_id,partition,target,logistic_regression,lightgbm,gru\n")
                for i in range(8):
                    target = int(i == 0)
                    values = [0.8, 0.7, 0.6] if target else [0.1, 0.2, 0.3]
                    handle.write(f"u{i},test,{target},{values[0]},{values[1]},{values[2]}\n")
            report = bootstrap(run, draws=20, seed=42)
            self.assertEqual(report["draws_requested"], 20)
            self.assertEqual(report["draws_used"] + report["single_class_draws_skipped"], 20)
            self.assertEqual(report["intervals"]["gru"]["average_precision"]["replicates"], report["draws_used"])


if __name__ == "__main__":
    unittest.main()
