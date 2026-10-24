"""Tests of actual v2 generation/preprocessing, not legacy reference helpers."""

from copy import deepcopy
import unittest

import numpy as np

from experiment_v2 import features, generate, load_contract, partitions, preprocess


class ImplementationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()
        cls.data = generate(cls.contract, "smoke", 42, "no_signal")
        cls.split = partitions(cls.data["ids"], cls.data["labels"], 42)

    def test_actual_preprocessing_is_invariant_to_heldout_values_and_labels(self):
        original = preprocess(self.data, self.split, 32)
        changed = deepcopy(self.data)
        heldout = np.r_[self.split["validation"], self.split["test"], self.split["early_stop"]]
        changed["static"][heldout] *= 1000000
        changed["labels"][heldout] = 2
        for i in heldout:
            changed["sequences"][i][:, [0, 2, 3]] *= 1000000
        transformed = preprocess(changed, self.split, 32)
        fit = self.split["fit"]
        for name in ["static", "sequence"]:
            np.testing.assert_array_equal(original[name][fit], transformed[name][fit])
            np.testing.assert_array_equal(original["preprocessors"][name].mean_, transformed["preprocessors"][name].mean_)

    def test_history_independent_of_outcome_randomness_and_order_static_invariant(self):
        contract = deepcopy(self.contract)
        contract["profiles"]["smoke"]["borrowers"] = 40
        base = generate(contract, "smoke", 42, "no_signal")
        changed = generate(contract, "smoke", 42, "no_signal", outcome_seed=999)
        ordered = generate(contract, "smoke", 42, "order_signal")
        np.testing.assert_array_equal(base["static"], changed["static"])
        self.assertFalse(np.array_equal(base["labels"], changed["labels"]))
        np.testing.assert_allclose(base["static"], ordered["static"], rtol=1e-12, atol=1e-9)
        self.assertEqual(ordered["order_invariance_checked"], 40)
        for a, b in zip(base["sequences"], changed["sequences"]):
            np.testing.assert_array_equal(a, b)

    def test_partition_roles_and_alignment(self):
        self.assertEqual([len(self.split[name]) for name in ["train", "validation", "test"]], [600, 200, 200])
        self.assertFalse(set(self.split["fit"]) & set(self.split["early_stop"]))
        self.assertEqual(set(self.split["fit"]) | set(self.split["early_stop"]), set(self.split["train"]))
        self.assertEqual(len(self.data["ids"]), len(set(self.data["ids"])))

    def test_explicit_padding_and_binary_schema(self):
        data = deepcopy(self.data)
        data["sequences"][0] = data["sequences"][0][-5:]
        transformed = preprocess(data, self.split, 32)
        self.assertEqual(transformed["lengths"][0], 5)
        self.assertFalse(transformed["mask"][0, :27].any())
        np.testing.assert_array_equal(transformed["mask"].sum(axis=1), transformed["lengths"])
        self.assertTrue(np.all(transformed["sequence"][~transformed["mask"]] == 0))
        for i, length in enumerate(transformed["lengths"]):
            np.testing.assert_array_equal(transformed["sequence"][i, -length:, 4:], data["sequences"][i][:, 4:])

    def test_memory_probe_reports_positive_finite_usage(self):
        from run_smoke import peak_memory_gib
        peak = peak_memory_gib()
        self.assertTrue(np.isfinite(peak))
        self.assertGreater(peak, 0)

    def test_pilot_uses_contract_length_and_epoch_budget(self):
        from run_smoke import execution_settings
        settings = execution_settings(self.contract, "pilot")
        self.assertEqual((settings["borrowers"], settings["sequence_length"], settings["epochs"]), (5000, 64, 20))
        settings = execution_settings(self.contract, "main")
        self.assertEqual((settings["borrowers"], settings["sequence_length"], settings["epochs"]), (25625, 64, 20))
        self.assertEqual(execution_settings(self.contract, "smoke")["epochs"], 2)
        with self.assertRaises(ValueError):
            execution_settings(self.contract, "unknown")

    def test_actual_features_ignore_index_and_future_events(self):
        category = self.contract["features"]["categories"][0]
        row = dict(phase="background", day=-1, amount=5., is_incoming=1, category=category,
                   event_id="event", counterparty="counterparty")
        a, seq_a = features([row], self.contract["features"]["categories"], 32)
        b, seq_b = features([row, {**row, "phase": "index_loan", "day": 0, "amount": 1000000},
                              {**row, "phase": "principal_repayment", "day": 2, "amount": 1000000}],
                             self.contract["features"]["categories"], 32)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(seq_a, seq_b)

    def test_control_and_target_choices_are_explicit(self):
        from run_smoke import run
        with self.assertRaises(ValueError):
            run(self.__class__.contract, 42, "no_signal", target="invalid")


if __name__ == "__main__":
    unittest.main()
