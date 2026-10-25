"""Tests of actual v2 generation/preprocessing, not legacy reference helpers."""

from copy import deepcopy
import unittest

import numpy as np

from experiment_v2 import features, generate, load_contract, partitions, preprocess, sample_event_count, sequence_batch


class ImplementationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()
        cls.contract["dataset"] = {"borrowers": 1000}
        cls.data = generate(cls.contract)
        cls.split = partitions(cls.data["ids"], cls.data["labels"], 42)

    def test_actual_preprocessing_is_invariant_to_heldout_values_and_labels(self):
        original = preprocess(self.data, self.split)
        changed = deepcopy(self.data)
        heldout = np.r_[self.split["validation"], self.split["test"], self.split["early_stop"]]
        changed["static"][heldout] *= 1000000
        changed["labels"][heldout] = 2
        for i in heldout:
            changed["sequences"][i][:, [0, 2, 3]] *= 1000000
        transformed = preprocess(changed, self.split)
        fit = self.split["fit"]
        for name in ["static", "sequence"]:
            if name == "static":
                np.testing.assert_array_equal(original[name][fit], transformed[name][fit])
            else:
                np.testing.assert_array_equal(sequence_batch(original, fit)["sequence"], sequence_batch(transformed, fit)["sequence"])
            np.testing.assert_array_equal(original["preprocessors"][name].mean_, transformed["preprocessors"][name].mean_)

    def test_history_independent_of_outcome_randomness(self):
        contract = deepcopy(self.contract)
        contract["dataset"]["borrowers"] = 40
        base = generate(contract)
        changed = generate(contract, outcome_seed=999)
        np.testing.assert_array_equal(base["static"], changed["static"])
        self.assertFalse(np.array_equal(base["labels"], changed["labels"]))
        for a, b in zip(base["sequences"], changed["sequences"]):
            np.testing.assert_array_equal(a, b)

    def test_history_is_noisy_rather_than_a_deterministic_order_code(self):
        candidates = [sequence for sequence in self.data["sequences"] if len(sequence) >= 8]
        noisy = 0
        for sequence in candidates:
            changes = np.diff(sequence[:, 0])
            noisy += bool(np.any(changes > 0) and np.any(changes < 0))
        self.assertGreater(noisy / len(candidates), 0.98)

    def test_activity_varies_and_is_reproducible(self):
        settings = self.contract["background_events"]
        counts = np.array([sample_event_count(settings, borrower=i) for i in range(1000)])
        np.testing.assert_array_equal(counts, [sample_event_count(settings, borrower=i) for i in range(1000)])
        self.assertTrue(np.all(counts >= 2))
        mean = settings["expected_per_borrower"]
        self.assertGreater(np.mean(counts < mean), 0.2)
        self.assertGreater(np.mean(counts >= mean), 0.2)
        self.assertGreater(np.mean(counts < 64), 0.05)
        self.assertAlmostEqual(float(counts.mean()), mean, delta=mean * 0.1)
        self.assertGreater(float(counts.std()), mean * 0.4)

    def test_activity_changes_do_not_change_outcomes(self):
        contract = deepcopy(self.contract)
        contract["dataset"]["borrowers"] = 40
        base = generate(contract)
        contract["background_events"]["count_distribution"]["activity_shape"] = 0.5
        changed = generate(contract)
        np.testing.assert_array_equal(base["labels"], changed["labels"])
        self.assertNotEqual(base["event_count"], changed["event_count"])

    def test_model_histories_really_have_variable_lengths(self):
        transformed = preprocess(self.data, self.split)
        self.assertTrue(np.any(transformed["lengths"] < 150))
        self.assertTrue(np.any(transformed["lengths"] > 150))
        np.testing.assert_array_equal(transformed["lengths"], self.data["static"][:, 0])
        self.assertEqual(int(transformed["lengths"].sum()), len(transformed["sequence"]))
        np.testing.assert_array_equal(np.diff(transformed["offsets"]), transformed["lengths"])

    def test_partition_roles_and_alignment(self):
        self.assertEqual([len(self.split[name]) for name in ["train", "validation", "test"]], [600, 200, 200])
        self.assertFalse(set(self.split["fit"]) & set(self.split["early_stop"]))
        self.assertEqual(set(self.split["fit"]) | set(self.split["early_stop"]), set(self.split["train"]))
        self.assertEqual(len(self.data["ids"]), len(set(self.data["ids"])))

    def test_explicit_padding_and_binary_schema(self):
        data = deepcopy(self.data)
        data["sequences"][0] = data["sequences"][0][-5:]
        transformed = preprocess(data, self.split)
        self.assertEqual(transformed["lengths"][0], 5)
        other = int(np.argmax(transformed["lengths"]))
        batch = sequence_batch(transformed, np.array([0, other]))
        self.assertTrue(batch["valid_mask"][0, :5].all())
        self.assertFalse(batch["valid_mask"][0, 5:].any())
        self.assertTrue(np.all(batch["sequence"][~batch["valid_mask"]] == 0))
        self.assertEqual(batch["sequence"].shape[1], transformed["lengths"][other])
        for i, length in enumerate(transformed["lengths"]):
            start, end = transformed["offsets"][i:i + 2]
            np.testing.assert_array_equal(transformed["sequence"][start:end, 4:], data["sequences"][i][:, 4:])

    def test_zero_valued_real_events_remain_unmasked(self):
        packed = {"sequence": np.zeros((3, 12), dtype=np.float32),
                  "lengths": np.array([1, 2]), "offsets": np.array([0, 1, 3])}
        batch = sequence_batch(packed, np.array([0, 1]))
        np.testing.assert_array_equal(batch["valid_mask"], [[True, False], [True, True]])

    def test_memory_probe_reports_positive_finite_usage(self):
        from competition import peak_memory_gib
        peak = peak_memory_gib()
        self.assertTrue(np.isfinite(peak))
        self.assertGreater(peak, 0)

    def test_execution_uses_one_dataset_and_fixed_training_budget(self):
        from competition import execution_settings
        settings = execution_settings(load_contract())
        self.assertEqual((settings["borrowers"], settings["history"], settings["epochs"]), (25625, "all eligible events", 20))

    def test_actual_features_ignore_index_and_future_events(self):
        category = self.contract["features"]["categories"][0]
        row = dict(phase="background", day=-1, amount=5., is_incoming=1, category=category,
                   event_id="event", counterparty="counterparty")
        a, seq_a = features([row], self.contract["features"]["categories"])
        b, seq_b = features([row, {**row, "phase": "index_loan", "day": 0, "amount": 1000000},
                              {**row, "phase": "principal_repayment", "day": 2, "amount": 1000000}],
                             self.contract["features"]["categories"])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(seq_a, seq_b)

    def test_control_and_target_choices_are_explicit(self):
        from competition import run
        with self.assertRaises(ValueError):
            run(None, target="invalid")


if __name__ == "__main__":
    unittest.main()
