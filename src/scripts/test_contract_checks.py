"""Adversarial contract checks; run without importing the ML package."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from contract_checks import (derive_label, fingerprint, select_history,
                             validate_cache, validate_events,
                             validate_feature_schema, validate_partitions)


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.events = [dict(borrower_id="a", event_id=str(i), day=day, phase=phase,
                            amount=10, loan_id="loan-a") for i, (day, phase) in enumerate([
                                (-90, "background"), (-1, "background"), (0, "index_loan"),
                                (60, "principal_repayment")])]

    def test_cutoff_and_truncation_exclude_index_and_future(self):
        validate_events(self.events, ["a"])
        self.assertEqual([row["day"] for row in select_history(self.events, "a", 1)], [-1])
        self.assertEqual(len(select_history(self.events, "a", 64)), 2)

    def test_future_perturbation_cannot_change_history(self):
        before = fingerprint(select_history(self.events, "a", 64))
        changed = deepcopy(self.events)
        changed[-1]["amount"] = 1000000
        changed[-1]["phase"] = "penalty"
        self.assertEqual(before, fingerprint(select_history(changed, "a", 64)))
        self.assertNotEqual(derive_label(self.events, "a", "loan-a"), derive_label(changed, "a", "loan-a"))

    def test_labels_respect_horizon_and_loan_link(self):
        self.assertEqual(derive_label(self.events, "a", "loan-a"), 0)
        self.events[-1]["day"] = 60.01
        self.assertEqual(derive_label(self.events, "a", "loan-a"), 2)
        self.events[-1]["day"] = 5
        self.events[-1]["loan_id"] = "another-loan"
        with self.assertRaisesRegex(ValueError, "different loan"):
            validate_events(self.events, ["a"])

    def test_invalid_phase_and_duplicate_events_rejected(self):
        self.events[0]["day"] = 0
        with self.assertRaisesRegex(ValueError, "phase boundary"):
            validate_events(self.events, ["a"])
        self.setUp()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_events(self.events + [self.events[0]], ["a"])

    def test_early_stopping_and_test_users_cannot_overlap_fit(self):
        parts = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
        validate_partitions(parts, list("abcd"), ["a"], ["b"])
        for fit, stop in [(["a", "d"], ["b"]), (["a"], ["c"]), (["a"], ["a", "b"])]:
            with self.assertRaises(ValueError):
                validate_partitions(parts, list("abcd"), fit, stop)

    def test_train_only_scaler_perturbation_reference(self):
        train = np.array([[1., 2.], [3., 4.]])
        heldout = np.array([[5., 6.]])
        scaler = StandardScaler().fit(train)
        original = scaler.transform(train).copy()
        mean = scaler.mean_.copy()
        scaler.transform(heldout * 1000000)
        np.testing.assert_array_equal(mean, scaler.mean_)
        np.testing.assert_array_equal(original, scaler.transform(train))
        self.assertEqual(scaler.n_samples_seen_, 2)

    def test_schema_order_and_generator_metadata_rejected(self):
        for actual in [["amount", "label"], ["time", "amount"]]:
            with self.assertRaises(ValueError):
                validate_feature_schema(actual, ["amount", "time"])

    def test_every_cache_identity_field_is_required(self):
        config = json.loads((Path(__file__).resolve().parents[2] / "src/configs/experiment_design.json").read_text())
        keys = config["cache"]["key_fields"]
        expected = {key: "original" for key in keys}
        validate_cache(expected, expected.copy(), keys)
        for key in keys:
            changed = {**expected, key: "changed"}
            with self.assertRaises(ValueError):
                validate_cache(expected, changed, keys)
            changed.pop(key)
            with self.assertRaises(ValueError):
                validate_cache(expected, changed, keys)


if __name__ == "__main__":
    unittest.main()
