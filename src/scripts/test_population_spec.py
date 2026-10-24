"""Standard-library checks for category precedence and population design."""

from pathlib import Path
import json
import unittest

from build_population_spec import build, classify


class PopulationChecks(unittest.TestCase):
    def test_reversal_precedes_financial_category(self):
        for name in ["Withdrawal Reversal", "Pay Merchant Reversal", "FSI Transfer from Bank to Wallet Reversal"]:
            self.assertEqual(classify(name)[:2], ("reversal_adjustment", "exclude"))

    def test_bank_and_betting_not_swallowed_by_generic_transfer(self):
        self.assertEqual(classify("FSI Customer Transfer from Bank to Wallet_CBG")[0], "bank_wallet")
        self.assertEqual(classify("Organisation Payment to Customer _Betting")[0], "betting")

    def test_infrastructure_and_missing_types_not_customer_activity(self):
        for name in ["Organization Transfer from Float Account to M-Pesa Account via STK", "Interest Payment To organisation", "null"]:
            self.assertEqual(classify(name)[1], "exclude")
        self.assertEqual(classify("Loan Payment Via API")[1], "loan_process_only")

    def test_coverage_mass_and_conditional_denominator(self):
        root = Path(__file__).resolve().parents[2]
        mapping = build(root)
        self.assertEqual(len(mapping["mapping"]), 243)
        self.assertEqual(sum(row["count"] for row in mapping["mapping"]), mapping["raw_count"])
        candidates = [row for row in mapping["mapping"] if row["role"] == "background_candidate"]
        self.assertAlmostEqual(sum(row["scenario_reference_share"] for row in candidates), 1.0)
        self.assertTrue(all(row["scenario_reference_share"] is None for row in mapping["mapping"] if row not in candidates))

    def test_design_discloses_assumptions_and_fits_memory_budget(self):
        root = Path(__file__).resolve().parents[2]
        design = json.loads((root / "src/configs/experiment_design.json").read_text())
        self.assertEqual(sum(design["split"][key] for key in ["train", "validation", "test"]), 1.0)
        main = design["profiles"]["main"]
        tensor_bytes = main["borrowers"] * main["max_sequence_length"] * design["execution"]["sequence_features_budget"] * 4
        self.assertEqual(main["borrowers"], 25625)
        self.assertLess(tensor_bytes, 160 * 1024**2)
        self.assertIn("scenario", design["window_provenance"])
        self.assertEqual(design["status"], "contract_frozen_implementation_pending")


if __name__ == "__main__":
    unittest.main()
