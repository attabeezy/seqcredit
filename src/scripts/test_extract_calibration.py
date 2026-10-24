"""Standalone extraction checks; no ML environment or notebook execution needed."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import extract_calibration as extractor


class ExtractionChecks(unittest.TestCase):
    def test_html_entities_and_nested_markup(self):
        cell = {"outputs": [{"data": {"text/html": [
            "<table><tr><th>type</th><th>count</th></tr>",
            "<tr><td>A &amp; <b>B</b></td><td>7</td></tr></table>"]}}]}
        self.assertEqual(extractor.table(cell, "type"), (0, [{"type": "A & B", "count": "7"}]))

    def test_truncated_row_rejected(self):
        cell = {"outputs": [{"data": {"text/html":
            "<tr><th>type</th><th>count</th></tr><tr><td>A</td></tr>"}}]}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            extractor.table(cell, "type")

    def test_missing_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            extractor.require_int(r"count: (\d+)", "count: unavailable")

    def test_archive_reconciles_and_excludes_identifiers(self):
        root = Path(__file__).resolve().parents[2]
        ledger = extractor.extract(root)
        by_id = {row["id"]: row for row in ledger["records"]}
        types = [row for row in ledger["records"] if row["id"].startswith("transaction_type:")]
        self.assertEqual(sum(row["value"]["count"] for row in types), by_id["raw_rows"]["value"])
        self.assertAlmostEqual(sum(row["value"]["share"] for row in types), 1.0)
        self.assertEqual(by_id["raw_timestamp_earliest"]["status"], "rejected")
        self.assertEqual(by_id["timing_avg_days_to_repay"]["status"], "limited_conditional_timing")
        self.assertNotIn("E7C89F8C4A27F173", json.dumps(ledger))
        self.assertTrue(all(row["evidence"] for row in ledger["records"]))

    def test_conflicting_notebooks_rejected(self):
        original = extractor.table
        changed = False

        def conflicting(cell, required):
            nonlocal changed
            output, rows = original(cell, required)
            if required == "avg_amount" and not changed:
                rows[0]["avg_amount"] = "123.0"
                changed = True
            return output, rows

        with patch.object(extractor, "table", side_effect=conflicting):
            with self.assertRaisesRegex(ValueError, "raw aggregates disagree"):
                extractor.extract(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    unittest.main()
