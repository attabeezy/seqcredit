"""Extract aggregate evidence without executing notebooks or reading raw rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] = []
        self.buffer: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.buffer is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.buffer is not None:
            self.row.append("".join(self.buffer).strip())
            self.buffer = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)


def joined(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def table(cell: dict, required: str) -> tuple[int, list[dict[str, str]]]:
    for index, output in enumerate(cell.get("outputs", [])):
        parser = TableParser()
        parser.feed(joined(output.get("data", {}).get("text/html", [])))
        if parser.rows and required in parser.rows[0]:
            headers = parser.rows[0]
            if any(len(row) != len(headers) for row in parser.rows[1:]):
                raise ValueError("Incomplete HTML table")
            return index, [dict(zip(headers, row)) for row in parser.rows[1:]]
    raise ValueError(f"Missing aggregate table: {required}")


def require_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Missing required count: {pattern}")
    return int(match.group(1).replace(",", ""))


def extract(root: Path) -> dict:
    notebooks = {}
    sources = []
    for letter in "ABC":
        path = root / "archive/notebooks" / f"Analysis Notebook {letter}.ipynb"
        payload = path.read_bytes()
        notebooks[letter] = json.loads(payload)
        sources.append({"id": letter, "path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(payload).hexdigest()})

    records: list[dict] = []

    def evidence(letter: str, cell_index: int, output_index: int) -> dict:
        code = joined(notebooks[letter]["cells"][cell_index]["source"])
        # Keep customer/provider identifiers out of the aggregate ledger.
        sanitized = re.sub(r"(['\"])[A-F0-9]{16}\1", "'<LENDER_ID_REDACTED>'", code)
        return {"notebook": letter, "notebook_sha256": sources["ABC".index(letter)]["sha256"],
                "cell_index": cell_index, "output_index": output_index,
                "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "source_code": sanitized, "identifier_redaction": sanitized != code}

    def add(identifier: str, value: object, units: str, population: str,
            denominator: object, status: str, derivation: str, limitations: str,
            refs: list[tuple[str, int, int]]) -> None:
        records.append(dict(id=identifier, value=value, units=units, population=population,
                            denominator=denominator, status=status, derivation=derivation,
                            limitations=limitations, evidence=[evidence(*ref) for ref in refs]))

    aggregates = {}
    for letter, offset in [("A", 0), ("B", 1)]:
        cells = notebooks[letter]["cells"]
        text = joined(cells[1 + offset]["outputs"][0]["text"])
        match = re.search(r"\((\d+),\s*(\d+)\)", text)
        if not match:
            raise ValueError("Missing schema count")
        n_rows, n_columns = map(int, match.groups())
        type_output, types = table(cells[2 + offset], "TRANSACTION_TYPE")
        amount_output, amounts = table(cells[3 + offset], "avg_amount")
        if len(amounts) != 1:
            raise ValueError("Expected one amount summary row")
        aggregates[letter] = (n_rows, n_columns, types, amounts[0])
    if aggregates["A"] != aggregates["B"]:
        raise ValueError("Notebook A/B raw aggregates disagree; review before combining")
    n_rows, n_columns, types, amounts = aggregates["B"]
    total = sum(int(row["count"]) for row in types)
    if total != n_rows or len({row['TRANSACTION_TYPE'] for row in types}) != len(types):
        raise ValueError("Type counts do not reconcile or categories repeat")
    add("raw_rows", n_rows, "transactions", "full raw table", None, "context_only",
        "df.count()", "Not eligible-borrower or pre-index history size.", [("A", 1, 0), ("B", 2, 0)])
    add("raw_columns", n_columns, "columns", "full raw table", None, "context_only",
        "len(df.columns)", "Schema shape, not a distribution.", [("A", 1, 0), ("B", 2, 0)])
    for row in types:
        name, count = row["TRANSACTION_TYPE"], int(row["count"])
        add("transaction_type:" + name, {"count": count, "share": count / n_rows},
            "count and fraction", "all raw transactions, all party types", n_rows,
            "supported_raw_marginal", "groupBy(TRANSACTION_TYPE).count(); share=count/raw_rows",
            "Not borrower-specific or pre-index; includes organisation flows and reversals. "
            "Literal 'null' display is retained as an unknown category; mapping to generator types is pending.",
            [("A", 2, type_output), ("B", 3, type_output)])
    for key in ["min_amount", "max_amount", "avg_amount"]:
        add(key, float(amounts[key]), "source amount units (currency not specified by schema)",
            "raw table, non-null transaction amounts", "non-null count not exported",
            "limited_raw_summary", key + "(TRANSACTION_AMOUNT)",
            "Includes all party/type mixtures; no variance, quantiles, currency proof or outlier policy. "
            "Cannot fit an amount distribution from these values alone.", [("A", 3, 0), ("B", 4, 0)])
    for key in ["earliest", "latest"]:
        add("raw_timestamp_" + key, amounts[key], "timestamp string", "raw timestamp strings",
            None, "rejected", "String min/max, not parsed chronological extrema",
            "Cannot determine observation duration or transaction frequency.", [("A", 3, 0), ("B", 4, 0)])

    exploratory = {}
    for letter, offset in [("A", 0), ("B", 1)]:
        t1 = joined(notebooks[letter]["cells"][4 + offset]["outputs"][0]["text"])
        t2 = joined(notebooks[letter]["cells"][5 + offset]["outputs"][0]["text"])
        exploratory[letter] = [require_int(pattern, t1 + t2) for pattern in [
            r"Total borrowers: ([\d,]+)", r"Borrowers with penalties: ([\d,]+)",
            r"Borrowers \+ penalties \(overlap\): ([\d,]+)",
            r"Borrowers who made at least one repayment: ([\d,]+)",
            r"Borrowers with zero repayments: ([\d,]+)"]]
    if exploratory["A"] != exploratory["B"]:
        raise ValueError("Exploratory counts disagree")
    borrowers, penalised, overlap, repayers, never = exploratory["B"]
    for identifier, value, population, cell in [
        ("exploratory_borrowers", borrowers, "distinct customers receiving the selected lender's exact disbursement type", 5),
        ("all_penalised_customers", penalised, "distinct debit customers with the exact penalty type, not restricted to borrowers", 5),
        ("borrower_penalty_overlap", overlap, "selected borrowers intersected with all penalised customers", 5),
        ("all_repaying_customers", repayers, "distinct debit customers with principal/interest events, not restricted to borrowers", 6),
        ("borrowers_without_observed_repayment", never, "selected borrowers absent from all observed repayers", 6)]:
        add(identifier, value, "customers", population, None, "context_only",
            "Distinct count / join from preserved exploratory cell",
            "Whole observed history; no future window, loan attribution or minimum follow-up. Printed labels can misdescribe population.",
            [("A", cell - 1, 0), ("B", cell, 0)])
    for identifier, numerator, derivation in [
        ("exploratory_no_repayment_share", never, "never_repaid / selected_borrowers"),
        ("exploratory_penalty_share", overlap, "borrower_penalty_overlap / selected_borrowers")]:
        add(identifier, numerator / borrowers, "fraction", "selected exploratory borrowers", borrowers,
            "limited_proxy", derivation, "Not contractual default, repayment timeliness, or eligible cohort rate.",
            [("B", 5, 0), ("B", 6, 0)])

    cohorts = {}
    for letter, cell_index in [("A", 10), ("B", 12), ("C", 7)]:
        cell = notebooks[letter]["cells"][cell_index]
        index, rows = table(cell, "credit_risk_label")
        cohorts[letter] = (index, {int(row["credit_risk_label"]): int(row["count"]) for row in rows})
    labels = cohorts["B"][1]
    if any(value[1] != labels for value in cohorts.values()):
        raise ValueError("Eligible cohort label counts disagree")
    cohort_n = sum(labels.values())
    refs = [(letter, cell, cohorts[letter][0]) for letter, cell in [("A", 10), ("B", 12), ("C", 7)]]
    add("eligible_borrowers", cohort_n, "borrowers", "pipeline eligibility: index loan with >=60 days follow-up", None,
        "context_only", "sum of preserved label counts", "Not the 149204 feature-matched population.", refs)
    for label, count in sorted(labels.items()):
        add(f"eligible_label_{label}", {"count": count, "share": count / cohort_n}, "count and fraction",
            "eligible labeled cohort before feature matching", cohort_n, "supported_cohort_proxy",
            "count(label)/sum(counts)", "Historical operational labels: repayment without penalty (0), repayment plus penalty (1), "
            "no observed post-index repayment (2). Not contractual default. Follow-up varies; full label semantics need design review.", refs)

    timing_output, rows = table(notebooks["A"]["cells"][16], "avg_days_to_repay")
    if len(rows) != 1:
        raise ValueError("Expected one timing summary")
    for key, raw in rows[0].items():
        value = [int(x) for x in re.findall(r"\d+", raw)] if raw.startswith("List(") else float(raw)
        if key.startswith("n_"):
            value = int(value)
        add("timing_" + key, value, "customers" if key.startswith("n_") else "calendar days",
            "earliest observed disbursement joined to first observed repayment/penalty, restricted to days_to_repay > 0",
            107111 if "penalt" in key else 220452, "limited_conditional_timing",
            "p10,p25,p50,p75,p90,p95 via percentile_approx; datediff or count/avg as named",
            "Conditions on positive repayment lag; penalty summaries share that filter. No event-to-loan matching, "
            "no post-loan constraint on penalty, no censoring adjustment, no quantile accuracy argument recorded. Not loan tenure.",
            [("A", 16, timing_output)])

    legacy = json.loads((root / "src/synthetic_params.json").read_text(encoding="utf-8"))
    legacy_inventory = []

    def flatten(value: dict, prefix: str = "") -> None:
        for key, item in value.items():
            path = prefix + key
            if isinstance(item, dict):
                flatten(item, path + ".")
            else:
                legacy_inventory.append({"parameter": path, "legacy_value": item,
                    "status": "not_verified_by_this_extraction",
                    "note": "No automatic mapping or replacement; legacy types and populations differ."})

    flatten(legacy)
    return {"schema_version": 1, "description": "Aggregate evidence, not an executable generator configuration",
            "sources": sources, "cell_indices": "zero-based", "records": records,
            "legacy_parameter_inventory": legacy_inventory,
            "checks": {"raw_type_counts_sum": total, "raw_rows": n_rows, "type_categories": len(types),
                       "A_B_raw_aggregates_equal": True, "A_B_C_labels_equal": True},
            "not_recovered": ["amount variance/quantiles and conditional amounts", "borrower-specific type distribution",
                "balance distribution", "interarrival distribution and chronology", "weekend/night rates",
                "joint dependencies and sequence dynamics", "fees/taxes", "contractual default and loan tenure"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    ledger = extract(args.root)
    destination = args.root / "docs/calibration"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (destination / "ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "value", "units", "population", "denominator", "status", "derivation", "limitations", "evidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ledger["records"]:
            writer.writerow({**record, "value": json.dumps(record["value"], ensure_ascii=False),
                             "evidence": "; ".join(f"{r['notebook']}:cell={r['cell_index']}:output={r['output_index']}:{r['notebook_sha256']}" for r in record['evidence'])})
    print(json.dumps({"records": len(ledger["records"]), "checks": ledger["checks"]}))


if __name__ == "__main__":
    main()
