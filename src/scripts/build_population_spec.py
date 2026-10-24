"""Build a conservative name-based mapping; never infer a borrower marginal."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def classify(name: str) -> tuple[str, str, str]:
    """Return category, role, and reason in precedence order."""
    text = name.casefold()
    if text == "null":
        return "unknown", "exclude", "Unknown source category"
    if any(word in text for word in ["reversal", "reverse", "clawback", "expire"]):
        return "reversal_adjustment", "exclude", "Requires linked original transaction; not ordinary activity"
    if any(word in text for word in ["loan", "od_principal", "od_interest", "od_penalty"]):
        category = ("loan_penalty" if "penalty" in text or "penal" in text else
                    "loan_interest" if "interest" in text else
                    "loan_principal_repayment" if "principal" in text else
                    "loan_disbursement" if "loan payment" in text else "loan_other")
        return category, "loan_process_only", "Simulate by explicit loan lifecycle, never raw event prevalence"
    if "organisation" in text and "to customer" not in text:
        return "organisation_infrastructure", "exclude", "Organisation flow without explicit customer destination"
    if any(word in text for word in ["commission", "float", "working account", "utility account",
                                      "interop account", "org mpesa", "org m-pesa", "hierarchy",
                                      "business deposit", "business withdrawal", "evd top-up",
                                      "top org", "organisation via", "organization", "org transfer",
                                      "business to business", "bulk b2b", "bulk o2o", "auto sweep"]):
        return "organisation_infrastructure", "exclude", "Name indicates institutional/infrastructure flow"
    if any(word in text for word in ["email statement", "subscription", "terms and conditions", "bulk sms"]):
        return "service_admin", "exclude", "Administrative/service event; no financial interpretation established"
    # More specific meanings precede generic transfer/payment words.
    if "betting" in text:
        return "betting", "background_candidate", "Name suggests betting flow; direction requires account perspective"
    if "fsi" in text or "bank" in text or "w2b" in text or "b2w" in text:
        return "bank_wallet", "background_candidate", "Bank/wallet flow; name alone does not establish customer population"
    if "airtime" in text or "data purchase" in text:
        return "airtime_data", "background_candidate", "Retail airtime/data name"
    if "withdrawal" in text or "give cash" in text:
        return "cash_out", "background_candidate", "Cash withdrawal name; customer-facing assumption needs validation"
    if "deposit at agent" in text:
        return "cash_in", "background_candidate", "Agent deposit name; customer-facing assumption needs validation"
    if any(word in text for word in ["salary", "interest payment to customer", "organisation payment to customer",
                                      "business payment to customer", "business to customer", "b2c", "international transfer",
                                      "insurance claim", "promotion payment"]):
        return "incoming_payment", "background_candidate", "Name suggests incoming payment to customer"
    if any(word in text for word in ["transfer", "p2p", "send money"]):
        return "transfer", "background_candidate", "Transfer name; party roles/direction unavailable in aggregates"
    if any(word in text for word in ["payment", "pay bill", "pay merchant", "pay utility", "customer to business", "c2b"]):
        return "merchant_bill_payment", "background_candidate", "Payment name; borrower applicability not established"
    return "unresolved", "exclude", "Ambiguous name retained explicitly; no guessed mapping"


def build(root: Path) -> dict:
    ledger_path = root / "docs/calibration/ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = []
    groups = defaultdict(lambda: {"count": 0, "categories": 0})
    for record in ledger["records"]:
        if not record["id"].startswith("transaction_type:"):
            continue
        name = record["id"].split(":", 1)[1]
        category, role, reason = classify(name)
        row = {"raw_type": name, "count": record["value"]["count"],
               "raw_share": record["value"]["share"], "category": category,
               "role": role, "reason": reason, "ledger_id": record["id"]}
        rows.append(row)
        groups[category]["count"] += row["count"]
        groups[category]["categories"] += 1
    total = ledger["checks"]["raw_rows"]
    if len(rows) != ledger["checks"]["type_categories"] or sum(row["count"] for row in rows) != total:
        raise ValueError("Mapping must cover every source category and conserve count mass")
    candidate_n = sum(row["count"] for row in rows if row["role"] == "background_candidate")
    for row in rows:
        row["scenario_reference_share"] = row["count"] / candidate_n if row["role"] == "background_candidate" else None
    return {"version": 1, "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            "interpretation": "Name-based semantic map; filtered raw shares are scenario references, not measured borrower probabilities",
            "raw_count": total, "background_candidate_count": candidate_n,
            "categories": dict(sorted(groups.items())), "mapping": rows}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = build(root)
    dest = root / "docs/calibration"
    (dest / "category_mapping.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (dest / "category_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["mapping"][0]))
        writer.writeheader()
        writer.writerows(result["mapping"])
    print(json.dumps({"mapped": len(result["mapping"]), "groups": result["categories"]}))


if __name__ == "__main__":
    main()
