"""Boundary checks for the v2 runner; independent of legacy model imports."""

import hashlib
import json
import math


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_history(events: list[dict], borrower_id: str, limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("Sequence limit must be positive")
    selected = [event for event in events if event["borrower_id"] == borrower_id
                and -90 <= event["day"] < 0 and event["phase"] == "background"]
    return sorted(selected, key=lambda event: (event["day"], event["event_id"]))[-limit:]


def derive_label(events: list[dict], borrower_id: str, loan_id: str) -> int:
    future = [event for event in events if event["borrower_id"] == borrower_id
              and event.get("loan_id") == loan_id and 0 < event["day"] <= 60]
    repaid = any(event["phase"] in {"principal_repayment", "interest_repayment"}
                 and event["amount"] > 0 for event in future)
    penalty = any(event["phase"] == "penalty" for event in future)
    return 2 if not repaid else 1 if penalty else 0


def validate_partitions(partitions: dict[str, list[str]], eligible_ids: list[str],
                        fit_ids: list[str], early_stop_ids: list[str]) -> None:
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("Require train, validation and test partitions")
    all_ids = [uid for ids in partitions.values() for uid in ids]
    if len(all_ids) != len(set(all_ids)) or len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("Duplicate or overlapping borrower IDs")
    if set(all_ids) != set(eligible_ids) or not all(partitions.values()):
        raise ValueError("Partition population mismatch or empty partition")
    if (not fit_ids or not early_stop_ids or len(set(fit_ids)) != len(fit_ids)
            or len(set(early_stop_ids)) != len(early_stop_ids)
            or set(fit_ids) & set(early_stop_ids)
            or set(fit_ids) | set(early_stop_ids) != set(partitions["train"])):
        raise ValueError("Fit and early-stop sets must partition training borrowers")


def validate_feature_schema(actual: list[str], expected: list[str]) -> None:
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("Feature allowlist/order mismatch")


def validate_events(events: list[dict], borrower_ids: list[str]) -> None:
    seen = set()
    eligible = set(borrower_ids)
    by_borrower = {uid: [] for uid in borrower_ids}
    for event in events:
        if event["borrower_id"] not in eligible or event["event_id"] in seen:
            raise ValueError("Unknown borrower or duplicate event")
        seen.add(event["event_id"])
        by_borrower[event["borrower_id"]].append(event)
        if not math.isfinite(event["day"]) or not math.isfinite(event["amount"]) or event["amount"] <= 0:
            raise ValueError("Nonfinite time or invalid amount")
        phase, day = event["phase"], event["day"]
        valid = ((phase == "background" and -90 <= day < 0)
                 or (phase == "index_loan" and day == 0)
                 or (phase in {"principal_repayment", "interest_repayment", "penalty"} and 0 < day <= 60))
        if not valid:
            raise ValueError("Event crosses its phase boundary")
    for uid in borrower_ids:
        own = by_borrower[uid]
        index = [event for event in own if event["phase"] == "index_loan"]
        if len(index) != 1 or not index[0].get("loan_id"):
            raise ValueError("Require exactly one identified index loan")
        if not any(event["phase"] == "background" for event in own):
            raise ValueError("Missing pre-index history")
        if any(event.get("loan_id") != index[0]["loan_id"] for event in own
               if event["phase"] in {"principal_repayment", "interest_repayment", "penalty"}):
            raise ValueError("Future event linked to a different loan")


def validate_cache(expected: dict, actual: dict, required_keys: list[str]) -> None:
    if any(key not in expected or key not in actual for key in required_keys):
        raise ValueError("Incomplete cache identity")
    if any(expected[key] != actual[key] for key in required_keys):
        raise ValueError("Cache identity mismatch; no fallback permitted")
