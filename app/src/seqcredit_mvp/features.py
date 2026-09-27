"""Stable aggregate and order-aware feature contracts for demo scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "borrower_id",
    "transaction_id",
    "timestamp",
    "transaction_type",
    "amount",
    "balance_before",
    "balance_after",
    "fee",
    "recipient_id",
}


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded synthetic transaction batch."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The transaction file is empty.")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed.")
    for column in ["amount", "balance_before", "balance_after", "fee"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains non-numeric or missing values.")
    if (df[["amount", "balance_before", "balance_after", "fee"]] < 0).any().any():
        raise ValueError("Amounts, balances, and fees must be non-negative.")
    if df[["borrower_id", "transaction_id", "transaction_type", "recipient_id"]].isna().any().any():
        raise ValueError("Identifier and transaction-type columns cannot be missing.")
    for column in ["borrower_id", "transaction_id", "transaction_type", "recipient_id"]:
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(f"Column '{column}' cannot contain blank values.")
    if df["transaction_id"].duplicated().any():
        raise ValueError("transaction_id values must be unique.")
    return df.sort_values(["borrower_id", "timestamp", "transaction_id"]).reset_index(drop=True)


def _slope(values: pd.Series) -> float:
    y = values.to_numpy(dtype=float)
    if len(y) < 2 or np.allclose(y, y[0]):
        return 0.0
    x = np.linspace(-1.0, 1.0, len(y))
    return float(np.polyfit(x, y, 1)[0] / (np.mean(np.abs(y)) + 1e-6))


def _late_early_ratio(values: pd.Series) -> float:
    y = values.to_numpy(dtype=float)
    width = max(1, len(y) // 3)
    return float((y[-width:].mean() + 1.0) / (y[:width].mean() + 1.0))


def build_feature_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aggregate-only and order-aware borrower feature tables."""
    df = validate_transactions(frame)
    df["is_cashout"] = (df["transaction_type"] == "CASH_OUT").astype(float)
    df["is_transfer"] = (df["transaction_type"] == "TRANSFER").astype(float)
    df["is_debit"] = (df["transaction_type"] == "DEBIT").astype(float)
    df["is_payment"] = df["transaction_type"].isin(["PAYMENT", "PAYMENT_SEND"]).astype(float)
    df["is_night"] = ((df["timestamp"].dt.hour < 6) | (df["timestamp"].dt.hour >= 22)).astype(float)
    df["is_low_balance"] = (df["balance_before"] < 20).astype(float)
    df["elapsed_hours"] = df.groupby("borrower_id")["timestamp"].diff().dt.total_seconds().div(3600)

    aggregate = df.groupby("borrower_id", sort=True).agg(
        transaction_count=("transaction_id", "count"),
        total_volume=("amount", "sum"),
        avg_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        std_amount=("amount", "std"),
        max_amount=("amount", "max"),
        pct_cashout=("is_cashout", "mean"),
        pct_transfer=("is_transfer", "mean"),
        pct_debit=("is_debit", "mean"),
        pct_payment=("is_payment", "mean"),
        pct_night=("is_night", "mean"),
        avg_balance=("balance_before", "mean"),
        min_balance=("balance_before", "min"),
        balance_volatility=("balance_before", "std"),
        pct_low_balance=("is_low_balance", "mean"),
        total_fees=("fee", "sum"),
        unique_recipients=("recipient_id", "nunique"),
        avg_hours_between=("elapsed_hours", "mean"),
    ).fillna(0.0)

    ordered_rows = []
    for borrower_id, group in df.groupby("borrower_id", sort=True):
        group = group.sort_values(["timestamp", "transaction_id"])
        ordered_rows.append(
            {
                "borrower_id": borrower_id,
                "amount_slope": _slope(group["amount"]),
                "balance_slope": _slope(group["balance_before"]),
                "amount_recent_ratio": _late_early_ratio(group["amount"]),
                "balance_recent_ratio": _late_early_ratio(group["balance_before"]),
                "cashout_slope": _slope(group["is_cashout"]),
                "night_slope": _slope(group["is_night"]),
                "recent_cashout_rate": group["is_cashout"].tail(max(1, len(group) // 3)).mean(),
                "recent_low_balance_rate": group["is_low_balance"].tail(max(1, len(group) // 3)).mean(),
            }
        )
    ordered = pd.DataFrame(ordered_rows).set_index("borrower_id")
    sequence = aggregate.join(ordered, how="inner")
    return aggregate.astype(float), sequence.astype(float)
