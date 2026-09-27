"""Generate compact fabricated transaction histories for the presentation MVP.

The schema mirrors the useful concepts in the research synthetic pipeline while
avoiding financial identifiers and per-user file sprawl. Labels are generated
from latent aggregate and ordering signals; no real records are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRANSACTION_TYPES = np.array(
    ["TRANSFER", "DEBIT", "PAYMENT", "PAYMENT_SEND", "CASH_OUT", "CASH_IN"]
)


MFS_ARCHETYPES = (
    "STANDARD",
    "SALARY_CYCLICAL",
    "SEASONAL_MERCHANT",
    "PAYDAY_RUSH",
    "AGENT_REBALANCING",
)


@dataclass(frozen=True)
class SyntheticConfig:
    n_borrowers: int = 4_500
    transactions_per_borrower: int = 36
    default_rate: float = 0.0413
    late_rate: float = 0.6190
    static_label_weight: float = 0.78
    sequence_label_weight: float = 0.56
    label_noise: float = 1.72
    sequence_observation_strength: float = 0.34
    seed: int = 42
    enable_mfs_archetypes: bool = True
    forced_archetype: str | None = None


def _assign_labels(propensity: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    """Assign exact presentation class proportions by latent-risk rank."""
    n = len(propensity)
    n_default = max(1, round(n * config.default_rate))
    n_late = max(1, round(n * config.late_rate))
    order = np.argsort(propensity)[::-1]
    labels = np.zeros(n, dtype=np.int8)
    labels[order[:n_default]] = 2
    labels[order[n_default : n_default + n_late]] = 1
    return labels


def generate_synthetic_cohort(
    config: SyntheticConfig = SyntheticConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fabricated transactions and borrower labels.

    Aggregate risk changes overall transaction mix and liquidity. Independent
    sequence risk changes *where* activity occurs in the history while keeping
    its full-period average comparatively stable. This creates a controlled
    reason for order-aware models to outperform aggregate-only models.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_borrowers
    t = config.transactions_per_borrower

    static_risk = rng.normal(size=n)
    sequence_risk = rng.normal(size=n)
    label_noise = rng.normal(size=n)
    propensity = (
        config.static_label_weight * static_risk
        + config.sequence_label_weight * sequence_risk
        + config.label_noise * label_noise
    )
    labels = _assign_labels(propensity, config)

    # Assign MFS Behavioral Archetypes
    if config.forced_archetype:
        archetype_assignments = np.array([config.forced_archetype] * n)
    elif config.enable_mfs_archetypes:
        archetype_probs = [0.40, 0.15, 0.15, 0.15, 0.15]
        archetype_assignments = rng.choice(MFS_ARCHETYPES, size=n, p=archetype_probs)
    else:
        archetype_assignments = np.array(["STANDARD"] * n)

    borrower_ids = np.array([f"DEMO_{i:05d}" for i in range(n)])
    user_idx = np.repeat(np.arange(n), t)
    position = np.tile(np.linspace(-1.0, 1.0, t), n)
    txn_number = np.tile(np.arange(1, t + 1), n)
    s = static_risk[user_idx]
    q = sequence_risk[user_idx]
    user_archetypes = archetype_assignments[user_idx]

    hour_gap = rng.gamma(2.0, 7.5, size=n * t)
    elapsed_hours = np.empty(n * t)
    for i in range(n):
        sl = slice(i * t, (i + 1) * t)
        elapsed_hours[sl] = np.cumsum(hour_gap[sl])
    start = pd.Timestamp("2026-01-01T08:00:00")
    timestamps = start + pd.to_timedelta(elapsed_hours, unit="h")

    base_amount = np.exp(3.20 + 0.20 * s + rng.normal(0, 0.28, n * t))
    order_multiplier = np.clip(
        1.0
        + config.sequence_observation_strength * q * position
        + rng.normal(0, 0.22, n * t),
        0.12,
        3.5,
    )

    # Modulate behavior based on MFS archetypes
    is_salary = user_archetypes == "SALARY_CYCLICAL"
    is_merchant = user_archetypes == "SEASONAL_MERCHANT"
    is_payday = user_archetypes == "PAYDAY_RUSH"
    is_agent = user_archetypes == "AGENT_REBALANCING"

    # Salary cyclicality: periodic influx and steady drawdown
    salary_cycle = np.sin(position * 2 * np.pi)
    base_balance_shift = np.zeros(n * t)
    base_balance_shift[is_salary] += 0.35 * salary_cycle[is_salary]

    # Seasonal merchant: burst spikes in amount
    merchant_spike = rng.random(n * t) < 0.25
    order_multiplier[is_merchant & merchant_spike] *= 1.85

    amounts = np.clip(base_amount * order_multiplier, 0.5, 1_500.0)

    base_balance = np.exp(5.55 - 0.28 * s + rng.normal(0, 0.24, n * t) + base_balance_shift)
    balance_trend = np.clip(1.0 - 0.18 * q * position, 0.25, 2.0)
    balance_before = np.clip(base_balance * balance_trend, 0.0, 4_000.0)

    cashout_score = -1.65 + 0.30 * s + 0.18 * q * position
    # Agent rebalancing: high cashout and cashin balance
    cashout_score[is_agent] += 0.45

    cashout_p = 1 / (1 + np.exp(-cashout_score))
    debit_p = np.clip(0.25 + 0.025 * s, 0.12, 0.40)
    payment_p = np.clip(0.20 - 0.015 * s, 0.08, 0.30)
    cashin_p = np.clip(0.08 - 0.012 * s, 0.03, 0.14)

    # Payday rush: higher payment & debit concentration
    debit_p[is_payday] += 0.10
    payment_p[is_payday] += 0.08
    # Agent: higher cashin
    cashin_p[is_agent] += 0.18

    transfer_p = np.clip(1.0 - cashout_p - debit_p - payment_p - cashin_p, 0.10, 0.55)
    probs = np.column_stack(
        [transfer_p, debit_p, payment_p * 0.55, payment_p * 0.45, cashout_p, cashin_p]
    )
    probs = probs / probs.sum(axis=1, keepdims=True)
    draws = rng.random(n * t)
    txn_type_idx = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    txn_types = TRANSACTION_TYPES[txn_type_idx]

    incoming = txn_types == "CASH_IN"
    outgoing = ~incoming
    balance_after = balance_before.copy()
    balance_after[incoming] += amounts[incoming]
    balance_after[outgoing] -= np.minimum(amounts[outgoing], balance_before[outgoing])
    balance_after = np.clip(balance_after, 0.0, None)

    night_shift = 2.5 * q * position
    hours = np.mod(timestamps.hour.to_numpy() + np.rint(night_shift).astype(int), 24)
    timestamps = timestamps.normalize() + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(
        timestamps.minute, unit="m"
    )

    recipient_bucket = rng.integers(0, 18, size=n * t)
    repeat_bias = rng.random(n * t) < np.clip(0.44 - 0.05 * s, 0.20, 0.70)
    recipient_bucket[repeat_bias] %= 5
    recipients = np.char.add("RECIPIENT_", np.char.zfill(recipient_bucket.astype(str), 2))

    fee_eligible = np.isin(txn_types, ["CASH_OUT", "PAYMENT_SEND"])
    fees = np.where(fee_eligible, np.minimum(5.0, np.maximum(0.25, amounts * 0.01)), 0.0)

    tx_data = {
        "borrower_id": borrower_ids[user_idx],
        "transaction_id": [f"TXN_{i:08d}" for i in range(n * t)],
        "timestamp": timestamps,
        "transaction_type": txn_types,
        "amount": amounts.round(2),
        "balance_before": balance_before.round(2),
        "balance_after": balance_after.round(2),
        "fee": fees.round(2),
        "recipient_id": recipients,
        "synthetic_demo": True,
        "transaction_number": txn_number,
    }
    if config.enable_mfs_archetypes:
        tx_data["archetype"] = archetype_assignments[user_idx]

    transactions = pd.DataFrame(tx_data)
    transactions = transactions.sort_values(["borrower_id", "timestamp", "transaction_id"])

    label_names = np.array(["good", "late", "default"])
    label_frame = pd.DataFrame(
        {
            "borrower_id": borrower_ids,
            "credit_risk_label": labels,
            "outcome": label_names[labels],
            "mfs_archetype": archetype_assignments,
            "synthetic_demo": True,
        }
    )
    return transactions.reset_index(drop=True), label_frame


def generate_mfs_archetype_sample(
    archetype: str | None = None,
    n_borrowers: int = 10,
    transactions_per_borrower: int = 36,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a specialized synthetic cohort exhibiting MFS archetypes."""
    if archetype is not None:
        if archetype not in MFS_ARCHETYPES:
            raise ValueError(f"Unknown archetype: {archetype}. Must be one of {MFS_ARCHETYPES}")
        cfg = SyntheticConfig(
            n_borrowers=n_borrowers,
            transactions_per_borrower=transactions_per_borrower,
            seed=seed,
            enable_mfs_archetypes=True,
            forced_archetype=archetype,
        )
        return generate_synthetic_cohort(cfg)

    # When archetype is None, guarantee equal representation across all canonical archetypes
    n_per = max(1, n_borrowers // len(MFS_ARCHETYPES))
    all_txs = []
    all_labels = []
    for idx, arch in enumerate(MFS_ARCHETYPES):
        cfg = SyntheticConfig(
            n_borrowers=n_per,
            transactions_per_borrower=transactions_per_borrower,
            seed=seed + idx * 101,
            enable_mfs_archetypes=True,
            forced_archetype=arch,
        )
        tx_part, lbl_part = generate_synthetic_cohort(cfg)
        id_map = {b: f"MFS_{arch[:3]}_{b}" for b in lbl_part["borrower_id"]}
        tx_part["borrower_id"] = tx_part["borrower_id"].map(id_map)
        tx_part["transaction_id"] = [f"TXN_ARCH_{idx}_{i:06d}" for i in range(len(tx_part))]
        lbl_part["borrower_id"] = lbl_part["borrower_id"].map(id_map)
        all_txs.append(tx_part)
        all_labels.append(lbl_part)

    combined_txs = pd.concat(all_txs, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    return combined_txs, combined_labels
