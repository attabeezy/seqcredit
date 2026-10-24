"""Guarded v2 generation, features, and partition-local preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from contract_checks import derive_label, fingerprint, validate_events, validate_partitions


ROOT = Path(__file__).resolve().parents[2]


def load_contract() -> dict:
    return json.loads((ROOT / "src/configs/experiment_design.json").read_text(encoding="utf-8"))


def stream(seed: int, name: str, borrower: int = 0) -> np.random.Generator:
    tag = int(fingerprint(name)[:8], 16)
    return np.random.default_rng(np.random.SeedSequence([seed, tag, borrower]))


def prevalence_intercept(prevalence: float, beta: float) -> tuple[float, dict]:
    reference = np.random.default_rng(1042).normal(size=100000)
    low, high = -30., 30.
    for _ in range(80):
        middle = (low + high) / 2
        if expit(middle + beta * reference).mean() < prevalence:
            low = middle
        else:
            high = middle
    intercept = (low + high) / 2
    achieved = float(expit(intercept + beta * reference).mean())
    if abs(achieved - prevalence) > 1e-10:
        raise ValueError("Independent generator calibration did not converge")
    return intercept, {"value": intercept, "target": prevalence, "achieved": achieved,
                       "reference_seed": 1042, "reference_hash": fingerprint(reference.tolist())}


def features(events: list[dict], categories: list[str], limit: int) -> tuple[np.ndarray, np.ndarray]:
    history = sorted([row for row in events if row["phase"] == "background" and -90 <= row["day"] < 0],
                     key=lambda row: (row["day"], row["event_id"]))[-limit:]
    if not history:
        raise ValueError("Missing eligible history")
    amount = np.array([row["amount"] for row in history])
    day = np.array([row["day"] for row in history])
    incoming = np.array([row["is_incoming"] for row in history], dtype=float)
    onehot = np.array([[float(row["category"] == name) for name in categories] for row in history])
    if not np.all(onehot.sum(axis=1) == 1):
        raise ValueError("Unknown category")
    gaps = np.diff(day) * 24
    static = np.array([len(history), amount.sum(), amount.mean(), amount.std(ddof=0), amount.min(),
                       amount.max(), gaps.mean() if len(gaps) else 0,
                       len({row["counterparty"] for row in history}), incoming.mean(),
                       np.sum(amount * (2 * incoming - 1)), *onehot.mean(axis=0)], dtype=np.float64)
    sequence = np.column_stack([np.log1p(amount), incoming, np.r_[0., gaps], day, onehot]).astype(np.float32)
    if not np.isfinite(static).all() or not np.isfinite(sequence).all():
        raise ValueError("Nonfinite feature")
    return static, sequence


def generate(contract: dict, profile: str, seed: int, scenario: str,
             event_sink=None, outcome_seed: int | None = None) -> dict:
    """Stream validated borrower events; return only compact model inputs."""
    scenarios = {item["id"]: item for item in contract["scenarios"]}
    if scenario not in scenarios:
        raise ValueError("Unknown scenario")
    setting = contract["profiles"][profile]
    n, limit = setting["borrowers"], setting["max_sequence_length"]
    counts = contract["outcome"]["reference_counts"]
    prevalence = counts["no_repayment"] / contract["outcome"]["reference_denominator"]
    beta = scenarios[scenario]["risk_beta"]
    intercept, calibration = prevalence_intercept(prevalence, beta)
    mapping_path = ROOT / contract["background_events"]["mapping"]
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    candidates = [row for row in mapping["mapping"] if row["role"] == "background_candidate"]
    weights = np.array([row["scenario_reference_share"] for row in candidates])
    if abs(weights.sum() - 1) > 1e-10:
        raise ValueError("Invalid category weights")
    categories = contract["features"]["categories"]
    amounts = contract["background_events"]["amount_distribution"]
    directions = contract["background_events"]["direction_and_counterparties"]
    loan = contract["loan_lifecycle"]["amount"]
    ids, labels, static_rows, sequences, label_rows = [], [], [], [], []
    invariance_checked = 0
    event_count = 0
    for i in range(n):
        uid, loan_id = f"SIM_{i:06d}", f"SIM_LOAN_{i:06d}"
        background, latent = stream(seed, "background", i), stream(seed, "latent", i)
        outcome = stream(seed if outcome_seed is None else outcome_seed, "outcome", i)
        future, principal_rng = stream(seed, "future_timing", i), stream(seed, "loan", i)
        z = float(latent.normal())
        count = 2 + int(background.poisson(98))
        days = np.sort(background.uniform(-90, 0, count))
        raw_amount = background.lognormal(np.log(amounts["median_units"]), amounts["log_sigma"], count)
        category_indices = background.choice(len(candidates), size=count, p=weights)
        incoming_uniform = background.random(count)
        counterparties = background.integers(0, directions["counterparty_pool_size"], count)
        if scenario == "aggregate_signal":
            raw_amount *= np.exp(0.35 * z)
        values = np.round(np.clip(raw_amount, amounts["minimum_units"], amounts["maximum_units"]), 2)
        events = []
        for j in range(count):
            row = candidates[int(category_indices[j])]
            category = row["category"]
            events.append(dict(borrower_id=uid, event_id=f"{uid}_{j:04d}", day=float(days[j]),
                               phase="background", amount=float(values[j]), loan_id="", category=category,
                               raw_type=row["raw_type"], is_incoming=int(incoming_uniform[j] < directions["incoming_probability"][category]),
                               counterparty=f"{uid}_CP_{counterparties[j]:02d}"))
        if scenario == "order_signal":
            before, _ = features(events, categories, limit)
            payload_names = ["amount", "category", "raw_type", "is_incoming", "counterparty"]
            retained = events[-limit:]
            payloads = sorted([{key: event[key] for key in payload_names} for event in retained],
                              key=lambda item: item["amount"], reverse=z < 0)
            for row, payload in zip(retained, payloads):
                row.update(payload)
            after, _ = features(events, categories, limit)
            np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-9)
            invariance_checked += 1
        principal = float(np.round(np.clip(principal_rng.lognormal(np.log(loan["median_units"]), loan["log_sigma"]),
                                         loan["minimum_units"], loan["maximum_units"]), 2))

        def future_event(phase: str, day: float, amount: float) -> dict:
            return dict(borrower_id=uid, event_id=f"{uid}_{phase}", day=day, phase=phase,
                        amount=amount, loan_id=loan_id, category=phase, raw_type="simulated_lifecycle",
                        is_incoming=int(phase == "index_loan"), counterparty="SIM_LENDER")

        events.append(future_event("index_loan", 0., principal))
        if outcome.random() >= expit(intercept + beta * z):
            if outcome.random() < counts["repaid_with_penalty"] / (counts["repaid_without_penalty"] + counts["repaid_with_penalty"]):
                penalty_day = float(future.uniform(31, 45))
                repay_day = float(future.uniform(np.nextafter(penalty_day, 60.), 60))
                events.append(future_event("penalty", penalty_day, round(principal * 0.01, 2)))
            else:
                repay_day = float(future.uniform(1, 30))
            events.append(future_event("principal_repayment", repay_day, principal))
        validate_events(events, [uid])
        label = derive_label(events, uid, loan_id)
        static, sequence = features(events, categories, limit)
        ids.append(uid)
        labels.append(label)
        static_rows.append(static)
        sequences.append(sequence)
        label_rows.append(dict(borrower_id=uid, loan_id=loan_id, label=label,
                               observation_start_day=-90, index_day=0, followup_end_day=60))
        if event_sink is not None:
            event_sink(events)
        event_count += len(events)
    return dict(ids=np.array(ids), labels=np.array(labels, dtype=np.int32),
                static=np.stack(static_rows), sequences=sequences, label_rows=label_rows,
                calibration=calibration, event_count=event_count, order_invariance_checked=invariance_checked)


def partitions(ids: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    indices = np.arange(len(ids))
    train, remainder = train_test_split(indices, train_size=0.6, stratify=labels, random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.5, stratify=labels[remainder], random_state=seed)
    fit, stop = train_test_split(train, test_size=0.15, stratify=labels[train], random_state=seed)
    result = dict(train=train, validation=validation, test=test, fit=fit, early_stop=stop)
    validate_partitions({name: ids[result[name]].tolist() for name in ["train", "validation", "test"]},
                        ids.tolist(), ids[fit].tolist(), ids[stop].tolist())
    for name, idx in result.items():
        if len(np.unique(labels[idx] == 2)) != 2:
            raise ValueError(f"Partition {name} lacks a primary class")
    return result


def preprocess(data: dict, split: dict, limit: int) -> dict:
    """Fit only inner-fit rows/timesteps; transform and explicitly mask all rows."""
    fit = split["fit"]
    static_scaler = StandardScaler().fit(data["static"][fit])
    continuous = [0, 2, 3]
    real_fit = np.concatenate([data["sequences"][i][:, continuous] for i in fit])
    seq_scaler = StandardScaler().fit(real_fit)
    n = len(data["ids"])
    tensor = np.zeros((n, limit, data["sequences"][0].shape[1]), dtype=np.float32)
    mask = np.zeros((n, limit), dtype=bool)
    for i, sequence in enumerate(data["sequences"]):
        if len(sequence) > limit:
            raise ValueError("Truncate before fitting preprocessing")
        transformed = sequence.copy()
        transformed[:, continuous] = seq_scaler.transform(sequence[:, continuous])
        tensor[i, -len(sequence):] = transformed
        mask[i, -len(sequence):] = True
    static = static_scaler.transform(data["static"])
    if not np.isfinite(static).all() or not np.isfinite(tensor).all():
        raise ValueError("Nonfinite transformed features")
    return dict(static=static, sequence=tensor, mask=mask, lengths=mask.sum(axis=1),
                preprocessors={"static": static_scaler, "sequence": seq_scaler,
                               "fit_ids": data["ids"][fit].tolist(), "continuous_sequence_columns": continuous})
