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


def sample_event_count(settings: dict, seed: int = 42, borrower: int = 0) -> int:
    """Draw heterogeneous activity independently of risk and outcome streams."""
    distribution = settings["count_distribution"]
    minimum = distribution["minimum_events"]
    shape = distribution["activity_shape"]
    mean = settings["expected_per_borrower"]
    if (distribution["family"] != "shifted_gamma_poisson" or
            not isinstance(minimum, int) or minimum < 1 or
            not np.isfinite([shape, mean]).all() or shape <= 0 or mean < minimum):
        raise ValueError("Invalid borrower activity distribution")
    rng = stream(seed, "activity", borrower)
    intensity = rng.gamma(shape, (mean - minimum) / shape)
    return minimum + int(rng.poisson(intensity))


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


def features(events: list[dict], categories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    history = sorted([row for row in events if row["phase"] == "background" and -90 <= row["day"] < 0],
                     key=lambda row: (row["day"], row["event_id"]))
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


def generate(contract: dict, seed: int = 42, event_sink=None,
             outcome_seed: int | None = None) -> dict:
    """Generate noisy histories before sampling future outcomes."""
    setting = contract["dataset"]
    n = setting["borrowers"]
    counts = contract["outcome"]["reference_counts"]
    prevalence = counts["no_repayment"] / contract["outcome"]["reference_denominator"]
    process = contract["history_process"]
    beta = process["risk_beta"]
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
    event_count = 0
    for i in range(n):
        uid, loan_id = f"SIM_{i:06d}", f"SIM_LOAN_{i:06d}"
        background, latent = stream(seed, "background", i), stream(seed, "latent", i)
        outcome = stream(seed if outcome_seed is None else outcome_seed, "outcome", i)
        future, principal_rng = stream(seed, "future_timing", i), stream(seed, "loan", i)
        z = float(latent.normal())
        count = sample_event_count(contract["background_events"], seed, i)
        days = np.sort(background.uniform(-90, 0, count))
        raw_amount = background.lognormal(np.log(amounts["median_units"]), amounts["log_sigma"], count)
        category_indices = background.choice(len(candidates), size=count, p=weights)
        incoming_uniform = background.random(count)
        counterparties = background.integers(0, directions["counterparty_pool_size"], count)
        history_rng = stream(seed, "history_process", i)
        level = process["amount_log_scale"] * z + history_rng.normal(0, process["borrower_level_noise"])
        trend = process["trend_log_scale"] * z + history_rng.normal(0, process["borrower_trend_noise"])
        time = (days + 45.) / 45.
        raw_amount *= np.exp(level + trend * time)
        values = np.round(np.clip(raw_amount, amounts["minimum_units"], amounts["maximum_units"]), 2)
        events = []
        for j in range(count):
            row = candidates[int(category_indices[j])]
            category = row["category"]
            events.append(dict(borrower_id=uid, event_id=f"{uid}_{j:04d}", day=float(days[j]),
                               phase="background", amount=float(values[j]), loan_id="", category=category,
                               raw_type=row["raw_type"], is_incoming=int(incoming_uniform[j] < directions["incoming_probability"][category]),
                               counterparty=f"{uid}_CP_{counterparties[j]:02d}"))
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
        static, sequence = features(events, categories)
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
                calibration=calibration, event_count=event_count)


def partitions(ids: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict:
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


def preprocess(data: dict, split: dict) -> dict:
    """Fit inner-training only; pack every real timestep without truncation."""
    fit = split["fit"]
    static_scaler = StandardScaler().fit(data["static"][fit])
    continuous = [0, 2, 3]
    real_fit = np.concatenate([data["sequences"][i][:, continuous] for i in fit])
    seq_scaler = StandardScaler().fit(real_fit)
    lengths = np.array([len(sequence) for sequence in data["sequences"]], dtype=np.int64)
    if np.any(lengths < 1):
        raise ValueError("Missing sequence history")
    offsets = np.r_[0, np.cumsum(lengths)]
    packed = np.empty((int(offsets[-1]), data["sequences"][0].shape[1]), dtype=np.float32)
    for i, sequence in enumerate(data["sequences"]):
        packed[offsets[i]:offsets[i + 1]] = sequence
        packed[offsets[i]:offsets[i + 1], continuous] = seq_scaler.transform(sequence[:, continuous])
    static = static_scaler.transform(data["static"])
    if not np.isfinite(static).all() or not np.isfinite(packed).all():
        raise ValueError("Nonfinite transformed features")
    return dict(static=static, sequence=packed, offsets=offsets, lengths=lengths,
                preprocessors={"static": static_scaler, "sequence": seq_scaler,
                               "fit_ids": data["ids"][fit].tolist(), "continuous_sequence_columns": continuous})


def sequence_batch(transformed: dict, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Right-pad only to this batch's longest history; never infer masks from values."""
    lengths = transformed["lengths"][indices]
    width = int(lengths.max())
    values = np.zeros((len(indices), width, transformed["sequence"].shape[1]), dtype=np.float32)
    mask = np.zeros((len(indices), width), dtype=bool)
    for row, index in enumerate(indices):
        start, end = transformed["offsets"][index:index + 2]
        length = int(lengths[row])
        values[row, :length] = transformed["sequence"][start:end]
        mask[row, :length] = True
    return {"sequence": values, "valid_mask": mask}
