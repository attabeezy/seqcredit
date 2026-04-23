# DATACARD: Sequential Credit Risk Modeling Datasets

**Author:** Benjamin Ekow Attabra  
**Repository:** https://github.com/attabeezy/seqcredit-model  
**Version:** 1.0  
**Date:** April 2026  
**Framework:** Datasheets for Datasets

This card documents the two public synthetic datasets produced by this repository:

1. **Raw transaction dataset**: per-user synthetic mobile money transaction CSVs in `data/user_transactions/` plus user-level labels in `data/user_labels.csv`
2. **Engineered feature dataset**: user-level aggregate features in `data/user_features.csv`

Both datasets are fully synthetic and calibrated to public information about Ghanaian mobile money usage. No real persons or real transaction records are included.

This repository's public contribution is a **synthetic framework and documented dataset pipeline**, not a claim of production-ready credit-risk performance. Model comparisons in the repo are internal analytic checks used to inspect the synthetic environment and refine the framework before real-data work.

---

## Quick Reference

| | Dataset 1A: Raw Transactions | Dataset 1B: User Labels | Dataset 2: User Features |
|---|---|---|---|
| **File(s)** | `data/user_transactions/USER_*.csv` | `data/user_labels.csv` | `data/user_features.csv` |
| **Format** | CSV, one file per user | CSV | CSV |
| **Rows** | 1,030,198 transaction rows total | 10,000 users | 10,000 users |
| **Columns** | 13 to 19, depending on loan activity | 6 | 30 |
| **Unit of analysis** | Transaction | User | User |
| **Produced by** | `CalibratedMoMoDataGenerator` | `CalibratedMoMoDataGenerator` | `TemporalTransactionFeatureEngineer` |
| **Git tracked** | No | Yes | Yes |

---

## 1. Motivation and Scope

These datasets were created to support framework development, feature engineering research, and method prototyping for mobile money credit-risk modeling without requiring access to private financial records.

The intended contribution is methodological:
- a reproducible synthetic transaction generator
- a documented label-generation process
- a reusable feature engineering pipeline
- a clean bridge to later real-data validation

This dataset should be read as a synthetic research substrate, not as a substitute for external validation on real borrowers.

---

## 2. Dataset 1A: Raw Transaction Files

### Composition

- 10,000 synthetic users
- Mean of about 103 transactions per user
- 1,030,198 transaction rows total
- Files named `USER_000000.csv` through `USER_009999.csv`

### Core Schema

All files contain these 13 core columns:

| Column | Description |
|---|---|
| `TRANSACTION DATE` | Transaction timestamp |
| `FROM ACCT` | Sender account identifier |
| `FROM NAME` | Sender display name |
| `FROM NO.` | Sender phone number |
| `TRANS. TYPE` | Transaction type |
| `AMOUNT` | Transaction value in GHS |
| `FEES` | Transaction fee in GHS |
| `E-LEVY` | Electronic levy in GHS |
| `BAL BEFORE` | Balance before transaction |
| `BAL AFTER` | Balance after transaction |
| `TO NO.` | Recipient phone number |
| `TO NAME` | Recipient name |
| `TO ACCT` | Recipient account identifier |

Borrower files may also contain:

| Column | Description |
|---|---|
| `LOAN_PROVIDER` | Lending provider |
| `LOAN_PRINCIPAL` | Loan principal amount |
| `LOAN_INTEREST_RATE` | Loan interest rate |
| `LOAN_DUE_DATE` | Loan due date |
| `LOAN_PRINCIPAL_PAID` | Principal repaid |
| `LOAN_INTEREST_PAID` | Interest repaid |

### Variable Schema

The raw files intentionally have variable width:
- 13 columns for non-borrowers
- 17 columns for users with loan disbursement fields only
- 19 columns for users with repayment fields

Consumers should check column presence before accessing loan-specific fields.

### Transaction Types

Observed transaction types:
- `TRANSFER`
- `DEBIT`
- `PAYMENT`
- `PAYMENT_SEND`
- `CASH_OUT`
- `CASH_IN`
- `CREDIT`
- `LOAN_REPAYMENT`

`ADJUSTMENT` exists in the feature engineering code but is not generated in the current synthetic dataset.

### Temporal Coverage

- Observation window: `2024-01-01` to `2024-06-30`
- Per-user start offset: uniform from 0 to 14 days after start
- Inter-arrival process: gamma-distributed with user-specific variation
- Weekend inter-arrival times are compressed by a `x0.7` multiplier

### Amount and Fee Generation

- Base amount distribution: lognormal with parameters from `src/synthetic_params.json`
- Per-type multipliers shape transaction size by transaction category
- Fees and e-levy are applied probabilistically according to simple rules in `synthesize.py`

---

## 3. Dataset 1B: User Labels

### Schema

| Column | Description |
|---|---|
| `user_id` | User identifier |
| `credit_archetype` | Assigned behavioral archetype |
| `loans_taken` | Number of loans taken |
| `credit_risk_label` | Multiclass credit label |
| `final_credit_limit` | Final simulated credit limit |
| `gen_txn_count` | Generated transaction count |

### Credit Archetypes

The generator uses five borrower archetypes:

| Archetype | Target share | Description |
|---|---|---|
| `non_borrower` | 40% | Never takes loans |
| `responsible_borrower` | 35% | Regular loans, on-time repayment |
| `occasional_borrower` | 15% | Infrequent loans, variable timing |
| `risky_borrower` | 8% | Frequent loans, elevated default risk |
| `defaulter` | 2% | Persistent default behavior |

### Label Definition

`credit_risk_label` uses the following convention:

| Label | Meaning |
|---|---|
| `-1` | No loans |
| `0` | Good repayment |
| `1` | Late repayment |
| `2` | Default |

A user's final label is the worst outcome across all loans taken during the simulation.

For model training in this repo, the main binary target is:
- `default = 1` if `credit_risk_label == 2`
- `default = 0` otherwise

Non-borrowers are excluded from model training.

### Loan Parameters

Current synthetic loan settings include:
- minimum loan amount: `GHS 25`
- maximum loan amount: `GHS 1,000`
- interest rate: `6.9%`
- penalty rate: `12.5%`
- loan term: `30 days`
- initial credit limit: `GHS 50`
- credit limit growth multiplier: `x1.25`
- maximum credit limit: `GHS 1,000`

These are simulation parameters, not product guarantees.

---

## 4. Dataset 2: Engineered Feature Dataset

### Purpose

`data/user_features.csv` aggregates each user's transaction history into scalar features suitable for static models and general downstream analysis.

### High-Level Feature Groups

The user-level feature table includes features drawn from:
- transaction volume and amount statistics
- transaction type mix
- temporal behavior
- balance statistics
- fee behavior
- recipient diversity
- activity span

### Loan Features Not Stored in `user_features.csv`

Nine additional loan-related features are computed at training time by `CreditRiskDataLoader._engineer_loan_features()` from the raw transaction files. These include:
- total loan volume
- average loan amount
- maximum loan amount
- credit transaction share
- loan timing in sequence
- balance-at-loan statistics

This is why `user_features.csv` should be interpreted as part of a pipeline, not as the complete modeling feature set by itself.

---

## 5. Data Generation Process

### Generator

Primary class: `CalibratedMoMoDataGenerator` in `src/seqcredit_model/synthesize.py`

Default synthetic run:
- 10,000 users
- average 100 transactions per user
- 180-day simulation window
- reusable recipient pool

### Calibration

The generator reads parameters from `src/synthetic_params.json`. These parameters are informed by public reports, operator materials, and regulatory documents related to Ghanaian mobile money and digital lending.

Calibration is approximate, not exact:
- it uses public aggregate information, not user-level proprietary data
- some values are directly observed from product descriptions
- some are inferred from reported aggregates
- some remain designer assumptions where public data does not exist

### Reproducibility

- random seed: `42`
- run order: `synthesize.py` before `pipeline.py`
- if transactions are regenerated, cached LSTM arrays should be deleted before rebuilding sequence inputs

---

## 6. Preprocessing and Transformations

### Static Feature Pipeline

The static feature pipeline:
1. reads per-user transaction CSVs
2. engineers transaction-level features
3. aggregates them into user-level summaries
4. saves `data/user_features.csv`

### Sequence Pipeline

The sequence pipeline:
1. reads per-user transaction CSVs
2. engineers transaction-level features
3. selects the LSTM feature subset
4. pads sequences to fixed length
5. caches arrays to `data/lstm_sequences.npz`

### Split Logic

The repository uses user-level splitting. `prepare_static_splits()` must be called before `load_sequences()` so static and sequential models share the same user IDs.

---

## 7. Intended Uses

### Intended

- research and education on synthetic mobile money credit-risk data
- developing and testing feature engineering pipelines for financial time-series
- prototyping loaders, model interfaces, and evaluation workflows before real-data access
- internal diagnostic comparisons to understand what signals the synthetic framework exposes

### Out of Scope

- actual credit decisions
- representation of real Ghanaian users or real transaction logs
- claims of real-world predictive effectiveness without external validation

---

## 8. Ethical Considerations

- all data is synthetic
- no real persons, accounts, or phone numbers are represented
- borrower archetypes and class balance are modeling choices
- because behavior and labels are generated inside the same synthetic world, apparent predictive success can partly reflect simulator structure rather than economically meaningful signal
- if adapted to real deployment later, both false positives and false negatives would have asymmetric harms in a financial inclusion setting

---

## 9. Known Limitations and Threats to Validity

The main validity boundary is simple: this dataset is useful for building and auditing a synthetic framework, but it cannot establish real-world model effectiveness on its own.

Key threats to validity:
- structural coupling between simulated borrower behavior and simulated labels
- calibration to aggregate public statistics rather than user-level ground truth
- unobserved parameters that required designer assumptions
- possible mismatch between what is predictive in the simulator and what would be predictive in real transaction data
- no external validation against real borrower outcomes in this repository

Additional limitations:
- `ADJUSTMENT` is defined in feature engineering but not currently generated
- raw file schema varies by borrower state
- several lending parameters rely on public proxies or judgment calls because no public micro-level Ghana dataset exists

---

## 10. Distribution and License

- license: MIT
- repository: https://github.com/attabeezy/seqcredit-model
- tracked outputs: `data/user_labels.csv`, `data/user_features.csv`
- untracked generated outputs: `data/user_transactions/`

To regenerate:

```bash
python -m seqcredit_model.synthesize
python -m seqcredit_model.pipeline
```

---

## 11. Citation

```bibtex
@misc{attabra2026seqcredit,
  author       = {Attabra, Benjamin Ekow},
  title        = {Sequential Credit Risk Modeling: A Synthetic Framework for Mobile Money Data},
  year         = {2026},
  howpublished = {\url{https://github.com/attabeezy/seqcredit-model}},
  note         = {Synthetic mobile money data generation and feature-engineering framework}
}
```
