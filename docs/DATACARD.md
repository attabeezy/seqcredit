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

| Type | Semantic | Observed % | Calibration target |
|---|---|---|---|
| `TRANSFER` | Peer-to-peer transfer | 45.1% | 52.9% |
| `DEBIT` | Merchant payment / bill deduction | 23.1% | 27.2% |
| `CASH_IN` | Wallet top-up | 12.1% | — (added on top) |
| `PAYMENT` | Service/utility payment | 6.3% | 4.1% |
| `PAYMENT_SEND` | Payment sent to another party | 6.2% | 10.8% |
| `CASH_OUT` | Cash withdrawal | 4.4% | 5.0% |
| `CREDIT` | Loan disbursement | 1.5% | — (loan) |
| `LOAN_REPAYMENT` | Loan repayment | 1.4% | — (loan) |
| `ADJUSTMENT` | System correction | 0% (never generated) | — |

Calibration drift note: `CASH_IN` and loan types are injected on top of the five base types; the generator normalises only among {TRANSFER, DEBIT, PAYMENT, PAYMENT_SEND, CASH_OUT}.

### Temporal Coverage

| Property | Value |
|---|---|
| Observation window | 2024-01-01 to 2024-06-30 (180 days) |
| Account start offset | Uniform(0, 14) days after 2024-01-01 per user |
| Mean transactions per user | 103.1 (std 10.4, min 66, max 144) |
| Inter-arrival distribution | Gamma(shape=2, scale=~10 hours mean) |
| Weekend transaction rate | 30.2% (calibration target: 32.2%) |
| Night transaction rate (22:00+) | 7.9% (calibration target: 8.1%) |

Weekend inter-arrival times are compressed by a ×0.7 multiplier. Night-hour transactions use a preferred hour drawn from Beta-distributed user preference.

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

| Archetype | Count | Actual % | Target % | Description |
|---|---|---|---|---|
| `non_borrower` | 4,048 | 40.48% | 40% | Never takes loans |
| `responsible_borrower` | 3,481 | 34.81% | 35% | Regular loans, on-time repayment |
| `occasional_borrower` | 1,516 | 15.16% | 15% | Infrequent loans, variable timing |
| `risky_borrower` | 747 | 7.47% | 8% | Frequent loans, elevated default risk |
| `defaulter` | 208 | 2.08% | 2% | Persistent default behavior |

### Label Definition

`credit_risk_label` uses the following convention:

| Label | Meaning | Count | % of all users | % of borrowers |
|---|---|---|---|---|
| `-1` | No loans | 4,048 | 40.48% | — |
| `0` | Good — all loans repaid on time | 3,241 | 32.41% | 54.5% |
| `1` | Late — at least one loan repaid after term | 2,050 | 20.50% | 34.4% |
| `2` | Default — at least one loan not repaid | 661 | 6.61% | 11.1% |

A user's final label is the worst outcome across all loans taken during the simulation.

For model training in this repo, the main binary target is:
- `default = 1` if `credit_risk_label == 2`  → **661 positives out of 5,952 borrowers ≈ 11.1% default rate** (~8:1 class imbalance)
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

### Transaction-Level Feature Space (LSTM Inputs)

`extract_all_features()` produces **113 transaction-level features** before user-level aggregation:

| Category | Count | Example features |
|---|---|---|
| Amount transforms | 3 | `log_amount`, `sqrt_amount`, `amount_squared` |
| Amount size bins | 4 | `is_micro_txn` (<10), `is_small_txn` (10–50), `is_medium_txn` (50–200), `is_large_txn` (200+) |
| Fee features | 5 | `has_fees`, `has_elevy`, `total_cost`, `fee_to_amount_ratio`, `elevy_to_amount_ratio` |
| Transaction type one-hots | 7 | `is_transfer`, `is_debit`, `is_payment`, `is_payment_send`, `is_cash_out`, `is_cash_in`, `is_adjustment` |
| Temporal features | 13 | `hour`, `day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, time-of-day bins |
| Balance dynamics | 10 | `log_balance_before`, `balance_change`, `balance_pct_change`, `is_low_balance`, `will_deplete_balance` |
| Sequence position / cumulative | 6 | `time_since_last_txn_hours`, `txn_number`, `cumulative_volume`, `cumulative_fees_paid` |
| Count-based rolling (n=3,5,10) | 24 | `last_{n}_avg_amount`, `last_{n}_std_amount`, `last_{n}_transfer_count` |
| Time-based rolling (3d,7d,14d,30d) | 28 | `rolling_{w}_count`, `rolling_{w}_sum`, `rolling_{w}_balance_volatility` |
| Behavioural patterns | 4 | `unique_recipients_so_far`, `unique_txn_types_last_10`, `is_repeated_recipient` |
| Risk indicators | 8 | `unusual_hour`, `rapid_transaction`, `rapid_balance_drop`, `consecutive_withdrawals`, `risk_score` |

The LSTM input uses a 38-feature subset (`LSTM_FEATURE_COLUMNS` in `credit_model.py`). Sequences are pre-padded with zeros to `max_seq_len=50`. A Keras `Masking(mask_value=0.0)` layer ignores padded timesteps during training.

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

The generator reads parameters from `src/synthetic_params.json`, which contains two blocks:

**Block 1 — Transaction-level parameters:** Derived from public Ghanaian mobile money data (Bank of Ghana reports, GSMA publications, operator disclosures, 2023–2025).

**Block 2 — Loan-specific parameters:** Each value's provenance is documented below:

| Parameter | Value | Category | Source |
|---|---|---|---|
| `min_loan_amount_ghs` | 25 | Directly observed | MTN QwikLoan product terms [1] |
| `max_loan_amount_ghs` | 1,000 | Directly observed | MTN QwikLoan product terms [1] |
| `interest_rate_monthly` | 0.069 | Directly observed | QwikLoan 6.9% per 30-day term [1][4] |
| `interest_rate_monthly_alt` | 0.089 | Directly observed | XpressLoan/Telecel Ready Loan [4][5] |
| `processing_fee_rate` | 0.01 | Directly observed | XpressLoan 1% processing fee (Oct 2023) [4] |
| `late_penalty_rate` | 0.125 | Directly observed | QwikLoan / Telecel Ready Loan [1][5] |
| `loan_tenure_days` | 30 | Directly observed | QwikLoan, XpressLoan, Telecel Ready Loan [1][4][5] |
| `repayment_structure` | "lump_sum" | Directly observed | All major 30-day products [1][5] |
| `min_account_age_days_eligibility` | 90 | Directly observed | MTN QwikLoan eligibility [1][12] |
| `regulatory_max_loan_ghs` | 10,000 | Directly observed | BoG Directive for Digital Credit (Sep 2025) [13] |
| `implied_avg_loan_ghs` | 430 | Computed | GH¢2.8B ÷ 6.5M customers (Letshego H1 2024) [2] |
| `repeat_borrowing_avg_loans_per_customer` | 13.3 | Computed | 128M loans ÷ 9.6M customers (JUMO Ghana) [3] |
| `loan_amount_lognormal_mu` | 5.0 | Estimated | ln(GH¢150) ≈ 5.0; bounded by [25, 1000] range [2] |
| `loan_amount_lognormal_sigma` | 1.0 | Estimated | Spread consistent with GH¢25–1,000 product range [2] |
| `inter_loan_gap_median_days_estimate` | 45 | Estimated | Derived from 13.3 lifetime loans over ~7 years [3] |
| `borrower_prevalence_adult_pct` | 0.22 | Computed | World Bank Global Findex 2025 [11] |
| `female_borrower_share` | 0.34 | Computed | Letshego H1 2024 [2] |
| `default_rate_mean` | 0.06 | Bounded estimate | CGAP Ghana fieldwork (2020) [8]; JUMO cost of risk <4% [9] |
| `loan_tenure_distribution` | 88%/5%/7% | Designer estimate | QwikLoan dominance; XtraCash minor; Fido niche [2][6][7] |
| `approval_rate` | NaN | No data | Not published by any operator or regulator |
| `early_repayment_pct` | NaN | No data | Not published |
| `on_time_repayment_pct` | NaN | No data | Not published |
| `late_repayment_pct` | NaN | No data | GSMA qualitative only [10] |

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

| # | Limitation | Impact | Status |
|---|---|---|---|
| 1 | ~~`user_features.csv` was stale~~ | ~~High~~ | **RESOLVED** — Data regenerated with consistent seed |
| 2 | ~~No random seed in `synthesize.py`~~ | ~~Medium~~ | **RESOLVED** — `RANDOM_SEED = 42` added |
| 3 | `ADJUSTMENT` type never generated — `is_adjustment` feature is always 0 | Low | Open |
| 4 | Variable file schema (13, 17, or 19 columns) | Medium | By design |
| 5 | ~~Calibration source undocumented~~ | ~~Medium~~ | **RESOLVED** — §5 now documents all sources with citations |
| 6 | Binary collapse of 4-class label — labels 0 and 1 are merged as "non-default" | Medium | By design |
| 7 | 8:1 class imbalance among borrowers (11.1% default rate) | Medium | Handled via `class_weight='balanced'` |
| 8 | ~~`account_age_days` implausibly short in stale `user_features.csv`~~ | ~~High~~ | **RESOLVED** — Now mean 40.7 days |
| 9 | Loan amount distribution (`mu`, `sigma`) estimated from product bounds, not micro-level data | Medium | Open — no published loan-level dataset exists for Ghana |
| 10 | Default rate relies on provider self-reports, not independent verification | Medium | Open |
| 11 | Repayment timing split (early/on-time/late %) set to `NaN` — no published data | High | Open |
| 12 | Approval rate unknown — no published data | Medium | Open |
| 13 | Calibration based on aggregate public data — no micro-level validation | Medium | Open |

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

## 11. Calibration References

Sources used to derive `src/synthetic_params.json` parameters. All from publicly available reports, regulatory filings, and operator disclosures (2023–2025).

| Ref | Short ID | Full Citation |
|---|---|---|
| [1] | Asetenapa 2025 | Asetenapa.com. "How to Borrow Money From MTN Mobile Money?" Updated 2025. — QwikLoan: GH¢25 minimum, GH¢1,000 maximum, 6.9% monthly interest, 12.5% late penalty, 30-day tenure, 90-day eligibility. |
| [2] | Letshego H1 2024 | Letshego Ghana Ltd. H1 2024 Results, Ghana Stock Exchange "Facts Behind the Figures" session, July 31, 2024. — GH¢2.8B disbursed to 6.5M customers; 34% to women; loan purpose breakdown; QwikLoan = 60% of Letshego Ghana portfolio. |
| [3] | JUMO Ghana 2025 | JUMO World. "Financial inclusion in Africa: Ghana is leading." Corporate blog, 2025. — Cumulative: 128 million loans to 9.6 million customers in Ghana. |
| [4] | CitiNewsroom 2023 | Citi Newsroom. "MoMo loan: A complete rip-off…?" October 2023. — XpressLoan rate increase to 8.9% and 1% processing fee documented with SMS evidence. |
| [5] | Telecel Ghana | Telecel Ghana. "Ready Loan Terms and Conditions." — 8.9% monthly interest, 12.5% late penalty, 30-day tenure, 4–6 months active wallet required. |
| [6] | Asetenapa XtraCash | Asetenapa.com. "How to borrow money from MTN." April 2025. — XtraCash: GH¢50 maximum, 7-day tenure. |
| [7] | TechCrunch Fido 2024 | TechCrunch. "Impact investors FMO and BlueOrchard back Ghana's digital lender Fido in $30M Series B round." September 3, 2024. — Fido: $20–$500 range, up to 6 months repayment, 7–12% interest. |
| [8] | CGAP Ghana 2020 | Oppong, K. and Mattern, M. "African Digital Credit Goes West." CGAP Blog, January 2020. — Ghana digital credit providers report NPL rates "in the single digits." |
| [9] | JUMO/Orange 2025 | JUMO World and Orange Money Group. Press release, July 2025. — JUMO AI credit models achieve cost of risk below 4%. |
| [10] | GSMA 2024 | GSMA. Mobile Money Evaluation: Ghana. March 2025. — More than half of African borrowers struggle with loan repayments; loan repayments consume ~34% of monthly household income in Ghana. |
| [11] | World Bank Findex 2025 | World Bank. Global Findex Database 2025 (based on 2024 survey). — 22% of Ghanaian adults (~4.7M people) borrowed from mobile money; 74% of all formal borrowers. |
| [12] | Asetena Eligibility | Asetena.com. "How to qualify for MTN loan." July 2025. — QwikLoan eligibility: MTN subscriber >90 days, active MoMo wallet, no outstanding defaults, JUMO AI scoring (15,000+ predictive features). |
| [13] | BoG Directive 2025 | Bank of Ghana. "Directive for Digital Credit Services Providers." Notice No. BG/GOV/SEC/2025/30, September 23, 2025. — Minimum paid-up capital GH¢2M; maximum loan per customer GH¢10,000; compliance deadline June 30, 2026. |
| [14] | BoG FinTech 2024 | Bank of Ghana. "FinTech Sector Report 2024 FY." March 2025. — Monthly mobile money transaction values for 2024 (p. 8); used to compute seasonality index. |
| [17] | CGAP Kenya 2018 | Kaffenberger, M. and Chege, P. "A Digital Credit Revolution: Insights from Borrowers in Kenya and Tanzania." CGAP Working Paper, October 2018. — Proxy data: typical digital loan sizes $30–$50; 50% late repayment rate; 12% default rate (Kenya, not Ghana). |

---

## 12. Citation

```bibtex
@misc{attabra2026seqcredit,
  author       = {Attabra, Benjamin Ekow},
  title        = {Sequential Credit Risk Modeling: A Synthetic Framework for Mobile Money Data},
  year         = {2026},
  howpublished = {\url{https://github.com/attabeezy/seqcredit-model}},
  note         = {Synthetic mobile money data generation and feature-engineering framework}
}
```
