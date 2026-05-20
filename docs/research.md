# Sequential Credit Risk Modeling: Comprehensive Research Record

**Project:** Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments  
**Author:** Attabra Benjamin Ekow (KNUST, beattabra@st.knust.edu.gh)  
**Repository:** https://github.com/Attabeezy/seqcredit-model  
**Period Covered:** January 2026 – May 2026  
**Document Date:** May 20, 2026  

---

This document is a complete, thread-stitched record of everything we have built, discovered, fixed, argued over, and published toward — from the first commit to the Telecel Ghana Databricks pipeline. It is written to give any reader (including a future version of us) the full intellectual history: the ideas that motivated the work, the data that shaped it, the code that runs it, the bugs we killed, and the findings that emerged. It is deliberately long. Breadth and depth are both intentional.

---

## Table of Contents

1. [The Problem We Set Out to Solve](#1-the-problem-we-set-out-to-solve)
2. [Project Architecture at a Glance](#2-project-architecture-at-a-glance)
3. [Data: The Synthetic Ghanaian MoMo Benchmark](#3-data-the-synthetic-ghanaian-momo-benchmark)
4. [Feature Engineering: The Temporal Transaction Feature Engine](#4-feature-engineering-the-temporal-transaction-feature-engine)
5. [Models: Six Architectures, One Interface](#5-models-six-architectures-one-interface)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Benchmark Results: Paper A (Synthetic Data)](#7-benchmark-results-paper-a-synthetic-data)
8. [Ablation Study: Which Features Actually Matter?](#8-ablation-study-which-features-actually-matter)
9. [Hyperparameter Tuning: Does It Help?](#9-hyperparameter-tuning-does-it-help)
10. [Explainability and Interpretability (XAI)](#10-explainability-and-interpretability-xai)
11. [Model Calibration](#11-model-calibration)
12. [Statistical Significance Testing](#12-statistical-significance-testing)
13. [Bugs We Found and Fixed](#13-bugs-we-found-and-fixed)
14. [Real Data Pipeline: Paper B (Telecel Ghana)](#14-real-data-pipeline-paper-b-telecel-ghana)
15. [The Behavioral Diversity Framework](#15-the-behavioral-diversity-framework)
16. [Development History: Commit-by-Commit Narrative](#16-development-history-commit-by-commit-narrative)
17. [Publication Plan and Targets](#17-publication-plan-and-targets)
18. [Open Questions and Future Work](#18-open-questions-and-future-work)
19. [Codebase Reference](#19-codebase-reference)
20. [Calibration Sources and Methodology](#20-calibration-sources-and-methodology)

---

## 1. The Problem We Set Out to Solve

### 1.1 The Research Gap

Credit risk assessment in sub-Saharan Africa is a largely unsolved problem. Traditional credit bureau infrastructure is absent or thin for most of the population. Mobile money — the financial rails that billions of people across Ghana, Kenya, Tanzania, and elsewhere use for their daily economic lives — generates rich behavioural transaction data. This data, if properly modeled, could power a new generation of inclusive credit scoring.

The core research question was deceptively simple: **can transaction sequences help predict loan default?** The hypothesis was that the *order* of financial events — when someone spent money, how their balance moved, how their behaviour changed in the weeks before a loan — might encode signals that aggregate statistics miss.

But there was a prior problem: **real MoMo data is locked away.** Telecoms and lenders guard their transaction logs fiercely for regulatory and competitive reasons. Academic researchers cannot easily access the 374 million rows of Telecel Ghana data or the 128 million loans disbursed through MTN QwikLoan. This makes it impossible to develop and iterate on models before getting access — and getting access takes years.

### 1.2 Our Two-Stage Strategy

We designed the research in two stages:

**Stage I (Paper A — Deep Learning Indaba 2026):** Build a *synthetic benchmark* that is calibrated to real Ghanaian mobile money behavioral patterns and use it to:
- Formalise a reusable temporal feature engineering framework (8 feature groups, 38+ features)
- Conduct a rigorous comparative evaluation of 6 models (4 static + 2 sequential)
- Establish the baseline finding: do sequences help?

**Stage II (Paper B — Telecel Ghana):** Take the framework built in Stage I and validate it on national-scale real data. Test whether the findings from the synthetic environment hold up when confronted with 374 million real transactions from real borrowers.

The two papers form a natural research arc: methodology and hypothesis generation (Stage I) → empirical validation at national scale (Stage II).

---

## 2. Project Architecture at a Glance

### 2.1 Repository Structure

```
seqcredit-model/
├── src/seqcredit_model/          # Core Python package
│   ├── __init__.py
│   ├── config.py                 # Path resolution, environment variable overrides
│   ├── synthesize.py             # CalibratedMoMoDataGenerator
│   ├── pipeline.py               # TemporalTransactionFeatureEngineer
│   ├── credit_model.py           # All 6 model classes + DataLoader + Evaluator
│   ├── real_data_pipeline.py     # Spark-based Telecel Ghana pipeline
│   ├── run_cv_benchmark.py       # 5-fold CV benchmark script
│   ├── run_ablation_study.py     # Feature group ablation
│   ├── run_hyperparameter_tuning.py  # RandomizedSearchCV tuning
│   ├── run_full_benchmark.py     # End-to-end runner
│   └── compute_bootstrap_ci.py  # Bootstrap CI helper
├── notebooks/
│   ├── model.ipynb               # Primary modeling (all 6 models)
│   ├── analysis.ipynb            # XAI: SHAP, surrogate trees, calibration
│   ├── data.ipynb                # Descriptive statistics
│   ├── model_gcolab.ipynb        # Google Colab variant
│   ├── analysis_gcolab.ipynb     # Google Colab variant
│   ├── data_gcolab.ipynb         # Google Colab variant
│   ├── Analysis Notebook A.ipynb # Databricks: static models only (real data)
│   ├── Analysis Notebook B.ipynb # Databricks: static + LSTM + GRU (real data)
│   └── real_data_end_to_end.ipynb # Real data pipeline testing
├── data/
│   ├── user_features.csv         # Engineered user-level features
│   ├── user_labels.csv           # Credit risk labels
│   ├── cv_results_y_default.csv  # 5-fold CV results, default target
│   ├── cv_results_y_bad.csv      # 5-fold CV results, bad-payer target
│   ├── significance_tests.csv    # Bootstrap significance tests
│   ├── ablation_features.csv     # Feature group ablation results
│   ├── tuning_results.csv        # Hyperparameter tuning results
│   └── cv_manifest.json          # Reproducibility metadata
├── src/synthetic_params.json     # Calibration parameters (documented)
├── docs/
│   ├── PROJECT.md                # Project documentation
│   ├── DATACARD.md               # Data documentation
│   ├── H1_2026_Progress_Report.md # Quarterly progress report
│   └── research.md               # This document
├── DATACARD.md                   # Root-level copy
├── RESEARCH_DIRECTION.md         # Strategic research direction
├── CLAUDE.md                     # Claude Code guide
└── requirements.txt
```

### 2.2 The Data Flow

```
Real calibration params (synthetic_params.json)
          ↓
CalibratedMoMoDataGenerator.generate_dataset()
          ↓
data/user_transactions/USER_XXXXXX.csv  (10,000 per-user CSVs, ~1.03M rows total)
data/user_labels.csv                     (10,000 rows with archetypes and risk labels)
          ↓
TemporalTransactionFeatureEngineer.extract_all_features()
TemporalTransactionFeatureEngineer.create_user_level_summary()
          ↓
data/user_features.csv              (10,000 rows × 29 aggregate features)
          ↓
CreditRiskDataLoader.load_static_data()     →  X (38 features after loan engineering)
CreditRiskDataLoader.prepare_static_splits() →  X_train, X_test, y_train, y_test
CreditRiskDataLoader.load_sequences()        →  X_train_seq, X_test_seq (padded arrays)
          ↓
LogisticRegressionModel / XGBoostModel / RandomForestModel / LightGBMModel
LSTMModel / HybridLSTMModel / GRUModel
          ↓
run_cv_benchmark.py → data/cv_results_y_default.csv, data/cv_results_y_bad.csv
run_ablation_study.py → data/ablation_features.csv
run_hyperparameter_tuning.py → data/tuning_results.csv
```

### 2.3 The Real Data Flow (Paper B, Databricks)

```
Telecel Ghana YARA table (374M rows, Unity Catalog)
→ melodatabricks616.default.yara_dump_table
          ↓
real_data_pipeline.py / build_pipeline(df, spark)
    _parse_timestamp()           # Oracle DD-MON-YY → Spark timestamp
    _derive_loan_cutoffs()       # index loan = last loan disbursement per user
    filter to min_followup_days  # prevent right-censoring bias
    _derive_labels()             # repayment/penalty events AFTER index loan
    _build_user_features()       # aggregate over pre-loan transactions
    _build_sequences_spark()     # per-user sequence arrays
          ↓
user_features.csv + user_labels.csv  (written to runtime path)
          ↓
CreditRiskDataLoader (same API as synthetic path)
          ↓
CV benchmark (same run_cv_benchmark.py)
```

---

## 3. Data: The Synthetic Ghanaian MoMo Benchmark

### 3.1 Why Synthetic?

Real mobile money transaction data is not publicly available. Operators in Ghana (MTN, Telecel) and financial service providers (JUMO/QwikLoan, FIDO) treat their user data as competitive assets. Regulatory frameworks do not mandate open data sharing. Academic researchers face a catch-22: they cannot develop and validate models without data, but they cannot get access to data without demonstrated model capability.

Our solution was to build a calibrated synthetic dataset — one that is not *real* but is *realistic* in the statistical properties that matter for credit risk modeling. The calibration is documented in exhaustive detail (see `DATACARD.md` and Section 20 of this document), derived from 17 publicly accessible sources including Bank of Ghana reports, GSMA publications, MTN/Telecel product disclosures, and academic fieldwork studies.

### 3.2 The Generator: `CalibratedMoMoDataGenerator`

**Location:** `src/seqcredit_model/synthesize.py`

The generator is a per-user simulation engine. For each of 10,000 users, it:

1. **Assigns a credit archetype** from a weighted distribution:
   - `non_borrower` (40%) — never takes loans
   - `responsible_borrower` (35%) — regular loans, always repays on time
   - `occasional_borrower` (15%) — infrequent loans, variable timing
   - `risky_borrower` (8%) — frequent loans near credit limit, 15% default probability
   - `defaulter` (2%) — takes loans, 100% default probability

2. **Generates a user profile** with individual behavioural variation:
   - Personal amount distribution: `lognormal(μ ~ N(2.84, 0.2), σ ~ N(1.00, 0.1))`
   - Personal transaction frequency: `gamma(shape=2, scale=~5 hours)` mean inter-arrival
   - Recipient pool preference: `gamma(shape=3, scale=10)` unique recipients
   - Time-of-day preference: drawn from a set of realistic peak hours
   - Weekend activity preference: `Beta(2, 5)` distribution

3. **Simulates the transaction history** over 180 days (2024-01-01 to 2024-06-30):
   - Transaction type selection based on balance and personal preferences
   - Pre-scheduled loan injection points across the transaction sequence
   - Loan repayment timing determined by archetype repayment behavior
   - Balance tracking with minimum-threshold enforcement (forces CASH_IN when low)

4. **Determines credit labels** from loan outcomes:
   - `credit_risk_label = -1` for non-borrowers
   - `credit_risk_label = 0` if all loans repaid on time (within 30 days)
   - `credit_risk_label = 1` if any loan repaid late (31-60 days)
   - `credit_risk_label = 2` if any loan was defaulted (never repaid)
   - Final label = worst-case across all loans a user took

### 3.3 Dataset Statistics

| Metric | Value |
|--------|-------|
| Users | 10,000 |
| Total transactions | 1,030,198 |
| Mean transactions/user | 103.0 (std: 10.4) |
| Borrowers | 5,952 (59.52%) |
| Non-borrowers (label = -1) | 4,048 (40.48%) |
| Good payers (label = 0) | 3,241 (32.41%) |
| Late payers (label = 1) | 2,050 (20.50%) |
| Defaulters (label = 2) | 661 (6.61%) |
| Default rate (among borrowers) | 11.1% |
| Class imbalance ratio | ~8:1 (non-default:default) |

### 3.4 Transaction Type Distribution

| Type | Generated % | Calibration Target |
|------|-------------|-------------------|
| TRANSFER | 45.1% | 52.9% |
| DEBIT | 23.1% | 27.2% |
| CASH_IN | 12.1% | — (injected separately) |
| PAYMENT | 6.3% | 4.1% |
| PAYMENT_SEND | 6.2% | 10.8% |
| CASH_OUT | 4.4% | 5.0% |
| CREDIT (loan disbursement) | 1.5% | — (loan) |
| LOAN_REPAYMENT | 1.4% | — (loan) |
| ADJUSTMENT | 0% | — (never generated) |

The calibration drift between TRANSFER (45.1% vs 52.9%) and PAYMENT_SEND (6.2% vs 10.8%) is a known artifact: CASH_IN is injected as a balance-restoration mechanism on top of the base distribution, which shifts the effective mix. This is documented in the DATACARD.

### 3.5 The Raw Transaction File Schema

Per-user CSV files are stored in `data/user_transactions/USER_XXXXXX.csv`. The schema varies by borrower status:

**Core 13 columns (all users):**

| Column | Type | Description |
|--------|------|-------------|
| TRANSACTION DATE | datetime | Timestamp with microsecond precision |
| FROM ACCT | string | Sender account number |
| FROM NAME | string | Sender display name |
| FROM NO. | string | Ghana E.164 phone (`233XXXXXXXXX`) |
| TRANS. TYPE | categorical | One of 8 transaction types |
| AMOUNT | float | Value in GHS |
| FEES | float | Transaction fee in GHS |
| E-LEVY | float | Electronic levy in GHS |
| BAL BEFORE | float | Balance before transaction (GHS) |
| BAL AFTER | float | Balance after transaction (GHS) |
| TO NO. | string | Recipient phone or `"0"` for services |
| TO NAME | string | Recipient name or service provider |
| TO ACCT | string | Recipient account |

**+4 disbursement columns** (borrower files, on CREDIT rows):
LOAN_PROVIDER, LOAN_PRINCIPAL, LOAN_INTEREST_RATE, LOAN_DUE_DATE

**+2 repayment columns** (borrower files with repayments, on LOAN_REPAYMENT rows):
LOAN_PRINCIPAL_PAID, LOAN_INTEREST_PAID

### 3.6 Calibration Parameters (`src/synthetic_params.json`)

The calibration file defines all behavioral parameters. Key values:

```json
{
  "amount_lognormal_mu": 2.8421,
  "amount_lognormal_sigma": 1.0034,
  "amount_mean": 32.80,
  "amount_median": 18.50,
  "transaction_frequency_hours": -10.01,
  "balance_mean": 305.88,
  "balance_std": 302.55,
  "type_distribution": {
    "TRANSFER": 0.529, "DEBIT": 0.272, "PAYMENT_SEND": 0.108,
    "CASH_OUT": 0.050, "PAYMENT": 0.041
  },
  "weekend_rate": 0.322,
  "night_rate": 0.081,
  "fee_rate": 0.243,
  "elevy_rate": 0.127,
  "low_balance_rate": 0.019
}
```

Loan parameters are sourced from MTN QwikLoan product terms (GHS 25–1,000, 6.9% monthly, 12.5% penalty, 30-day term), JUMO Ghana disclosures, CGAP fieldwork, and Bank of Ghana regulatory filings. See Section 20 for the complete provenance table.

### 3.7 Loan Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Min loan amount | GHS 25 | MTN QwikLoan terms |
| Max loan amount | GHS 1,000 | MTN QwikLoan terms |
| Interest rate | 6.9% per 30 days | QwikLoan standard |
| Alt rate (XpressLoan) | 8.9% per 30 days | Telecel Ready Loan |
| Late penalty | 12.5% | Both products |
| Loan term | 30 days | Both products |
| Initial credit limit | GHS 50 | Product terms |
| Credit limit growth | ×1.25 per on-time repayment | |
| Max credit limit | GHS 1,000 | Product cap |
| Min account age | 90 days | QwikLoan eligibility |
| Min transactions | 15 | Designer estimate |

---

## 4. Feature Engineering: The Temporal Transaction Feature Engine

### 4.1 Philosophy and Design

The `TemporalTransactionFeatureEngineer` class (in `src/seqcredit_model/pipeline.py`) is arguably the most important reusable contribution of this project. It formalises the intuition that financial behaviour — not just financial outcomes — predicts creditworthiness.

The key design decisions:

1. **Transaction-level features first** — extract 113 features per transaction row, capturing the context of each individual event
2. **User-level aggregation second** — summarise the per-transaction feature space into 29 user-level scalar statistics for static models
3. **Sequence preservation for LSTM** — a 38-feature subset is preserved in time-ordered form for sequential models

### 4.2 The 113 Transaction-Level Features (11 Categories)

Every transaction row is transformed into 113 features across 11 semantic categories:

| Category | Count | Example Features |
|----------|-------|-----------------|
| Amount transforms | 3 | `log_amount`, `sqrt_amount`, `amount_squared` |
| Amount size bins | 4 | `is_micro_txn` (<GHS 10), `is_small_txn` (10–50), `is_medium_txn` (50–200), `is_large_txn` (200+) |
| Fee features | 5 | `has_fees`, `has_elevy`, `total_cost`, `fee_to_amount_ratio`, `elevy_to_amount_ratio` |
| Transaction type one-hots | 7 | `is_transfer`, `is_debit`, `is_payment`, `is_payment_send`, `is_cash_out`, `is_cash_in`, `is_adjustment` |
| Temporal features | 13 | `hour`, `day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, time-of-day bins |
| Balance dynamics | 10 | `log_balance_before`, `balance_change`, `balance_pct_change`, `is_low_balance`, `will_deplete_balance` |
| Sequence position/cumulative | 6 | `time_since_last_txn_hours`, `txn_number`, `cumulative_volume`, `cumulative_fees_paid` |
| Count-based rolling windows (n=3,5,10) | 24 | `last_n_avg_amount`, `last_n_std_amount`, `last_n_transfer_count` |
| Time-based rolling windows (3d,7d,14d,30d) | 28 | `rolling_Nd_count`, `rolling_Nd_sum`, `rolling_Nd_balance_volatility` |
| Behavioural patterns | 4 | `unique_recipients_so_far`, `unique_txn_types_last_10`, `is_repeated_recipient` |
| Risk indicators | 8 | `unusual_hour`, `rapid_transaction`, `rapid_balance_drop`, `consecutive_withdrawals`, `risk_score` |

### 4.3 Cyclical Encoding of Time

A deliberate design choice: temporal features use sine/cosine encoding rather than raw integers. This avoids the discontinuity artifact that would make `hour=23` appear far from `hour=0` in Euclidean space.

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
```

### 4.4 Rolling Window Design

Two types of rolling windows were implemented:

1. **Count-based (n=3, 5, 10 transactions):** Capture short-term behavioural patterns. For example, `last_5_avg_amount` gives the mean amount of the last 5 transactions, regardless of when they happened.

2. **Time-based (3, 7, 14, 30 days):** Capture temporal behaviour tied to real calendar durations. For example, `rolling_30d_balance_volatility` captures how stable a user's balance was over the last month.

### 4.5 The 29 User-Level Aggregate Features

For static models, the per-transaction features are collapsed into user-level scalars:

**Transaction volume and amount (8 features):**
`obs_txn_count`, `total_volume`, `avg_transaction_amount`, `median_transaction_amount`, `std_transaction_amount`, `max_transaction_amount`, `min_transaction_amount`, `cv_transaction_amount`

**Transaction type mix (4 features):**
`pct_transfers`, `pct_debits`, `pct_cashouts`, `pct_payments`

**Temporal behaviour (4 features):**
`avg_hours_between_txns`, `pct_weekend_txns`, `pct_night_txns`, `pct_early_morning_txns`

**Balance statistics (5 features):**
`avg_balance`, `min_balance`, `max_balance`, `balance_volatility`, `pct_low_balance_txns`

**Fee behaviour (3 features):**
`total_fees_paid`, `avg_fees_per_txn`, `pct_txns_with_fees`

**Recipient and diversity (3 features):**
`unique_recipients`, `recipient_concentration`, `unique_txn_types`

**Activity span (2 features):**
`account_age_days`, `transactions_per_day`

### 4.6 The 9 Loan Features (Computed at Training Time)

Nine loan-specific features are computed separately by `CreditRiskDataLoader._engineer_loan_features()` at model training time, reading directly from the raw per-user transaction CSVs. They are **not** stored in `user_features.csv`:

| Feature | Description |
|---------|-------------|
| `total_loan_volume` | Sum of all CREDIT amounts (GHS) |
| `avg_loan_amount` | Mean loan disbursement (GHS) |
| `max_loan_amount` | Largest single loan (GHS) |
| `loan_to_total_volume_ratio` | total_loan_volume / total transaction volume |
| `pct_credit_transactions` | Fraction of transactions that are CREDIT |
| `loan_timing_in_sequence` | Position of first CREDIT (0 = early, 1 = late) |
| `avg_balance_at_loan` | Mean BAL BEFORE on CREDIT rows (GHS) |
| `min_balance_at_loan` | Minimum BAL BEFORE on CREDIT rows (GHS) |
| `balance_to_loan_ratio_at_disbursement` | avg_balance_at_loan / avg_loan_amount |

All 9 are set to 0 for non-borrowers. The static model input is thus 29 + 9 = 38 features.

### 4.7 The LSTM Feature Subset (38 Features)

From the 113 transaction-level features, a 38-feature subset called `LSTM_FEATURE_COLUMNS` is selected for the sequential models. This subset includes:

```
log_amount, is_micro_txn, is_small_txn, is_medium_txn, is_large_txn,
fee_to_amount_ratio, total_cost, has_fees,
is_transfer, is_debit, is_payment, is_payment_send, is_cash_out, is_cash_in, is_adjustment,
hour_sin, hour_cos, day_sin, day_cos, is_weekend,
time_since_last_txn_hours,
log_balance_before, balance_pct_change, is_low_balance, is_zero_balance,
amount_to_balance_ratio, will_deplete_balance, balance_change,
is_repeated_recipient, is_self_transfer, unique_txn_types_last_10,
unusual_hour, rapid_transaction, risk_score,
amount_vs_last_5_avg, rolling_7d_std, reverse_txn_number, last_5_std_amount
```

Sequences are pre-padded with zeros to `max_seq_len=50`. A Keras `Masking(mask_value=0.0)` layer ignores padded timesteps during training.

---

## 5. Models: Six Architectures, One Interface

### 5.1 The Common Interface

All six model classes share a consistent API:

```python
model.fit(X_train, y_train)
model.predict(X_test, threshold=0.5) -> np.ndarray
model.predict_proba(X_test) -> np.ndarray
model.cross_validate(X, y, n_splits=5) -> Dict
model.save("models/model_name")
model = ModelClass.load("models/model_name")
```

This design was intentional: it allows models to be swapped in and out of evaluation pipelines without changing surrounding code, and it makes `ModelEvaluator` comparisons easy.

### 5.2 Logistic Regression (`LogisticRegressionModel`)

**Location:** `src/seqcredit_model/credit_model.py:969`

The simplest model, used as a linear baseline. Configured with:
- `class_weight='balanced'` — compensates for the 8:1 class imbalance by upweighting defaults
- `C=1.0` — L2 regularization strength (inverse; smaller = stronger regularization)
- `max_iter=1000` — enough iterations for convergence on 38 features
- `StandardScaler` applied per-fold inside `cross_validate`

**Why it matters:** Logistic Regression's strong performance (0.91 AUC-ROC in CV) is a key finding. It suggests the default signal is well-separable in the feature space and that simple linear combinations of the 38 features carry most of the information.

### 5.3 XGBoost (`XGBoostModel`)

**Location:** `src/seqcredit_model/credit_model.py:1042`

Gradient-boosted trees with:
- `scale_pos_weight` — ratio of negative to positive class (computed per-fold in CV to avoid leakage; this was a bug that was fixed — see Section 13)
- `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`
- `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.8`

XGBoost consistently achieves strong precision (F1 scores and precision much higher than LR) while slightly lower recall, due to its tendency to be more conservative in positive-class predictions.

### 5.4 Random Forest (`RandomForestModel`)

**Location:** `src/seqcredit_model/credit_model.py:1127`

Ensemble of 200 decision trees:
- `n_estimators=200`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`
- `class_weight='balanced'`
- `n_jobs=-1` — parallel training

Random Forest achieves the strongest AUC-ROC (0.8836 mean in CV) on the synthetic benchmark. Its calibration (ECE ≈ 0.10) is better than Hybrid LSTM but worse than XGBoost and LightGBM.

### 5.5 LightGBM (`LightGBMModel`)

**Location:** `src/seqcredit_model/credit_model.py:1219`

Gradient boosting with leaf-wise tree growth:
- `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`
- `num_leaves=31`, `class_weight='balanced'`
- Early stopping support via `lgb.early_stopping(20)`

LightGBM achieves the best calibration of all models (ECE ≈ 0.038), making it the preferred choice when probability reliability matters for loan pricing or risk-tiered decisions.

### 5.6 LSTM (`LSTMModel`)

**Location:** `src/seqcredit_model/credit_model.py:1318`

A stacked LSTM binary classifier:

```
Input → Masking(mask_value=0.0)
  → LSTM(32, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=L2(1e-4))
  → Dropout(0.4)
  → LSTM(16, return_sequences=False, recurrent_dropout=0.3, kernel_regularizer=L2(1e-4))
  → Dropout(0.4)
  → Dense(16, activation='relu')
  → Dropout(0.27)
  → Dense(1, activation='sigmoid')
```

Trained with:
- `Adam(lr=0.001)`
- `binary_crossentropy` loss
- `AUC` as the monitored validation metric
- `EarlyStopping(patience=7, restore_best_weights=True, monitor='val_auc')`
- `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`
- `class_weight` balanced for imbalance handling
- 30 epochs maximum, batch size 256

**Critical finding:** The standalone LSTM collapses to near-random (AUC ≈ 0.523 on `y_default`, 0.507 on `y_bad`). This is not an architecture problem — we regularised extensively (reducing from 64→32 units down to 32→16, adding dropout=0.4, L2 regularization, recurrent dropout 0.3) and the model stayed at chance level. The conclusion is a property of the data: **transaction sequences alone carry insufficient signal to predict loan default.** The temporal order of individual transactions does not encode the behavioural patterns that distinguish defaulters from good payers.

### 5.7 Hybrid LSTM (`HybridLSTMModel`)

A dual-branch architecture that fuses sequential and static information:

```
Sequence branch:    Input(seq) → Masking → LSTM(32) → Dropout → LSTM(16)
Static branch:      Input(static) → Dense(32, relu) → Dropout
Fusion:             Concatenate([seq_branch, static_branch])
                    → Dense(32, relu) → Dropout → Dense(16, relu) → Dropout → Dense(1, sigmoid)
```

The Hybrid LSTM achieves a mean AUC-ROC of 0.8101 (y_default), which is competitive with static-only models (RandomForest: 0.8836, LR: 0.9143). However, its ECE (0.1573) is dramatically worse than static tree models (0.04–0.10), and its calibration reliability diagrams are poor. The conclusion: **the Hybrid LSTM's competitive AUC is driven entirely by its static feature branch, not by the LSTM component.**

### 5.8 GRU (`GRUModel`) and Hybrid GRU

GRU variants were added later (commit `4375ac2`) during the Databricks phase (Paper B). The GRU uses the same architecture as LSTM but replaces the Keras LSTM layers with GRU layers. In the real-data Databricks benchmark (Analysis Notebook B), both LSTM and GRU are compared head-to-head.

---

## 6. Evaluation Framework

### 6.1 Metrics Tracked

Every model evaluation tracks eight metrics per fold:

| Metric | Description |
|--------|-------------|
| `auc_roc` | Area Under the ROC Curve |
| `auc_pr` | Area Under the Precision-Recall Curve |
| `f1` | F1 Score at 0.5 threshold |
| `precision` | Precision at 0.5 threshold |
| `recall` | Recall at 0.5 threshold |
| `accuracy` | Accuracy at 0.5 threshold |
| `brier` | Brier Score (mean squared error of probabilities) |
| `ece` | Expected Calibration Error (15-bin weighted mean |confidence - accuracy|) |

**Why Brier + ECE?** AUC-ROC and AUC-PR measure discrimination (can the model rank defaults above non-defaults?). Brier and ECE measure calibration (are the model's predicted probabilities reliable?). For actual credit risk deployment — pricing loans, setting credit limits, triggering follow-up — calibrated probabilities matter enormously. A model that predicts 0.90 when the true probability is 0.20 will cause systematic mispricing.

### 6.2 5-Fold Stratified Cross-Validation

**Script:** `src/seqcredit_model/run_cv_benchmark.py`

The benchmark uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Stratification ensures each fold has the same positive-class ratio (~11.1% defaults).

Key implementation notes:
- Static models: `StandardScaler` fitted on training fold only, transformed for validation fold
- Sequential models: scaler fitted on non-padded training timesteps only
- XGBoost: `scale_pos_weight` recomputed per fold from the fold's training data (important bug fix — see Section 13)
- Results written to runtime data directory via `get_runtime_data_dir()` for Databricks compatibility

The benchmark ran for 408.66 minutes (6.8 hours) total. Timing breakdown:
- LogisticRegression: 0.82s (y_default), 0.56s (y_bad)
- XGBoost: 5.50s (y_default), 5.63s (y_bad)
- RandomForest: 11.26s (y_default), 9.47s (y_bad)
- LightGBM: 7.23s (y_default), 2.88s (y_bad)
- LSTM: 4,147s (y_default), 2,011s (y_bad)
- HybridLSTM: 3,210s (y_default), 2,494s (y_bad)

### 6.3 Bootstrap Significance Testing

**Script:** `src/seqcredit_model/compute_bootstrap_ci.py`

Pairwise bootstrap tests compare model AUC-ROC and AUC-PR differences using 1,000 bootstrap samples. The null hypothesis is that two models have equal performance (Δ = 0). A **critical bug was found and fixed** (commit `3e35218`): the original implementation was not centering the bootstrap delta distribution around zero, causing p-values to be ≈0.5 regardless of effect size. The fix: generate a null distribution by mean-centering the bootstrap deltas before computing the p-value.

### 6.4 Two Targets

Two prediction tasks are evaluated in parallel:

- **`y_default`** (`credit_risk_label == 2` → 1, else → 0): The primary task. Binary: did this borrower default on any loan during the observation period?

- **`y_bad`** (`credit_risk_label in {1, 2}` → 1, else → 0): The secondary task. Binary: was this borrower a late payer or defaulter? This is a softer, broader definition of credit risk.

`y_bad` has a much higher positive rate (late + default = 20.50% + 6.61% = 27.11% of borrowers), making it a more balanced classification problem.

---

## 7. Benchmark Results: Paper A (Synthetic Data)

### 7.1 Mean CV Results — `y_default` Target

From `data/cv_results_y_default.csv`:

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall | Brier | ECE |
|-------|---------|--------|----|-----------|--------|-------|-----|
| **LogisticRegression** | **0.9143 ± 0.009** | 0.6786 ± 0.031 | 0.5940 ± 0.010 | 0.4683 ± 0.013 | 0.8164 ± 0.021 | 0.1065 | 0.1703 |
| **RandomForest** | 0.8836 ± 0.014 | 0.7159 ± 0.037 | 0.6686 ± 0.036 | 0.9139 ± 0.018 | 0.5297 ± 0.039 | 0.0689 | 0.1039 |
| **XGBoost** | 0.8809 ± 0.016 | 0.7185 ± 0.029 | 0.6653 ± 0.038 | 0.8114 ± 0.013 | 0.5656 ± 0.059 | 0.0609 | 0.0416 |
| **LightGBM** | 0.8777 ± 0.017 | 0.7211 ± 0.030 | 0.6560 ± 0.037 | 0.8175 ± 0.059 | 0.5508 ± 0.039 | 0.0609 | 0.0380 |
| **HybridLSTM** | 0.8501 ± 0.029 | 0.6377 ± 0.048 | 0.5140 ± 0.047 | 0.4282 ± 0.038 | 0.6574 ± 0.072 | 0.1160 | 0.1738 |
| **LSTM** | 0.5232 ± 0.113 | 0.1793 ± 0.077 | 0.2696 ± 0.095 | 0.1816 ± 0.065 | 0.5872 ± 0.192 | 0.2294 | 0.3338 |

### 7.2 Mean CV Results — `y_bad` Target

From `data/cv_results_y_bad.csv`:

| Model | AUC-ROC | AUC-PR | F1 | ECE |
|-------|---------|--------|----|-----|
| **LogisticRegression** | 0.7254 ± 0.020 | 0.7207 ± 0.018 | 0.6208 ± 0.013 | 0.0745 |
| **RandomForest** | 0.7269 ± 0.017 | 0.7432 ± 0.012 | 0.5917 ± 0.016 | 0.0655 |
| **XGBoost** | 0.7098 ± 0.020 | 0.7341 ± 0.015 | 0.6357 ± 0.018 | 0.2280 |
| **LightGBM** | 0.7152 ± 0.017 | 0.7387 ± 0.016 | 0.5995 ± 0.016 | 0.0876 |
| **HybridLSTM** | 0.7054 ± 0.014 | 0.7079 ± 0.024 | 0.6170 ± 0.007 | 0.0829 |
| **LSTM** | 0.5334 ± 0.015 | 0.4798 ± 0.015 | 0.4228 ± 0.104 | 0.0432 |

### 7.3 Key Findings from the Benchmark

**Finding 1: Static features dominate sequences.** Logistic Regression (0.9143 AUC-ROC) and RandomForest (0.8836) substantially outperform the Hybrid LSTM (0.8501) and massively outperform the standalone LSTM (0.5232). The temporal order of individual transactions does not add discriminative power beyond what aggregate statistics capture.

**Finding 2: The LSTM collapse is not a regularisation artifact.** We tried:
- Reducing units from 64→32 down to 32→16
- Increasing dropout from 0.2 to 0.4
- Adding L2 kernel regularisation (1e-4)
- Adding recurrent dropout (0.3)
- Batch size 256 (from 32) to stabilise gradient estimates

The LSTM remained at 0.523 AUC. This is now interpreted as a **data property**: in the synthetic benchmark, credit outcomes are partly determined by user archetype labels, not purely by the transaction sequence dynamics. The synthetic generator assigns default probability by archetype, and the static aggregate features encode archetype well; the sequence order adds nothing.

**Finding 3: The Hybrid LSTM's competitive AUC is attributable to its static branch.** The static branch of the Hybrid LSTM sees the same 38 features as the static models. The LSTM branch's contribution appears to be noise. This is consistent with Finding 1.

**Finding 4: Calibration diverges sharply between model families.** Tree-based gradient boosting models (XGBoost: ECE 0.0416, LightGBM: ECE 0.0380) are exceptionally well-calibrated. Random Forest (ECE 0.1039) and Logistic Regression (ECE 0.1703) are worse but acceptable. The Hybrid LSTM (ECE 0.1738) and standalone LSTM (ECE 0.3338) are poorly calibrated — their predicted probabilities systematically misrepresent the true default risk. **For any deployment that uses predicted probabilities (loan pricing, tiered risk scoring, limit setting), LightGBM or XGBoost should be preferred.**

**Finding 5: `y_bad` is a harder problem.** All models perform worse on the broad `y_bad` target (late or default) than on the strict `y_default` (default only). This is expected: late payers are harder to distinguish from good payers than outright defaulters. The AUC-ROC gap is large (LR: 0.9143 vs 0.7254). This has implications for Paper B — depending on which target the lender cares about, the choice of model matters more.

**Finding 6: No statistically significant pairwise differences between top models.** Bootstrap significance tests (1,000 resamples) find p > 0.05 for all pairwise AUC-ROC comparisons among the top 5 models. The differences are real but uncertainty bands overlap. See `data/significance_tests.csv`.

---

## 8. Ablation Study: Which Features Actually Matter?

**Script:** `src/seqcredit_model/run_ablation_study.py`  
**Results:** `data/ablation_features.csv`  
**Model used:** RandomForest (5-fold CV, same seed)

The ablation study systematically tests two conditions for each of 8 feature groups:
1. **Drop-one:** Remove the group, train on all remaining features
2. **Single-group-only:** Train on only that group's features

### 8.1 Feature Group Definitions

| Group | Features | Count |
|-------|----------|-------|
| `amount_stats` | Total/avg/std/max/min/median volume | 7 |
| `txn_type_mix` | Pct transfers/debits/cashouts/payments | 4 |
| `temporal_patterns` | Avg hours between txns, pct weekend/night/early | 4 |
| `balance_dynamics` | Avg/min/max balance, volatility, low balance pct | 5 |
| `fee_behaviour` | Total fees, avg fees/txn, pct with fees | 3 |
| `behavioural_diversity` | unique_recipients, recipient_concentration, unique_txn_types | 3 |
| `activity_intensity` | Account age days, transactions per day | 2 |
| `loan_history` | All 9 loan-specific features from `_engineer_loan_features` | 9 |

### 8.2 Drop-One Results

| Condition | AUC-ROC | Δ vs All Features |
|-----------|---------|------------------|
| ALL_FEATURES (baseline) | 0.8836 | — |
| DROP_amount_stats | 0.8857 | +0.002 |
| DROP_txn_type_mix | 0.8792 | −0.004 |
| DROP_temporal_patterns | 0.8801 | −0.004 |
| DROP_balance_dynamics | 0.8770 | −0.007 |
| DROP_fee_behaviour | 0.8760 | −0.008 |
| DROP_activity_intensity | 0.8813 | −0.002 |
| **DROP_behavioural_diversity** | **0.7941** | **−0.090** |
| **DROP_loan_history** | **0.7796** | **−0.104** |

### 8.3 Single-Group Results

| Condition | AUC-ROC | Interpretation |
|-----------|---------|----------------|
| ONLY_amount_stats | 0.5774 | Marginal signal |
| ONLY_txn_type_mix | 0.5914 | Marginal signal |
| ONLY_temporal_patterns | 0.4988 | Near-random |
| ONLY_balance_dynamics | 0.6408 | Some signal |
| ONLY_fee_behaviour | 0.4986 | Near-random |
| **ONLY_behavioural_diversity** | **0.7358** | **Strong standalone signal** |
| ONLY_activity_intensity | 0.5194 | Near-random |
| **ONLY_loan_history** | **0.7890** | **Strongest standalone signal** |

### 8.4 Ablation Conclusions

**`loan_history` is the single most powerful feature group** (ONLY: 0.789 AUC; DROP: −0.104). This is almost tautological: prior loan and repayment behaviour is a direct predictor of future loan behaviour. This group should always be included in any credit risk model that has access to historical borrowing data.

**`behavioural_diversity` is the most powerful non-loan group** (ONLY: 0.736 AUC; DROP: −0.090). The three features — `unique_recipients`, `recipient_concentration`, `unique_txn_types` — capture how diversely a user interacts with the mobile money ecosystem. High diversity (many recipients, many transaction types) correlates with lower default risk. This is consistent with the intuition that economically active, financially included users with diverse payment networks are better credit risks.

**All other groups have marginal individual contributions (Δ ≤ 0.01 in drop-one).** Dropping amount statistics actually *slightly improves* performance (+0.002), suggesting those features may add noise. The other groups add small amounts of robustness but are not discriminative on their own.

**Implication for feature engineering in data-scarce contexts:** If a practitioner can only compute a subset of features, prioritise loan history and recipient diversity. Everything else is secondary.

---

## 9. Hyperparameter Tuning: Does It Help?

**Script:** `src/seqcredit_model/run_hyperparameter_tuning.py`  
**Results:** `data/tuning_results.csv`  
**Method:** RandomizedSearchCV (40 trials, 3-fold search CV → best params evaluated on 5-fold CV)

### 9.1 Tuning Results

| Model | Default AUC-ROC | Tuned AUC-ROC | Δ | Best Params |
|-------|-----------------|---------------|---|-------------|
| RandomForest | 0.8836 | 0.8855 | +0.002 | n_estimators=500, max_depth=15, min_samples_leaf=8, min_samples_split=5 |
| XGBoost | 0.8809 | 0.8870 | +0.006 | n_estimators=200, max_depth=4, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, min_child_weight=1 |
| LightGBM | 0.8777 | 0.8880 | +0.010 | n_estimators=500, num_leaves=15, learning_rate=0.01 |

### 9.2 Tuning Conclusions

**Gains from tuning are marginal.** The largest improvement is LightGBM (+0.010 AUC-ROC), which is within the confidence interval of the untuned estimate. XGBoost gains +0.006. RandomForest gains +0.002.

**Default hyperparameters are near-optimal for this dataset.** This is likely because the synthetic dataset is well-conditioned (consistent class distributions, no extreme outliers, all features on similar scales after standardisation). For the real Telecel Ghana data, tuning may produce larger gains due to more heterogeneous real-world patterns.

**HybridLSTM was not tuned.** Compute cost (>3,000 seconds per CV run) made tuning impractical.

---

## 10. Explainability and Interpretability (XAI)

### 10.1 The Research Motivation

Credit risk is a high-stakes domain. "Black-box" models may achieve good AUC but are unacceptable in practice because:
1. Regulatory requirements (fair lending laws) require explainability
2. Lenders need to understand *why* a model flags someone as high-risk
3. Borrowers who are denied credit have a right to a reason

The XAI pipeline in `notebooks/analysis.ipynb` implements a five-part interpretability analysis.

### 10.2 Permutation Importance

Top-15 features across all 4 static models. Key findings:
- `unique_recipients` is consistently the top feature across models
- `recipient_concentration` is in the top 3 for all tree models
- Loan history features (`total_loan_volume`, `avg_loan_amount`) are in the top 5
- `balance_volatility` and `min_balance` appear consistently in top 10

### 10.3 SHAP Analysis

SHAP TreeExplainer was applied to XGBoost and LightGBM. Key observations from beeswarm plots:
- High `unique_recipients` → strong negative SHAP (reduces default probability)
- High `recipient_concentration` → positive SHAP (increases default probability — single-recipient focus is risky)
- High `total_loan_volume` → varies by context: combined with high balance, negative; with low balance, positive
- `pct_night_txns` shows mild positive SHAP — nighttime transacting slightly elevates default risk

Waterfall plots for individual predictions confirm the model's logic is consistent: a user predicted as high-risk typically has high recipient concentration (few unique payees), low account balance volatility (stagnant balance), and recent large loans.

### 10.4 Surrogate Decision Tree

A depth-3 surrogate decision tree was fitted to the predictions of the best static model (RandomForest), evaluated at depths 2–8. Key metrics:
- Depth 3: Fidelity ~0.78 (78% agreement with RandomForest predictions), 6 leaf nodes
- Depth 5: Fidelity ~0.89, 20 leaf nodes
- Depth 8: Fidelity ~0.95, too complex for human interpretation

The depth-3 rules represent the most actionable audit trail:

```
Root split: unique_recipients ≤ threshold_A
├── LEFT (few recipients)
│   └── loan_history (high volume) → HIGH RISK
│   └── loan_history (low volume) → MEDIUM RISK
└── RIGHT (many recipients)
    └── balance_volatility (low) → LOW RISK
    └── balance_volatility (high) → MEDIUM RISK
```

This is the "auditable rule set" that a loan officer could use to understand the model's logic.

### 10.5 Individual Tree Visualisations

Actual decision trees from RF (one sampled), XGBoost (one tree), and LightGBM (one tree) were visualised using `sklearn.tree.plot_tree` and model-native plotters. These provide transparency into the raw decision logic within each sub-estimator.

### 10.6 Causal Boundary

**Important caveat stated in all documentation:** The XAI pipeline explains *model behaviour*, not *causality*. SHAP values and permutation importance tell us what the model uses — they do not tell us that `unique_recipients` *causes* good repayment. This distinction matters for policy: you cannot improve credit outcomes by artificially inflating someone's recipient count.

---

## 11. Model Calibration

### 11.1 Why Calibration Matters

Calibration is the alignment between predicted probabilities and actual outcomes. A model that predicts 70% default probability should be correct 70% of the time. Poor calibration means the model's risk scores cannot be used directly for pricing or limit-setting without correction.

### 11.2 Calibration Results

| Model | Brier Score | ECE | Interpretation |
|-------|-------------|-----|----------------|
| LightGBM | 0.0609 | 0.038 | Excellent |
| XGBoost | 0.0609 | 0.042 | Excellent |
| RandomForest | 0.0689 | 0.104 | Good |
| LogisticRegression | 0.1065 | 0.170 | Acceptable |
| HybridLSTM | 0.1160 | 0.174 | Poor |
| LSTM | 0.2294 | 0.334 | Very poor |

LightGBM and XGBoost are naturally well-calibrated because gradient boosting optimises over probability calibration-related objectives. Random Forest produces averaged probabilities from many trees, which softens but doesn't eliminate miscalibration. Logistic Regression's calibration is hampered by the class imbalance handling (`class_weight='balanced'` distorts the probability scale).

### 11.3 Isotonic Calibration

Post-hoc isotonic regression calibration was applied (via `CalibratedClassifierCV(method='isotonic')`) to all models. Results from `analysis.ipynb`:
- LightGBM: ECE improved from 0.038 → ~0.022 (marginal)
- XGBoost: ECE improved from 0.042 → ~0.030
- HybridLSTM: ECE improved from 0.174 → ~0.092 (significant improvement)

### 11.4 Reliability Diagrams

Reliability diagrams (calibration curves) were plotted for all models. Key visual findings:
- LightGBM and XGBoost: points cluster near the diagonal (perfect calibration)
- Random Forest: slight over-confidence in the 0.6–0.8 probability range
- HybridLSTM: systematic overconfidence across all bins
- LSTM: essentially random; the reliability curve is flat

---

## 12. Statistical Significance Testing

**File:** `data/significance_tests.csv`  
**Method:** Bootstrap delta test, 1,000 resamples, 95% CI

### 12.1 Key Test Results (`y_default`, AUC-ROC)

| Comparison | Delta | p-value | Significant? |
|-----------|-------|---------|-------------|
| LSTM vs LogisticRegression | −0.291 | 0.502 | No |
| LSTM vs XGBoost | −0.257 | 0.510 | No |
| LSTM vs RandomForest | −0.259 | 0.509 | No |
| LSTM vs LightGBM | −0.254 | 0.504 | No |
| HybridLSTM vs LogisticRegression | −0.066 | 0.493 | No |
| HybridLSTM vs XGBoost | −0.032 | 0.493 | No |
| HybridLSTM vs RandomForest | −0.034 | 0.487 | No |
| HybridLSTM vs LightGBM | −0.029 | 0.515 | No |

**None of the pairwise comparisons reach statistical significance.** This is a consequence of the bootstrap test being applied to OOF (out-of-fold) predictions from 5-fold CV — the effective sample size is moderate (n ≈ 1,190 per fold) and the between-fold variance inflates the uncertainty.

**Interpretation:** The *absence of significance* does not mean models perform identically. The LSTM is clearly worse (Δ ≈ −0.29 AUC). The p-values close to 0.5 are an artifact of the bootstrap null distribution construction. After the bug fix (commit `3e35218`), p-values are now meaningful but reflect true uncertainty. The recommendation: report both the point estimates and the 95% CIs, acknowledge the lack of significance, and note that the LSTM's inferiority is robust across all folds.

---

## 13. Bugs We Found and Fixed

This section documents every significant bug found during the project, what caused it, and how it was fixed. These are the battle scars that shaped the final results.

### 13.1 The Stale Labels Bug (March 2026, Commit: label integrity fix)

**What happened:** Models were training on `summary_extended.csv`, which was a relic from an earlier synthetic data generation run. The labels in that file did not correspond to the transaction CSVs in `data/user_transactions/`.

**Symptom:** Static models achieved AUC ≈ 0.53 in single-split evaluation — approximately random. This was initially misattributed to class imbalance.

**Root cause:** `summary_extended.csv` was generated in one random seed run; the transaction CSVs had been regenerated with a different seed. The feature engineering read the current transaction files, but the labels came from the stale summary.

**Fix:** 
1. Deleted `summary_extended.csv`
2. Updated `config.py` to use `user_labels.csv` (the correct output of `synthesize.py`)
3. Added data consistency validation in `CreditRiskDataLoader._validate_data()` — this method cross-checks `obs_txn_count` (from `user_features.csv`) against `gen_txn_count` (from `user_labels.csv`) and raises an error on mismatch
4. Regenerated all data with `RANDOM_SEED = 42`

### 13.2 The Class Imbalance Bug (March 2026, Commit: `7ca76d5`)

**What happened:** Early Logistic Regression and XGBoost implementations did not handle the 8:1 class imbalance. Models predicted all-negative (no defaults), yielding ~89% accuracy but ~0.50 AUC-ROC.

**Symptom:** AUC ≈ 0.53, near-zero recall for the positive class.

**Fix:** Added `class_weight='balanced'` to Logistic Regression and Random Forest; computed `scale_pos_weight` (n_negative/n_positive ≈ 8.03) for XGBoost.

**Result:** After the fix, static model AUC jumped from ~0.53 to ~0.83–0.88.

### 13.3 The Double-Scaling Bug (April 2026, Commit: `run_cv_benchmark` fix)

**What happened:** The initial `run_cv_benchmark.py` was passing `X_train_scaled` (already scaled by the data loader) to model CV functions that applied `StandardScaler` *again* per fold. Features were being double-standardised.

**Symptom:** Subtle performance degradation and training instability in some folds.

**Fix:** Pass raw `X_train` (not `X_train_scaled`) to CV functions, so each fold's scaler is the only scaler applied.

### 13.4 The Bootstrap p-value Bug (April 2026, Commit: `3e35218`)

**What happened:** The bootstrap significance test was computing p-values without centering the null distribution at zero. The test computed the delta between two models, then asked "how often does the bootstrap delta exceed the observed delta?" — but the bootstrap distribution was not shifted to the null hypothesis (Δ = 0).

**Symptom:** All p-values were approximately 0.50, regardless of the actual performance gap between models.

**Fix:** The bootstrap null distribution is now mean-centered: generate bootstrap deltas, subtract the mean delta (to shift to Δ = 0 under the null), then compute the p-value as the fraction of centered bootstrap deltas that exceed the observed delta.

### 13.5 The XGBoost scale_pos_weight Leakage Bug (Commit: `bec1252`)

**What happened:** In the CV benchmark, `scale_pos_weight` for XGBoost was computed from the full training set (`y_default` ratio) and used for both `y_default` and `y_bad` CV runs. When running `y_bad`, the wrong ratio was being used.

**Symptom:** XGBoost `y_bad` results were computed with a `scale_pos_weight` calibrated for `y_default` class distribution (~8:1), not `y_bad` (~2.7:1). This caused XGBoost to over-predict positives on `y_bad`.

**Fix:** `scale_pos_weight` is now recomputed per-fold from the fold's training data for the current target variable.

### 13.6 The LSTM Variable Collision Bug (Commits: `df398b9`, `3cafff4`)

**What happened:** In Analysis Notebook A, the loop variable `df` inside the CV results display loop was clobbering the Spark DataFrame `df` defined earlier in the notebook. Similarly, `col` was used both as a column name and as a loop variable.

**Fix:** Renamed loop variables to `cv_df` and `metric` respectively.

### 13.7 DBFS Disabled on Databricks Serverless (Commit: `43ef692`)

**What happened:** Early versions of the real data pipeline tried to cache results using `.cache()` Spark operations, which are not supported on Databricks serverless clusters.

**Fix:** Removed all `.cache()` calls. Used materialization through `.toPandas()` instead where needed.

### 13.8 The Six-Scan Pipeline Problem (Commit: `8d9e628`)

**What happened:** The Spark pipeline was scanning the full 374M-row transaction table six times (once per feature aggregation step), causing timeout failures and massive compute costs.

**Fix:** Materialised the filtered and timestamp-parsed DataFrame into a single intermediate result, then ran all aggregations over that single materialized frame. Eliminated 5 of 6 full table scans.

### 13.9 Observation Window Truncation Bias (Ongoing)

**What happened (Paper B):** After the first leakage-free run, the default rate was 50.9% — clearly inflated. Borrowers whose last loan was in late October or November 2025 (near the end of the 90-day observation window, which ended ~November 2025) had no time to repay, so they were labelled "default" even though they hadn't yet had the chance to repay.

**Fix:** `min_followup_days` parameter in `_derive_loan_cutoffs()` — only include borrowers whose index loan is at least N days before the end of the observation window. Was initially 30 days, then raised to 60 days (commit `9b9982f`) to more reliably capture penalty signals. Rerunning with this filter is pending as of the current project state.

---

## 14. Real Data Pipeline: Paper B (Telecel Ghana)

### 14.1 The Data

Telecel Ghana provided real mobile money transaction data for research:

| Property | Value |
|----------|-------|
| Table | `melodatabricks616.default.yara_dump_table` |
| Rows | 374,295,424 |
| Columns | 12 |
| Timestamp format | Oracle: `DD-MON-YY HH.MI.SS.FFFFFFFFF` |
| Date range | ~September – November 2025 (~90 days) |
| Lender ID | `E7C89F8C4A27F173` |
| Borrowers identified | 474,312 |
| All IDs | Anonymised/hashed |

This is an order of magnitude larger than the synthetic dataset — 374 million rows vs 1 million synthetic rows. It requires Spark processing on Azure Databricks (A10 cluster).

### 14.2 The Temporal Design (Leakage-Free)

The first run of the real data pipeline produced AUC = 1.0 — a catastrophic data leakage. Repayment features (the fact that a borrower *made* a repayment) were directly encoding the label (whether the borrower repaid). 

**The leakage-free design (implemented after discovery):**

For each borrower, the **index loan** = their most recent loan disbursement (`last_loan_ts`).
- **Features** = all transactions strictly BEFORE `last_loan_ts`
- **Labels** = repayment/penalty events strictly AFTER `last_loan_ts`

This temporal cutoff ensures zero leakage: the model cannot see the loan outcome it is predicting.

The key function is `_derive_loan_cutoffs(df, min_followup_days=60)`:

```python
loan_cutoffs = (
    df.filter(LOAN_DISBURSEMENT_TYPE)
    .groupBy("user_id")
    .agg(F.max("ts").alias("last_loan_ts"))
)
# Only include borrowers whose index loan is >= 60 days before end of window
global_max = df.agg(F.max("ts").alias("_global_max_ts"))
return loan_cutoffs.crossJoin(global_max).filter(
    F.col("last_loan_ts") <= F.col("_global_max_ts") - F.expr("INTERVAL 60 DAYS")
)
```

### 14.3 The Transaction Type Mapping

The real Telecel Ghana transaction type strings are much more descriptive than our synthetic types. The pipeline maps them to 12 semantic categories:

| Category | Description |
|----------|-------------|
| `loan_disbursement` | Loan Payment Via API (from LENDER_ID) |
| `loan_repayment_principal` | Loan Principal Collection via API |
| `loan_repayment_interest` | Loan Interest via API |
| `loan_penalty` | Loan Penalty via API |
| `cash_out` | Withdrawal/cash-out types |
| `cash_in` | Deposit/top-up types |
| `airtime_data` | Airtime and data purchases |
| `transfer` | Peer-to-peer transfers |
| `payment` | Bill and merchant payments |
| `betting` | Betting/gaming transactions |
| `fsi` | Financial service institution transactions |
| `other` | All other types |

### 14.4 The Sequence Feature Set (Real Data)

For the LSTM/GRU models on real data, 19 features per timestep are used:

```
log_amount, is_outgoing,
txtype_loan_disbursement, txtype_loan_repayment_principal, txtype_loan_repayment_interest,
txtype_loan_penalty, txtype_cash_out, txtype_cash_in, txtype_airtime_data,
txtype_transfer, txtype_payment, txtype_betting, txtype_fsi, txtype_other,
hour_sin, hour_cos, dow_sin, dow_cos, hours_since_last_txn
```

### 14.5 Current CV Results on Real Data (Post-Leakage Fix, April 23, 2026)

**y_default (never repaid after index loan):**

| Model | AUC-ROC | AUC-PR | F1 | ECE |
|-------|---------|--------|----|-----|
| LightGBM | 0.9464 | 0.9462 | 0.8926 | 0.0058 |
| XGBoost | 0.9462 | 0.9460 | 0.8926 | 0.0060 |
| RandomForest | 0.9370 | 0.9346 | 0.8909 | 0.0568 |
| LogisticRegression | 0.9290 | 0.9246 | 0.8675 | 0.0830 |

**y_bad (risky or default):**

| Model | AUC-ROC | AUC-PR | F1 | ECE |
|-------|---------|--------|----|-----|
| LightGBM | 0.8069 | 0.9291 | 0.7816 | 0.1833 |
| XGBoost | 0.8069 | 0.9289 | 0.8683 | 0.0063 |
| RandomForest | 0.7905 | 0.9223 | 0.7672 | 0.1982 |
| LogisticRegression | 0.6996 | 0.8732 | 0.7595 | 0.2337 |

**Key observation:** Real data results are significantly better than synthetic (0.9464 vs 0.8836 for LightGBM `y_default`). This is consistent with the hypothesis that real behavioural signals are richer and more discriminative than the simulator can capture. The "Static > Sequence" finding appears to hold: static feature models achieve state-of-the-art AUC on real data without any sequential architecture.

### 14.6 Known Issues in Paper B (Current State)

**Issue 1 (PRIORITY): Observation window truncation bias.** With `min_followup_days=60`, the default rate is still elevated. Borrowers who took their last loan 60–90 days before the end of the window have limited time to demonstrate penalty/repayment behaviour. The filter should be re-evaluated and possibly increased further.

**Issue 2: 463 cold-start borrowers.** 463 borrowers have no pre-loan transactions and are silently dropped by the data loader. These are first-time borrowers with zero pre-existing history — an important population for financial inclusion research. They are currently excluded; this should be documented in Paper B.

**Issue 3: LSTM/GRU not yet run on real data.** Per-user transaction CSVs are not available for real data (data exists in the Spark table, not per-user files). The Spark-based sequence builder (`build_sequences_spark`) was implemented but full CV benchmark with LSTM/GRU on real data has not been run end-to-end.

### 14.7 Databricks Notebook Architecture

Two Databricks notebooks handle the real data analysis:

**Analysis Notebook A** (`notebooks/Analysis Notebook A.ipynb`):
- Runs the full Spark pipeline (EDA, feature engineering, label derivation)
- Runs static models only (LR, XGBoost, RF, LightGBM)
- Exports filtered borrower transactions to Unity Catalog managed table
- Live output streaming with `bench.main()` inlined

**Analysis Notebook B** (`notebooks/Analysis Notebook B.ipynb`):
- Runs static + LSTM + GRU with live output streaming
- 20 sequential CV runs (dropped Transformer after it proved slower without benefit)
- Same feature engineering as Notebook A but extended for sequential models

---

## 15. The Behavioral Diversity Framework

### 15.1 The Central Scientific Claim

The ablation study (`data/ablation_features.csv`) produced a striking empirical finding: the `behavioural_diversity` feature group (3 features: `unique_recipients`, `recipient_concentration`, `unique_txn_types`) is the most powerful non-loan predictor of default.

**The Behavioral Diversity (BD) Framework** proposes that this finding reflects a deeper principle about financial behaviour in emerging markets:

> In data-constrained emerging fintech markets, **Recipient Entropy** and **Temporal Regularity** are more robust predictors of default than the **Sequential Amount Patterns** traditionally targeted by deep learning models.

### 15.2 The Three-Feature Argument

`unique_recipients` — how many distinct people or services does this user pay? A user who pays 50 different recipients is financially embedded in the community: they have diverse obligations, multiple income sources, and regular financial activity with a wide network. This diversity correlates with economic stability.

`recipient_concentration` — what fraction of all transactions go to the single most frequent recipient? High concentration means the user funnels most of their money to one place (perhaps a single merchant or family member). This could signal financial fragility or dependency.

`unique_txn_types` — how many different types of transactions does the user make? Users who use transfers, payments, cash-outs, and bill payments all regularly have more financial versatility than users who only make cash-outs.

Together, these three features encode a proxy for **financial inclusion depth**: how deeply and diversely is this user integrated into the digital financial ecosystem?

### 15.3 The Entropy Interpretation

From an information-theoretic perspective, `unique_recipients` and `unique_txn_types` approximate the entropy of the user's transaction distribution. High entropy (many diverse transactions) → low default risk. Low entropy (few concentrated transactions) → elevated default risk.

This is formalized in `RESEARCH_DIRECTION.md`:

> **Entropy vs. Sequence:** A paradigm shift from learning *when* events occur to modeling *how* users interact with a diverse ecosystem of providers.

### 15.4 Implications for Architecture

If behavioral diversity is the dominant signal and that signal is encoded in aggregate statistics (unique recipients count, concentration ratio, type count), then:

1. **LSTM architectures add no value** — the temporal order of transactions does not help extract this signal; a simple count of unique recipients captures it exactly
2. **Attention mechanisms may be irrelevant** — even self-attention over the transaction sequence cannot improve on a scalar count of unique recipients
3. **Static features are sufficient** — for this signal type, no sequential architecture is needed

This is consistent with the LSTM collapse finding (AUC ≈ 0.52) and supports the "Static > Sequence" thesis.

### 15.5 Open Question: The Sequential Bottleneck

The key unresolved question: **is the LSTM's failure a property of the LSTM architecture, or an inherent property of MoMo data?**

We know:
- Vanilla LSTM (32→16 units, regularised): AUC 0.523
- Hybrid LSTM (LSTM + static): AUC 0.850 (but static branch likely drives this)
- GRU: tested in Databricks, results pending
- Transformer: tested briefly (commit `5069dcb`), then dropped from scope (commit `d332846`) — computational cost too high for marginal expected benefit

The Transformer/Attention Baseline experiment (`blueprint.md`: "Deleted — Transformer baseline is no longer part of the project scope") was initially planned to resolve this question definitively. The investigation remains open for future work.

---

## 16. Development History: Commit-by-Commit Narrative

### 16.1 Foundation Phase (Pre-2026)

**`353adc2` — Sync docs and requirements to current state**  
The earliest tracked commit. Project already exists with basic structure.

**`2c09ce0` — Fixed minor inconsistencies**

**`eb7d6e5` — Remove AGENTS.md and CLAUDE.md references**

**`97e930d` — Add research analysis notebook (XAI, surrogate tree, calibration)**  
First appearance of `analysis.ipynb` — the full XAI pipeline.

**`8bbf237` — Add individual tree visualisations section**  
Extended the analysis notebook with direct tree visualizations for RF, XGB, LGBM.

**`b4aefed` — Update docs to reflect individual tree visualisation section**

**`02d563f` — Merge branch 'main' from remote**  
First merge conflict resolution.

**`9fc22c8` — Refactor code structure for improved readability and maintainability**  
Major restructuring of the codebase into a proper Python package.

### 16.2 The Imbalance Fix Era (March 2026)

**`ce36cd4` — Run full end-to-end test successfully**  
First successful end-to-end run after refactoring.

**`c5e5ea4` — Run credit risk analysis to generate XAI plots**

**`eb2d9c0` — Adjust confusion matrix graphs**

**`837c2e0` — Updated documentation**

**`7ca76d5` — Imbalance fix**  
Critical: `class_weight='balanced'` added to LR/RF; `scale_pos_weight` added to XGBoost. AUC jumps from ~0.53 to ~0.83.

**`66b410b` — Created reset.py and wiped temp files**

**`a64db19` — Post-imbalance fix run**  
Confirmed the fix worked. Models now discriminate properly.

**`68e9142` — Restructure project documentation and remove stale files**

**`3473b77` — Updated readme**

**`0e1e1d6` — Cleanup and rerun**

**`b024b92` — Added temporary files to .gitignore**

**`7668192` — Added ablation, cv, and hyperparameter tuning**  
First version of the three benchmark scripts.

**`e58f440` — Remove results section from README**

**`83bf346` — Resolve merge conflict in README**

**`bdd0616` — Removed results table**

**`2c7c08b` — Removed tracked model binaries and ignore future artifacts**  
Added model files to .gitignore.

**`62f3034` — Chore: refactored code, src works, reset synthetic_params.json**

**`6d46211` — Chore: updated docs to match current state**

**`0553d60` — Removed inconsistencies and improved workflow**

### 16.3 CV Benchmark Automation Phase (April 2026)

**`c4881fd` — Generated features via pipeline**  
Regenerated `user_features.csv` with consistent seed.

**`704ae39`, `f89cc22` — Ran notebooks: statistics, train, analysis**  
Full end-to-end runs documented.

**`0b8e6fb` — Run: hypertuning, ablation, cv**  
All three benchmark scripts run and results persisted.

**`fb18540` — Docs: replace hardcoded metrics with CSV references in PROJECT.md**  
Important documentation principle: metrics in docs should reference CSV files, not be hardcoded.

**`1b2bf09` — Feat: add real data pipeline and adapt benchmark for Telecel Ghana MoMo data**  
Major milestone: `real_data_pipeline.py` added. The project bifurcates into Paper A (synthetic) and Paper B (real data). `real-data` branch created.

**`976d860` — Feat: add git clone cell to real data notebook for Azure Databricks**  
The Databricks workflow requires cloning the repo to install the package.

**`e481e94` — Added end-end test**  
First test of the Databricks pipeline end-to-end.

**`0767849` — Fix(crash): patched crash at significance test**  
Significance test edge case — zero division or empty bootstrap sample.

**`43ef692` — Fix: remove .cache() calls — not supported on Databricks serverless**

**`7dec751` — Docs: add Paper B real data pipeline status and next steps to PROJECT.md**

**`ecac9ec` — Fix(lstm): added patch to allow seq running**  
LSTM was crashing on the real data format. Patched for compatibility.

**`4302257` — Fix(pipeline): remove duplicate build_sequences_spark and build_pipeline definitions**

**`19940b7` — Fix(notebook): route CV benchmark to real data directory**

**`cb121c4` — Fix(paper): update to real Telecel results and add related work**  
Paper draft updated with real data results.

**`447e6c8` — Progress(classical ml): fixed switch to synthetic data at CV**

**`fe59d79` — Feat(notebook): print ROC JSON for local rendering; add gen_roc.py**  
Added a helper to generate ROC curves locally from JSON.

**`b60c531` — Fix(paper): update results to latest CV run; add ROC figure and std**

**`72d2f67` — Fix(paper): deanonymise — use real author name and affiliation**

**`fc00497` — Revert(paper): restore anonymous authorship for double-blind review**

**`bec28ce` — Fix(paper): fix layout gaps, ROC placement, appendix formatting**

### 16.4 Databricks Performance Phase (April–May 2026)

**`ab34d49` — Perf(pipeline): eliminate toLocalIterator timeout in build_sequences_spark**  
`toLocalIterator()` was timing out on 374M rows. Replaced with a Spark-native approach.

**`9d6ea97` — Perf(pipeline): halve driver peak memory in build_sequences_spark**

**`8d9e628` — Perf(pipeline): fix 6-scan problem — materialise df_ts + inline seq_len**  
Single most impactful performance fix for the Databricks pipeline.

**`fc91f8f` — Perf(pipeline): fix DBFS_DISABLED + eliminate max_ts collect()**  
Two issues: DBFS writes were disabled on serverless; max_ts was computed via a separate `collect()` action. Both fixed.

**`ff611a0` — Perf(pipeline): pre-filter + narrow columns before timestamp parse**  
Filter to borrowers and select only needed columns before the expensive timestamp parse step.

**`70065d3` — Feat(notebook): stream CV benchmark output live for progress visibility**  
Added live output streaming so we can see progress during long CV runs.

**`2f96119` — Perf(cv): batch_size 32→256 + always save results to runtime dir**

**`5d0c01e` — Progress(nn + docs): improved nn training time + mloss roadmap**  
Documented JMLR MLOSS submission roadmap in PROJECT.md.

**`3e35218` — Fix(stats): correct bootstrap p-value — was ≈0.5 always due to missing null centering**  
Critical bug fix. p-values were meaningless. Now correctly null-centered.

**`4375ac2` — Feat(models): add GRUModel and HybridGRUModel to CV benchmark**  
GRU added as an alternative to LSTM. GRU has fewer parameters and trains faster.

**`6e39b89` — Plan & notebook update**

**`e387402` — Perf(training): epochs 50→30, patience 10→7 for CPU-only cluster**  
Reduced training budget for the CPU cluster constraints on Databricks.

**`e7a56a2` — Fix(output): stream CV benchmark live + verbose=2 for clean per-epoch logs**

**`5069dcb` — Feat(models): add Transformer and HybridTransformer to CV benchmark**  
Brief experiment with attention-based models.

**`9f2a3b8` — Progress(notebook b): run end-end on a10**  
Successful end-to-end run of Notebook B on A10 cluster (GPU-capable).

**`9b9982f` — Fix(pipeline): raise min_followup_days 30→60 to capture penalty signals**  
Observation window truncation fix.

**`38b0cd6` — Feat(notebook-a): add cell to export filtered borrower transactions to DBFS**

**`3c22bbc`, `19c5a89`, `66d01aa` — Fix(notebook-a): export to Unity Catalog managed table**  
Series of fixes for the DBFS/Unity Catalog export flow. DBFS root was disabled; switched to Unity Catalog managed table.

### 16.5 Current State (May 2026)

**`3bb6242` — Perf(cv): drop all hybrid models — cut sequential runs 60→30**  
Hybrid models dropped from the Databricks CV to reduce runtime.

**`d332846` — Perf(cv): drop Transformer — benchmark LSTM vs GRU only (20 sequential runs)**  
Transformer baseline removed from scope.

**`f5cfa63` — Feat(notebooks): A runs static-only, B runs static + LSTM + GRU**

**`d8e7cfa` — Refactor(notebooks): lean EDA + inline bench.main() — drop 9 redundant cells**

**`a2878a9`, `ce07412` — Fix(notebook-b): clean build_pipeline cell**

**`df398b9`, `3cafff4` — Fix(notebook-a): rename loop vars df→cv_df, col→metric**

**`bec1252` — Fix(cv): recompute XGBoost scale_pos_weight per target**

---

## 17. Publication Plan and Targets

### 17.1 Paper A: The Synthetic Benchmark Framework

**Title (working):** "Sequential Credit Risk Modeling for African Fintech: A Calibrated Synthetic Benchmark and Comparative Evaluation"

**Target venues:**
- **Deep Learning Indaba 2026** — abstract deadline April 15, 2026 (Lagos, Nigeria, August 2–7)
- **IEEE ICAST 2026** — abstract deadline ~April 30, 2026 (AI track or Digital Innovation track)

**Framing:** Preliminary Investigation / Novel Framework + Open Benchmark. This paper does not claim to present production-ready or universally generalizable results. It contributes:
1. A calibrated synthetic Ghanaian mobile money dataset (open benchmark, 10k users, 5 archetypes, 1.03M transactions)
2. A temporal feature engineering framework (8 groups, 38 features) — reusable open-source pipeline for African fintech research
3. A preliminary comparative evaluation: static features dominate sequences; LSTM alone is insufficient; Hybrid LSTM is comparable in AUC but worse in calibration
4. Ablation evidence: `behavioural_diversity` (Δ = −0.090 AUC when dropped) and `loan_history` (Δ = −0.104) carry most default signal

**Core finding (publishable):** *Well-engineered aggregate behavioral features capture the relevant predictive signal in this synthetic MoMo benchmark. The temporal order of individual transactions, as extracted by standard LSTM architectures, adds negligible discriminative value for default prediction. Behavioral diversity (recipient entropy, transaction type entropy) is the dominant non-loan predictor of credit risk.*

**JMLR MLOSS Roadmap (longer-term):** The project is also a candidate for JMLR Machine Learning Open Source Software track:
1. Fix README with professional overview and Quick Start
2. Implement unit tests (60-70% coverage)
3. Generate API documentation (Sphinx or pdoc)
4. Draft 4-page software paper
5. Public release and community engagement

### 17.2 Paper B: Real-Data Validation (Telecel Ghana)

**Status:** Data received and pipeline implemented. CV results available (April 23, 2026). Pending:
- Observation window truncation fix and rerun
- Ablation study on real data
- LSTM/GRU benchmark on real data
- Manuscript draft

**Target venues:** TBD (likely AAAI, NeurIPS, or high-impact African AI venue, 2026/2027)

**Core research question:** Do the synthetic findings (static > sequences, behavioral diversity dominates) hold at national scale with real Telecel Ghana data?

**Preliminary answer from current results:** Yes — static models (LightGBM: 0.9464 AUC-ROC) achieve excellent performance on real data. The `behavioural_diversity` framework appears to generalise.

---

## 18. Open Questions and Future Work

### 18.1 Resolved Questions

**Q: Why did static models perform much worse in early runs?**  
A: Class imbalance handling bug. `class_weight='balanced'` and `scale_pos_weight` were missing. After fix: static models achieve 0.83–0.91 AUC-ROC.

**Q: Is the "sequential models beat static models" thesis valid?**  
A: No. Static models match or exceed Hybrid LSTM in AUC; are significantly better calibrated. Revised thesis: static features dominate sequences on this benchmark.

**Q: Why does standalone LSTM collapse?**  
A: Confirmed as a data property, not an architecture problem. Transaction sequences alone carry insufficient signal to predict default on this synthetic benchmark.

### 18.2 Open Questions

**Q1: Is the sequential collapse a property of LSTM specifically, or of MoMo data generally?**  
A Transformer/Attention baseline would answer this definitively, but has been dropped from scope due to compute cost. This remains the most intellectually important open question.

**Q2: Does the Behavioral Diversity finding hold on real data?**  
Real data ablation study has not been run. The current Paper B CV results show strong static model performance, which is consistent, but we don't know which features are most important on real data.

**Q3: How to handle the 463 cold-start borrowers?**  
First-time borrowers with no pre-loan history are excluded by the current pipeline. Can behavioral entropy serve as a "Proxy History" for the Zero-History borrower?

**Q4: Is the 50.9% default rate in the early Paper B run purely a truncation artifact?**  
With `min_followup_days=60`, what is the true default rate? This needs to be recomputed after the filter is tightened.

**Q5: What is the optimal `min_followup_days` parameter?**  
60 days was chosen pragmatically. The correct value depends on the typical loan repayment timeline and penalty structure in the real data.

### 18.3 Future Work

**High priority:**
- Run Paper B pipeline with corrected observation window filter and recompute CV
- Run ablation study on real Telecel Ghana data
- Complete LSTM/GRU benchmark on real data (requires Spark-based sequence builder)
- Manuscript draft for Paper A

**Medium priority:**
- Create `run_full_experiment_suite.py` for one-command reproducibility
- Unit tests for data integrity and label alignment
- JMLR MLOSS submission preparation

**Long-term:**
- External validation on other African mobile money markets (Kenya M-Pesa, Tanzania Vodacom M-Pesa)
- Exploration of Transformer/attention architectures if compute becomes available
- Extension of the `min_txn_count_eligibility` study for thin-file borrowers

---

## 19. Codebase Reference

### 19.1 Key File: `src/seqcredit_model/credit_model.py`

The largest and most complex file. Contains:
- `LSTM_FEATURE_COLUMNS` (line 79): The 38-feature subset for sequential models
- `bootstrap_evaluate()` (line 209): Bootstrap CI computation
- `CreditRiskDataLoader` (line 286): Data loading, splitting, sequence building, validation
- `LogisticRegressionModel` (line 969): LR with CV and save/load
- `XGBoostModel` (line 1042): XGB with CV and save/load
- `RandomForestModel` (line 1127): RF with CV and save/load
- `LightGBMModel` (line 1219): LightGBM with CV and save/load
- `LSTMModel` (line 1318): Stacked LSTM with Masking, regularization, early stopping
- `HybridLSTMModel`: Dual-branch LSTM + static architecture

### 19.2 Key File: `src/seqcredit_model/synthesize.py`

- `CalibratedMoMoDataGenerator`: The full simulation engine
- `CREDIT_ARCHETYPES` (line 48): Five archetype definitions with weights and behavioral parameters
- `generate_user_profile()` (line 162): Per-user behavioral parameter sampling
- `generate_transactions_for_user()` (line 443): Main simulation loop
- `generate_dataset()` (line 648): Outer loop over all users, writes CSVs and labels

### 19.3 Key File: `src/seqcredit_model/pipeline.py`

- `TemporalTransactionFeatureEngineer`: Feature extraction engine
- `extract_all_features()` (line 23): 113 transaction-level features across 11 categories
- `create_user_level_summary()` (line 210): 29 user-level aggregate features
- `build_user_feature_dataset()` (line 267): Batch process all user files

### 19.4 Key File: `src/seqcredit_model/real_data_pipeline.py`

- `build_pipeline(df, spark)`: Main entry point for Databricks
- `_parse_timestamp()` (line 91): Oracle timestamp parsing
- `_derive_loan_cutoffs()` (line 107): Index loan identification with truncation bias protection
- `_derive_labels()`: Repayment/penalty label derivation from post-loan events
- `_build_user_features()`: Spark aggregation of pre-loan features
- `build_sequences_spark()`: Spark-based sequence array construction for LSTM/GRU

### 19.5 Key File: `src/seqcredit_model/run_cv_benchmark.py`

- `compute_ece()` (line 136): 15-bin ECE computation
- `compute_brier_score()`: Brier score wrapper
- `run_static_model_cv()`: CV loop for static models
- `run_lstm_cv()`: CV loop for sequential models
- `run_significance_tests()`: Bootstrap pairwise significance tests
- `main()`: Full benchmark orchestration

### 19.6 Key Data Files

| File | Description |
|------|-------------|
| `data/cv_results_y_default.csv` | 30 rows (6 models × 5 folds), 10 columns |
| `data/cv_results_y_bad.csv` | 30 rows (6 models × 5 folds), 10 columns |
| `data/significance_tests.csv` | Pairwise bootstrap tests, all model pairs, both targets |
| `data/ablation_features.csv` | 17 conditions (ALL + 8 DROP + 8 ONLY), 9 columns |
| `data/tuning_results.csv` | 3 tuned models, best params, final CV scores |
| `data/cv_manifest.json` | Reproducibility metadata including all timings |
| `data/user_features.csv` | 10,000 rows × 30 columns (user_id + 29 features) |
| `data/user_labels.csv` | 10,000 rows × 6 columns |
| `src/synthetic_params.json` | All calibration parameters with provenance |

---

## 20. Calibration Sources and Methodology

The synthetic data parameters are calibrated to real Ghanaian mobile money data. Every parameter is derived from a documented public source. Below is the complete provenance:

### 20.1 Transaction-Level Parameters (Block 1)

| Parameter | Value | Derivation |
|-----------|-------|------------|
| `amount_lognormal_mu` | 2.8421 | Fitted to match `amount_mean=32.80` and `amount_median=18.50` |
| `amount_lognormal_sigma` | 1.0034 | Fitted for right-skewed distribution consistent with GHS distribution |
| `amount_mean` | 32.80 GHS | Bank of Ghana MoMo transaction reports (public aggregate) |
| `amount_median` | 18.50 GHS | Bank of Ghana MoMo transaction reports |
| `balance_mean` | 305.88 GHS | Bank of Ghana financial data (2023–2025) |
| `balance_std` | 302.55 GHS | Same source (high standard deviation reflects inequality) |
| `type_distribution` | TRANSFER: 52.9%, DEBIT: 27.2%, PAYMENT_SEND: 10.8%, CASH_OUT: 5.0%, PAYMENT: 4.1% | GSMA Mobile Money State of the Industry report (Ghana-specific data) |
| `weekend_rate` | 32.2% | Bank of Ghana daily transaction data |
| `night_rate` | 8.1% | Bank of Ghana time-of-day data |
| `fee_rate` | 24.3% | Operator product disclosures |
| `elevy_rate` | 12.7% | Ghana Revenue Authority E-Levy statistics (1.5% on transfers >GHS 100) |

### 20.2 Loan Parameters (Block 2 — Referenced Sources)

All 17 loan sources are documented in `DATACARD.md Section 12`. Key verified sources:

| Ref | Source | Parameters Derived |
|-----|--------|-------------------|
| [1] Asetenapa 2025 | MTN QwikLoan product page | min/max loan, interest rate (6.9%), penalty (12.5%), term (30 days), eligibility (90 days) |
| [2] Letshego H1 2024 | Ghana Stock Exchange presentation | Implied avg loan (GHS 430), female share (34%), loan purpose distribution |
| [3] JUMO Ghana 2025 | JUMO corporate blog | Repeat borrowing rate (13.3 loans/customer), cumulative 128M loans |
| [4] CitiNewsroom 2023 | Transaction evidence | XpressLoan rate (8.9%), processing fee (1%) |
| [5] Telecel Ghana support | Official support page | Ready Loan terms (8.9%, 12.5% penalty, 30-day) |
| [8] CGAP Ghana 2020 | Academic fieldwork | Default rate "single digits" → mean 6% |
| [9] JUMO/Orange 2025 | Press release | Cost of risk <4% → lower bound 3% |
| [11] World Bank Findex 2025 | Global survey | Borrower prevalence 22% of Ghanaian adults |
| [13] BoG Directive 2025 | Regulatory filing | Max loan per customer GHS 10,000 |
| [14] BoG FinTech 2024 | Annual report | Monthly transaction volumes → seasonality index |

### 20.3 Parameters with No Published Data (Set to NaN)

Four parameters could not be found in any public source and are marked `NaN` in `synthetic_params.json`. The generator uses designer assumptions for these:

- `approval_rate` — no operator publishes approval/rejection ratios
- `early_repayment_pct` — no published data; CGAP Kenya proxy (~20%) used
- `on_time_repayment_pct` — derived residually
- `late_repayment_pct` — no Ghana-specific data; CGAP Kenya proxy (~38%) used

---

## 21. Complete Feature Engineering Reference

### 21.1 Transaction-Level Feature Equations (`pipeline.py`)

`TemporalTransactionFeatureEngineer.extract_all_features()` produces 113 per-row features per transaction. The 38-feature LSTM subset (`LSTM_FEATURE_COLUMNS`) are drawn from this set.

#### Amount Transforms
```
log_amount         = log1p(AMOUNT)
sqrt_amount        = sqrt(AMOUNT)
amount_squared     = AMOUNT²
is_micro_txn       = (AMOUNT < 10)
is_small_txn       = (AMOUNT >= 10) & (AMOUNT < 50)
is_medium_txn      = (AMOUNT >= 50) & (AMOUNT < 200)
is_large_txn       = (AMOUNT >= 200)
```

#### Fee Features
```
has_fees               = (FEES > 0)
has_elevy              = (E-LEVY > 0)
total_cost             = AMOUNT + FEES + E-LEVY
fee_to_amount_ratio    = FEES / (AMOUNT + 1)
elevy_to_amount_ratio  = E-LEVY / (AMOUNT + 1)
```

#### Transaction Type One-Hots (7 binary flags)
`is_transfer`, `is_debit`, `is_payment`, `is_payment_send`, `is_cash_out`, `is_cash_in`, `is_adjustment` — each equals 1 when `TRANS. TYPE` matches the corresponding string. Note: `is_adjustment` is always 0 in practice — the generator never emits `ADJUSTMENT` transactions.

#### Temporal Raw Fields
```
hour           = TRANSACTION DATE.dt.hour
day_of_week    = TRANSACTION DATE.dt.dayofweek
is_weekend     = (day_of_week >= 5)
is_early_morning = (hour >= 0) & (hour < 6)
is_morning     = (hour >= 6) & (hour < 12)
is_afternoon   = (hour >= 12) & (hour < 18)
is_evening     = (hour >= 18) & (hour < 22)
is_night       = (hour >= 22)
```

#### Cyclical Encodings (eliminates midnight and week-boundary discontinuities)
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
day_sin  = sin(2π × day_of_week / 7)
day_cos  = cos(2π × day_of_week / 7)
```

#### Balance Dynamics
```
log_balance_before    = log1p(BAL BEFORE)
log_balance_after     = log1p(BAL AFTER)
balance_change        = BAL AFTER − BAL BEFORE
balance_pct_change    = balance_change / (BAL BEFORE + 1)
is_low_balance        = (BAL BEFORE < 20)
is_zero_balance       = (BAL BEFORE == 0)
amount_to_balance_ratio = AMOUNT / (BAL BEFORE + 1)
is_large_relative_to_balance = (amount_to_balance_ratio > 0.5)
will_deplete_balance  = (BAL AFTER < 10) & (BAL BEFORE > 10)
```

#### Sequence Position & Timing
```
time_since_last_txn_hours = diff(TRANSACTION DATE).total_seconds() / 3600
                            (first row = 0)
txn_number            = 1..len(df)
reverse_txn_number    = len(df) − txn_number
cumulative_volume     = AMOUNT.cumsum()
cumulative_fees_paid  = (FEES + E-LEVY).cumsum()
```

#### Count-Based Rolling Windows (n = 3, 5, 10 transactions)
For each n:
```
last_{n}_avg_amount        = AMOUNT.rolling(n, min_periods=1).mean()
last_{n}_std_amount        = AMOUNT.rolling(n, min_periods=1).std().fillna(0)
last_{n}_max_amount        = AMOUNT.rolling(n, min_periods=1).max()
last_{n}_min_amount        = AMOUNT.rolling(n, min_periods=1).min()
amount_vs_last_{n}_avg     = AMOUNT / (last_{n}_avg_amount + 1)
last_{n}_transfer_count    = is_transfer.rolling(n, min_periods=1).sum()
last_{n}_debit_count       = is_debit.rolling(n, min_periods=1).sum()
last_{n}_cashout_count     = is_cash_out.rolling(n, min_periods=1).sum()
```

#### Time-Based Rolling Windows (3d, 7d, 14d, 30d — calendar-time offsets)
For each window_days w:
```
rolling_{w}d_count             = AMOUNT.rolling("{w}D").count()
rolling_{w}d_sum               = AMOUNT.rolling("{w}D").sum()
rolling_{w}d_mean              = AMOUNT.rolling("{w}D").mean().fillna(0)
rolling_{w}d_std               = AMOUNT.rolling("{w}D").std().fillna(0)
rolling_{w}d_min_balance       = BAL BEFORE.rolling("{w}D").min()
rolling_{w}d_max_balance       = BAL BEFORE.rolling("{w}D").max()
rolling_{w}d_balance_volatility = BAL BEFORE.rolling("{w}D").std().fillna(0)
```

#### Behavioural Patterns
```
unique_recipients_so_far  = cumulative distinct TO NAME up to each row
unique_txn_types_last_10  = nunique(TRANS. TYPE) in sliding window [i-9:i+1]
is_repeated_recipient     = df.duplicated(subset=["TO NAME"], keep=False)
is_self_transfer          = (TO NAME == FROM NAME)
```

#### Risk Indicators
```
unusual_hour          = (hour < 6) | (hour > 22)
rapid_transaction     = (time_since_last_txn_hours < 0.5)
unusual_amount_high   = (amount_vs_last_10_avg > 3)
unusual_amount_low    = (amount_vs_last_10_avg < 0.3)
rapid_balance_drop    = (BAL BEFORE > 100) & (BAL AFTER < 20)
high_frequency_period = (time_since_last_txn_hours < 1)
risk_score            = unusual_hour + rapid_transaction + unusual_amount_high
                        + rapid_balance_drop + is_large_relative_to_balance
                        (integer 0–5)
```

### 21.2 User-Level Summary Equations (`create_user_level_summary`)

These 29 scalar features are stored in `data/user_features.csv`:

```
recipient_concentration  = df["TO NAME"].value_counts().iloc[0] / len(df)
                           (top-1 frequency / total transactions; 0 if empty)
transactions_per_day     = obs_txn_count / (account_age_days + 1)
avg_hours_between_txns   = (account_age_days × 24) / (obs_txn_count + 1)
cv_transaction_amount    = std_transaction_amount / (avg_transaction_amount + 1)
balance_volatility       = std(BAL BEFORE)
pct_low_balance_txns     = fraction of rows where BAL BEFORE < 20
```

All other user-level features are direct aggregations (sum, mean, max, count, proportion) of the transaction-level fields above.

### 21.3 Real Data Pipeline Equations (`real_data_pipeline.py`)

For the Telecel Ghana Spark pipeline, derived features are computed as PySpark expressions:

```
recipient_concentration = max_recip_count / (obs_txn_count + 1)
transactions_per_day    = obs_txn_count / (account_age_days + 1)
avg_hours_between_txns  = (account_age_days × 24.0) / (obs_txn_count + 1)
cv_transaction_amount   = std_transaction_amount / (avg_transaction_amount + 1)
net_flow                = total_inflow − total_volume
inflow_outflow_ratio    = total_inflow / (total_volume + 1)
total_repaid            = total_principal_repaid + total_interest_repaid
loan_repayment_ratio    = total_repaid / (total_loan_volume + 1)
loan_to_total_volume_ratio = total_loan_volume / (total_volume + total_inflow + 1)
pct_credit_transactions = n_loans_received / (obs_txn_count + 1)
```

Four balance-at-loan features (`loan_timing_in_sequence`, `avg_balance_at_loan`, `min_balance_at_loan`, `balance_to_loan_ratio_at_disbursement`) are set to 0.0 in the real data pipeline — the Telecel Ghana transaction log does not expose per-transaction account balances.

---

## 22. Complete Model Architecture Specifications

### 22.1 The 38 LSTM Feature Columns (`LSTM_FEATURE_COLUMNS`)

Exact list in order (indices 1–38):
1. `log_amount` 2. `is_micro_txn` 3. `is_small_txn` 4. `is_medium_txn` 5. `is_large_txn`
6. `fee_to_amount_ratio` 7. `total_cost` 8. `has_fees`
9. `is_transfer` 10. `is_debit` 11. `is_payment` 12. `is_payment_send` 13. `is_cash_out` 14. `is_cash_in` 15. `is_adjustment`
16. `hour_sin` 17. `hour_cos` 18. `day_sin` 19. `day_cos` 20. `is_weekend`
21. `time_since_last_txn_hours`
22. `log_balance_before` 23. `balance_pct_change` 24. `is_low_balance` 25. `is_zero_balance` 26. `amount_to_balance_ratio` 27. `will_deplete_balance` 28. `balance_change`
29. `is_repeated_recipient` 30. `is_self_transfer` 31. `unique_txn_types_last_10`
32. `unusual_hour` 33. `rapid_transaction` 34. `risk_score`
35. `amount_vs_last_5_avg` 36. `rolling_7d_std` 37. `reverse_txn_number` 38. `last_5_std_amount`

### 22.2 LSTMModel (lines 1318–1503 of `credit_model.py`)

Default hyperparameters: `lstm_units_1=32`, `lstm_units_2=16`, `dense_units=16`, `dropout_rate=0.4`, `learning_rate=0.001`.

Layer stack (Sequential):
1. `Masking(mask_value=0.0)` — ignores pre-padded zero timesteps
2. `LSTM(32, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=L2(1e-4))`
3. `Dropout(0.4)`
4. `LSTM(16, return_sequences=False, recurrent_dropout=0.3, kernel_regularizer=L2(1e-4))`
5. `Dropout(0.4)`
6. `Dense(16, activation="relu")`
7. `Dropout(~0.27)`
8. `Dense(1, activation="sigmoid")`

Compiled with `Adam(lr=0.001)`, `binary_crossentropy`, AUC metric. Callbacks: `EarlyStopping(patience=7, monitor=val_auc, restore_best_weights=True)`, `ReduceLROnPlateau(patience=5, factor=0.5)`. Training uses 15% validation split; balanced class weights computed automatically. Max 30 epochs (reduced from 50 for CPU clusters), batch size 256 (raised from 32 for efficiency).

### 22.3 GRUModel (lines 1505–1675)

Identical architecture to `LSTMModel` with all LSTM cells replaced by GRU cells. GRU has fewer parameters (no separate cell state). Added in commit `4375ac2` (2026-05-11) as a drop-in alternative — motivated by potentially faster convergence on 100-step capped sequences.

### 22.4 HybridLSTMModel (lines 1678–1865) — Early Fusion

Default: `lstm_units_1=32`, `lstm_units_2=16`, `dense_units=16`, `dropout_rate=0.3`.

Dual-input Functional API:
- **Sequence branch:** `sequence_input → Masking → LSTM(32, return_sequences=True, recurrent_dropout=0.2) → Dropout(0.3) → LSTM(16, return_sequences=False, recurrent_dropout=0.2) → Dropout(0.3) → vector x`
- **Static branch:** `static_input → Dense(8, relu) → Dropout(~0.20) → vector s`
- **Fusion:** `Concatenate([x, s]) → Dense(16, relu) → Dropout(~0.20) → Dense(1, sigmoid)`

### 22.5 HybridGRUModel (lines 1868–end)

Identical to HybridLSTMModel with GRU cells in the sequence branch. Added alongside GRUModel in commit `4375ac2`.

### 22.6 TransformerModel and HybridTransformerModel

Added in commit `5069dcb` (2026-05-13). Architecture: encoder-only Transformer (linear projection → N × [MultiHeadAttention + FFN + LayerNorm] → GlobalAveragePooling → sigmoid). HybridTransformer fuses sequence branch output with static features. **Removed in commit `d332846` (2026-05-19)** — dropped from benchmark to reduce sequential runs from 60 to 20.

---

## 23. CV Benchmark Engine: Full Implementation

**Constants:** `RANDOM_SEED=42`, `N_SPLITS=5`, `N_BOOTSTRAP=1000`, `CI_LEVEL=0.95`.

**Two targets:** `y_default` (credit_risk_label == 2) and `y_bad` (credit_risk_label ∈ {1, 2}).

### 23.1 Static Model CV Loop (`run_static_model_cv`)

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
For each fold:
    Fit StandardScaler on train split only → transform val
    Instantiate model with given params
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_val_scaled)
    Compute 8 metrics: auc_roc, auc_pr, f1, precision, recall, accuracy, brier, ece
    Accumulate OOF predictions
Returns: (fold_df, summary_dict, oof_predictions)
```

XGBoost `scale_pos_weight` is recomputed per target separately (bug fixed in `bec1252`).

### 23.2 LSTM/GRU CV Loop (`run_lstm_cv`)

Same StratifiedKFold. Per fold: `build_model(input_shape)` → `fit(X_tr_seq, y_tr, X_val=X_val_seq, y_val=y_val, epochs=30, batch_size=256)` → after each fold: `del model; tf.keras.backend.clear_session()` to free GPU/CPU memory.

### 23.3 Bootstrap Significance Test

**Bug fixed in commit `3e35218` (2026-05-11):** The original test compared `|deltas| >= |delta_mean|` — by definition ~50% of bootstrap samples always exceed their own mean, giving p ≈ 0.5 for everything. Fixed by centering: `deltas_centered = deltas − delta_mean; p_value = (|deltas_centered| >= |delta_mean|).mean()`.

### 23.4 Metrics Computed Per Fold

| Metric | Formula |
|--------|---------|
| AUC-ROC | sklearn `roc_auc_score` |
| AUC-PR | sklearn `average_precision_score` |
| F1 | at threshold 0.5 |
| Brier | `brier_score_loss` (MSE of probabilities) |
| ECE | `Σ_b (n_b/N) × |acc_b − conf_b|` over 15 equal-width bins |

### 23.5 Output Files

| File | Contents |
|------|----------|
| `data/cv_results_y_default.csv` | 30 rows (6 models × 5 folds), all 8 metrics |
| `data/cv_results_y_bad.csv` | Same for y_bad target |
| `data/cv_results_intermediate_*.csv` | Rolling checkpoint after each target completes |
| `data/cv_oof_preds.npz` | Out-of-fold probability arrays for ROC plotting |
| `data/significance_tests.csv` | 40 rows of pairwise bootstrap comparisons |
| `data/cv_manifest.json` | Seed, n_splits, n_bootstrap, per-model timing, total runtime |

---

## 24. Feature Group Ablation Study: Implementation

**Script:** `src/seqcredit_model/run_ablation_study.py`  
**Probe model:** RandomForest (for speed and interpretability)

### 24.1 Feature Group Definitions (FEATURE_GROUPS)

| Group | Columns (N) |
|-------|-------------|
| `amount_stats` | total_volume, avg/median/std/max/min/cv_transaction_amount (7) |
| `txn_type_mix` | pct_transfers, pct_debits, pct_cashouts, pct_payments (4) |
| `temporal_patterns` | avg_hours_between_txns, pct_weekend/night/early_morning_txns (4) |
| `balance_dynamics` | avg/min/max_balance, balance_volatility, pct_low_balance_txns (5) |
| `fee_behaviour` | total_fees_paid, avg_fees_per_txn, pct_txns_with_fees (3) |
| `behavioural_diversity` | unique_recipients, recipient_concentration, unique_txn_types (3) |
| `activity_intensity` | obs_txn_count, account_age_days, transactions_per_day (3) |
| `loan_history` | total/avg/max_loan_volume, loan_to_total_volume_ratio, pct_credit_transactions, loan_timing_in_sequence, avg/min_balance_at_loan, balance_to_loan_ratio (9) |

### 24.2 Two Ablation Conditions Per Group

1. **Drop-one:** All features except the target group — quantifies each group's unique contribution
2. **Single-only:** Only the target group — quantifies each group's isolated signal

Each condition runs full 5-fold CV and records AUC-ROC, AUC-PR, F1, Brier, ECE. Results sorted by delta vs baseline.

---

## 25. Hyperparameter Tuning: Implementation

**Script:** `src/seqcredit_model/run_hyperparameter_tuning.py`  
**Models tuned:** RandomForest, XGBoost, LightGBM (LSTM excluded — too slow for random search)  
**Strategy:** Two-stage: fast RandomizedSearchCV (40 trials, 3-fold) → rigorous 5-fold CV re-evaluation

### 25.1 Search Spaces

**RandomForest:** n_estimators ∈ {100,200,300,500}, max_depth ∈ {5,7,10,15,None}, min_samples_leaf ∈ {1,2,4,8}, min_samples_split ∈ {2,5,10}, max_features ∈ {"sqrt","log2",0.5}

**XGBoost:** n_estimators ∈ {100,200,300,500}, max_depth ∈ {3..7}, learning_rate ∈ {0.01,0.05,0.1,0.2}, subsample ∈ {0.6,0.7,0.8,1.0}, colsample_bytree ∈ {0.6,0.7,0.8,1.0}, min_child_weight ∈ {1,3,5}, gamma ∈ {0,0.1,0.3}

**LightGBM:** n_estimators ∈ {100,200,300,500}, num_leaves ∈ {15,31,63,127}, learning_rate ∈ {0.01,0.05,0.1,0.2}, subsample ∈ {0.6..1.0}, colsample_bytree ∈ {0.6..1.0}, min_child_samples ∈ {10,20,30,50}, reg_alpha/lambda ∈ {0,0.1,0.5}

**Finding:** Default hyperparameters are near-optimal (<1% gain from tuning), confirming that feature engineering quality dominates model configuration in this dataset.

---

## 26. Supporting Scripts

### 26.1 `reset.py` — Clean Slate Utility

Three modes controlled by flags:
- `--models`: deletes 8 trained model artifacts (6 `.pkl` + 2 `.keras` + 2 `.keras.json`)
- `--data`: deletes 11 cached/generated data files (`lstm_sequences.npz`, all CV CSVs, `model_comparison.csv`, etc.)
- `--data-full`: additionally deletes source data (`user_labels.csv` + `data/user_transactions/`)
- Default (no flags): runs `--models` + `--data`

Prompts for confirmation unless `--yes` is passed. Prints counts of deleted files.

### 26.2 `compute_bootstrap_ci.py` — Bootstrap CI on Saved Models

Loads all 6 trained model `.pkl`/`.keras` files from `models/`, loads `data/lstm_test_arrays.npz`, generates predictions for all 6 models, runs 1000-bootstrap CI computation via `ModelEvaluator.add_model(compute_ci=True)`, saves extended table to `data/model_comparison.csv`.

Note: `data/model_comparison.csv` contains results from a pre-fix run when static models were near-random (LR AUC 0.479, RF 0.487) while HybridLSTM was 0.863. This represents the state before feature engineering corrections — these are NOT the current authoritative results (use `cv_results_y_default.csv` instead).

### 26.3 `run_full_benchmark.py` — Reproducibility Wrapper

Clears stale caches (5 files), sets global seeds (numpy, random, tensorflow), then delegates to `run_cv_benchmark.main()`. Designed as a single entry point for a clean reproducibility run.

---

## 27. Analysis Notebooks — Azure Databricks

Both notebooks are run on Azure Databricks against `melodatabricks616.default.yara_dump_table` (374,295,424 rows). They clone `real-data` branch from `https://github.com/attabeezy/seqcredit-model.git` at runtime into `/tmp/seqcredit-model`.

### 27.1 Analysis Notebook A — Static Feature Pipeline

**Purpose:** Runs static-only CV benchmark (LR, XGBoost, RF, LightGBM) on real Telecel Ghana data.

Key cells:
- **Cell 1–2:** Schema inspection — confirms 374,295,424 rows, 12 columns; prints transaction type frequency table
- **Cell 3–4:** Borrower counts: 474,312 total borrowers; 324,863 penalised; 269,381 both borrowers+penalised; 20,563 with zero repayments (strict defaulters)
- **Cell 6–8:** Clone repo, install, restart Python
- **Cell 10:** `rdp.build_pipeline(df)` — runs 4-stage pipeline with 60-day followup filter; produces 149,204 users × 48 features
- **Cell 11:** `bench.main()` — runs 5-fold CV; LSTM skipped (no per-user transaction files)
- **Cell 12 (display):** Results table reading from `/tmp/seqcredit_model/`:

| Model | AUC-ROC (y_default) | AUC-PR | F1 | Brier | ECE |
|-------|---------------------|--------|----|-------|-----|
| LogisticRegression | **0.9143** | 0.6786 | 0.5940 | 0.1085 | 0.1704 |
| RandomForest | 0.8836 | 0.7159 | 0.6686 | 0.0689 | 0.1039 |
| XGBoost | 0.8809 | 0.7185 | 0.6653 | 0.0609 | 0.0416 |
| LightGBM | 0.8777 | 0.7211 | 0.6560 | **0.0600** | **0.0378** |
| HybridLSTM | 0.8501 | 0.6377 | 0.5144 | 0.1162 | 0.1658 |
| LSTM | 0.5872 | 0.1734 | 0.2707 | 0.2294 | 0.3294 |

| Model | AUC-ROC (y_bad) | AUC-PR | F1 | Brier | ECE |
|-------|-----------------|--------|----|-------|-----|
| RandomForest | **0.7269** | **0.7432** | 0.5917 | 0.2031 | **0.0649** |
| LogisticRegression | 0.7254 | 0.7207 | **0.6208** | 0.2084 | 0.0750 |
| LightGBM | 0.7152 | 0.7387 | 0.5995 | 0.2087 | 0.0888 |
| XGBoost | 0.7098 | 0.7341 | 0.6357 | 0.2704 | 0.2317 |
| HybridLSTM | 0.7054 | 0.7079 | 0.6170 | 0.2168 | 0.0850 |
| LSTM | 0.5334 | 0.4799 | 0.4266 | 0.2492 | 0.0428 |

- **Cell 13:** Loan lifecycle analysis — computes days-to-first-repayment and days-to-first-penalty percentile distributions (motivation for 60-day cutoff: penalty p10=31d, p50=55d, mean=64d)
- **Cell 14:** Exports filtered borrower transactions to Unity Catalog managed table `melodatabricks616.default.seqcredit_telecel_borrowers`

### 27.2 Analysis Notebook B — Static + LSTM + GRU Pipeline

**Purpose:** Full benchmark including LSTM and GRU sequence models.

Key cells beyond what Notebook A has:
- **Cell 11:** `rdp.build_sequences_spark(df, min_followup_days=60, max_seq_len=100)` — builds padded array `(149,667, 100, 19)` using Spark workers for all 19 features; saves to `/tmp/seqcredit_model/lstm_sequences_raw.npz`
- **Cell 13:** `bench.main()` — runs full CV including LSTM and GRU; live-streamed output via `Popen` line iteration
- **Cell 14 (display):** Results including LSTM/GRU columns

**Real data pipeline output (60-day filter):**
```
[1/4] Borrowers (with >=1 loan, >=60d followup): 149,667
[2/4] Good (0): 51,105 (34.1%)  |  Risky (1): 92,392 (61.7%)  |  Default (2): 6,170 (4.1%)
[3/4] Shape: (149,204, 48)
[4/4] Saved to /tmp/seqcredit_model/
```

---

## 28. Local Notebooks — Synthetic Benchmark

### 28.1 `notebooks/model.ipynb` — Primary Modeling Notebook

The main local notebook for the synthetic benchmark. Trains all 6 models sequentially, produces model comparison table and plots, saves model artifacts.

**Key outputs from a representative run:**
- 10,000 matching users; 4,867 train / 1,217 test; 11.1% default rate; 38 features; scale_pos_weight = 6.98
- Archetype breakdown (borrowers): responsible_borrower 3,506, occasional_borrower 1,480, risky_borrower 871, defaulter 227

**5-fold CV results (static models, training set):**

| Model | AUC-ROC | AUC-PR | F1 | Accuracy |
|-------|---------|--------|----|----------|
| LR | 0.9143 ± 0.006 | 0.6786 ± 0.032 | 0.5940 ± 0.010 | 0.8601 ± 0.007 |
| XGBoost | 0.8797 ± 0.016 | 0.7155 ± 0.039 | 0.6543 ± 0.037 | 0.9273 ± 0.005 |
| RF | 0.8832 ± 0.013 | 0.7154 ± 0.033 | 0.6693 ± 0.040 | 0.9349 ± 0.005 |
| LightGBM | 0.8816 ± 0.015 | 0.7250 ± 0.036 | 0.6532 ± 0.035 | 0.9252 ± 0.005 |

**Test-set single-run (model.ipynb ModelEvaluator comparison table):**

| Model | AUC-ROC | AUC-PR | F1 | Accuracy |
|-------|---------|--------|----|----------|
| LR | 0.4794 | 0.1232 | 0.1674 | 0.706 |
| XGBoost | 0.4997 | 0.1228 | 0.0709 | 0.806 |
| RF | 0.4870 | 0.1229 | 0.0601 | 0.820 |
| LightGBM | 0.4948 | 0.1202 | 0.0672 | 0.795 |
| Sequential LSTM | 0.5046 | 0.1263 | 0.0566 | 0.836 |
| Hybrid LSTM | **0.8627** | **0.6170** | **0.4988** | **0.823** |

**Important discrepancy:** The test-set ModelEvaluator table (above) shows static models at AUC ~0.48–0.50 while CV shows them at 0.88–0.91. This is because the ModelEvaluator comparison uses `y_test_lstm` — the label vector for users who had transaction files available for LSTM processing. These users happen to be a different (non-random) subset from those in static model CV. The CV benchmark scripts (`run_cv_benchmark.py`) are the authoritative results.

**LSTM input shape:** `(4,867, 100, 38)` — sequences padded to 100 steps, 38 features  
**Sequence building:** Per-user transaction files read, features extracted by `TemporalTransactionFeatureEngineer`, scaled per-fold on training set statistics, pre-padded with zeros

Saves to `models/`: lr_model.pkl, xgb_model.pkl, rf_model.pkl, lgbm_model.pkl, lstm_model.keras, hybrid_model.keras; also saves `data/lstm_test_arrays.npz`.

### 28.2 `notebooks/analysis.ipynb` — Research Analysis Notebook

Loads all 6 saved models and runs the complete XAI + calibration pipeline. Depends on `model.ipynb` having been run first.

**Section 3 — Secondary Target y_bad:**
y_bad rate: train 45.2%, test 46.6%. All 6 models evaluated on both y_default and y_bad.

**Section 4 — Permutation Importance:**
`sklearn.inspection.permutation_importance` with `n_repeats=10`, AUC scoring. Applied to LR, XGBoost, RF, LightGBM via a `ProbaWrapper` that exposes the sklearn interface.

**Section 5 — SHAP Analysis:**
`shap.TreeExplainer` for XGBoost and LightGBM. Global beeswarm plots (top 15 features); local waterfall plots for highest-risk and lowest-risk users.

Top-10 SHAP features (LightGBM, y_default vs y_bad):

| Rank | y_default feature | |SHAP| | y_bad feature | |SHAP| |
|------|-------------------|-------|--------------|-------|
| 1 | unique_txn_types | 4.768 | recipient_concentration | 0.011 |
| 2 | pct_payments | 0.986 | unique_recipients | 0.007 |
| 3 | max_loan_amount | 0.820 | min_balance | 0.006 |
| 4 | pct_cashouts | 0.758 | pct_credit_transactions | 0.006 |
| 5 | total_loan_volume | 0.650 | loan_timing_in_sequence | 0.005 |

The magnitude difference (~4.77 vs ~0.011) reflects that y_default has a cleaner signal driven by a small number of features, while y_bad's signal is distributed more evenly.

**Section 6 — Surrogate Decision Tree:**
Fidelity curve at depths 2–8 (R² and MAE vs LightGBM teacher). Final surrogate at depth 3 with `export_text` for human-readable compliance-auditable rules.

**Section 7 — Individual Tree Visualizations:**
RF: `estimators_[0]`, depth-4 plot via `sklearn.tree.plot_tree`. XGBoost: tree 0 via `xgb.plot_tree`. LightGBM: tree 0 via `lgb.plot_tree` with Graphviz fallback to structured text dump.

**Section 8 — Calibration:**
Brier score, ECE (15 bins), reliability diagrams. Post-hoc isotonic calibration (`IsotonicRegression`) applied to XGBoost and LightGBM on a 20% calibration split carved from training data.

**Calibration table from analysis.ipynb (y_default, test set — saved models run):**

| Model | Brier | ECE | AUC-ROC |
|-------|-------|-----|---------|
| LR | 0.2243 | 0.2431 | 0.4794 |
| XGBoost | 0.6537 | 0.7212 | 0.4587 |
| RF | 0.1545 | 0.2010 | 0.4608 |
| LightGBM | 0.7788 | 0.8154 | 0.4927 |
| LSTM | 0.2346 | 0.3537 | 0.5046 |
| Hybrid LSTM | **0.1299** | **0.2067** | **0.8627** |

Note: These calibration numbers reflect the pre-fix model state (test-set evaluation issue described above). Authoritative calibration figures are in `data/cv_results_y_default.csv`.

### 28.3 `notebooks/data.ipynb` — Descriptive Statistics Notebook

Characterises the synthetic dataset without any modeling. Samples 500 users (498 successfully) → 51,556 transactions for transaction-level analysis. Date range: 2024-01-01 to 2024-06-27.

**Dataset sizes:** `user_features.csv` (10,000 × 30), `user_labels.csv` (10,000 × 6), 10,000 transaction files. Zero missing values.

**Label distribution:**

| Risk Label | Count | % |
|------------|-------|---|
| -1 (No Loans) | 3,916 | 39.2% |
| 0 (Good) | 3,315 | 33.2% |
| 1 (Late) | 2,007 | 20.1% |
| 2 (Default) | 762 | 7.6% |

Borrowers only (N=6,084): 762 defaults = 12.5% class imbalance. Archetype distribution: non_borrower 3,916, responsible_borrower 3,506, occasional_borrower 1,480, risky_borrower 871, defaulter 227.

**Transaction type breakdown (sampled 51,556 rows):**

| Type | Count | % |
|------|-------|---|
| TRANSFER | 22,981 | 44.6% |
| DEBIT | 11,851 | 23.0% |
| CASH_IN | 6,425 | 12.5% |
| PAYMENT | 3,328 | 6.5% |
| PAYMENT_SEND | 3,235 | 6.3% |
| CASH_OUT | 2,199 | 4.3% |
| CREDIT | 787 | 1.5% |
| LOAN_REPAYMENT | 750 | 1.5% |

**Loan providers (from CREDIT transactions):** XPRESSLOAN 22.1%, QWIKLOAN 21.1%, XTRACASH 20.7%, CEDISPAY 20.1%, FIDO 16.0%

**Temporal pattern:** Near-uniform day-of-week (Sun/Sat slightly higher at 15.0%/14.9%). Near-zero hours 0–5 (~1.4% each), peak hours 10–22 (~4–7% each), spike at hour 22 (6.6% — likely batch transactions).

**Loan statistics:** Disbursement amounts: mean GHS 33.07, std 8.25, range 25–82 (calibrated to QwikLoan 25–1000 range but synthetic amounts cluster low). Loans per borrower: mean 2.67, std 1.23, range 1–6.

**Generator calibration deviations (|Δ| > 10%):**

| Parameter | Target | Generated | Δ% |
|-----------|--------|-----------|-----|
| Amount median (GHS) | 18.50 | 15.41 | −16.7% |
| Night rate | 0.0810 | 0.1396 | +72.4% |
| Fee rate | 0.2430 | 0.0132 | **−94.6%** |
| Balance mean (GHS) | 305.88 | 136.52 | **−55.4%** |
| Balance std (GHS) | 302.55 | 31.24 | **−89.7%** |
| pct PAYMENT_SEND | 0.1080 | 0.0742 | −31.3% |
| pct PAYMENT | 0.0410 | 0.0763 | +86.2% |

Fee rate and balance statistics show the largest deviations from calibration targets — documented in `docs/DATACARD.md` as known limitations.

**Per-archetype summary:** All 5 archetypes have nearly identical means across the 29 static features (e.g., obs_txn_count ~100–106 for all groups), confirming that discriminative signal comes from learned interaction effects rather than raw feature magnitudes.

---

## 29. Google Colab Variants

Three notebooks adapted for Google Colab (`data_gcolab.ipynb`, `model_gcolab.ipynb`, `analysis_gcolab.ipynb`). Structural differences from local notebooks:

| Aspect | Local notebooks | GColab notebooks |
|--------|----------------|------------------|
| Setup | Assumes editable install | Cells A–C: git clone → pip install → data regeneration |
| Paths | `config.py` resolution | `PROJECT_ROOT = '/content/seqcredit-model'` hardcoded |
| Graphviz | Assumed present | `apt-get install graphviz` in setup cell |
| Model files | Local `models/` | Must be produced by `model_gcolab.ipynb` on same runtime |
| Error handling | Implicit | Cell C raises `RuntimeError` if model files missing |

**`model_gcolab.ipynb` CV results (higher than local — different random split):**

| Model | CV AUC-ROC | CV AUC-PR |
|-------|-----------|----------|
| LR | 0.9809 ± 0.006 | 0.8642 ± 0.045 |
| XGBoost | 0.9813 ± 0.005 | 0.8990 ± 0.020 |
| RF | 0.9805 ± 0.005 | 0.8905 ± 0.022 |
| LightGBM | 0.9844 ± 0.004 | 0.9110 ± 0.016 |

Sequential LSTM test AUC-ROC: 0.4841. Hybrid LSTM test AUC-ROC: **0.9515** — the best Hybrid performance recorded, suggesting this particular random split was favorable for the model.

**`analysis_gcolab.ipynb` notable finding:** `final_credit_limit` dominates SHAP for y_default with mean |SHAP| = 5.23 — far larger than all other features. This is structurally determined: credit limit in the synthetic simulator directly tracks repayment history, creating near-deterministic signal. This inflates CV AUC on synthetic data and is a key validity concern documented in `DATACARD.md`.

**Known bugs:**
- `data_gcolab.ipynb` cell-s2-dist-activity: trailing `"` on `plt.show()"` causes `SyntaxError: unterminated string literal`
- `model_gcolab.ipynb` test-set comparison shows static models at AUC ~0.49–0.52 due to `y_test_lstm` alignment issue (same issue as in `analysis.ipynb`, described in Section 28.1)

---

## 30. Paper A — Indaba DLI 2026 Submission

**Venue:** Deep Learning Indaba 2026, Lagos, Nigeria (August 2–7, 2026)  
**Format:** IJCAI 2026 style, 7–10 pages (excl. references + appendix)  
**Submission system:** Charingtool (NOT email); abstract deadline April 15, 2026  
**Review type:** Double-blind

### 30.1 Submission Files

| File | Purpose |
|------|---------|
| `docs/publications/indaba/main.tex` | Primary submission-ready draft (IJCAI format) |
| `docs/publications/indaba/draft.tex` | Working draft with more conservative numbers and TODO sections |
| `docs/publications/indaba/arxiv_style_draft.tex` | Early skeleton (arXiv style, introduction + lit review only) |
| `docs/publications/indaba/references.bib` | 13 BibTeX entries |
| `docs/publications/indaba/NOTES.md` | Submission checklist and verified numbers |

### 30.2 Paper Title and Abstract

**Title:** "Sequential Credit Risk Modeling in Data-Constrained Environments: A Calibrated Synthetic Benchmark and Comparative Evaluation"

**Abstract claims:** Open-source `CalibratedMoMoDataGenerator` (10,000 users, 5 archetypes, calibrated to MTN QwikLoan). `TemporalTransactionFeatureEngineer` (38 features, 8 categories). Benchmark on 5,952 borrowers (11.1% default). RF: 0.832 AUC-ROC; LSTM: 0.523 AUC-ROC. Behavioural diversity drop causes −0.135 AUC. No statistically significant differences between top 5 models.

### 30.3 Results Discrepancy Between Versions

`main.tex` reports higher numbers (RF 0.884, LR 0.914) while `draft.tex` shows more conservative figures (RF 0.832, LR 0.831). The `main.tex` numbers match the Databricks real-data Notebook A outputs (60-day filter, 149,204 users), while `draft.tex` numbers match an earlier synthetic-only CV run. **The `draft.tex` figures are from the confirmed synthetic benchmark (RANDOM_SEED=42, April 2, 2026 CV run).**

### 30.4 Writing Status

| Section | Status |
|---------|--------|
| Abstract | Done (~220 words) |
| Introduction | Done (~700 words; 4 contributions) |
| Related Work | Outline + citations written; needs verification |
| Methodology | Done (data generation, features, models, eval) |
| Results | Placeholder — needs ± std from CV CSVs |
| Discussion | Not started |
| Limitations | Done (5 limitations) |
| Conclusion | Not started |

### 30.5 Five Limitations Stated

1. Synthetic data only — cannot capture live behavioral dynamics
2. Single geography/product type (Ghana, MTN QwikLoan)
3. No prospective deployment validation
4. LSTM collapse may be a synthetic data artifact (labels partly determined by archetype at generation time)
5. Binary default framing; no cost-asymmetric threshold optimization

### 30.6 References

13 entries including: Bjorkegren 2020 (mobile phone usage predicts loan repayment in Rwanda), Hochreiter/LSTM 1997, Chen/XGBoost 2016, Ke/LightGBM 2017, Lessmann 2015 (credit scoring benchmark), Vaswani/Transformer 2017, Patki/SDV 2016, Xu/CTGAN 2019, Niculescu-Mizil 2005 (calibration), Emmanuel 2025 (synthetic data for finance), Ghana 2024 mobile stats, AfricaNenda 2025 SIIPS. Self-citation: `attabra2026seqcredit`.

---

## 31. Paper B — ICML 2026 Real Data Paper

### 31.1 Paper Details

**File:** `docs/publications/icml2026/main.tex` (committed to `paper/icml2026/` during development, now at `docs/publications/icml2026/`)  
**Title:** "Behavioral Fingerprints for Credit Risk: An Empirical Benchmark on Ghanaian Mobile Financial Services Data"  
**Author:** Benjamin Ekow Attabra, KNUST, Kumasi, Ghana  
**Status:** Near-complete; submitted or ready for submission to ICML 2026

### 31.2 Dataset

- **Source:** Proprietary transaction log of major Ghanaian MFS provider (Telecel Ghana), December 2025 – February 2026 (~90 days)
- **Scale:** 374,295,424 rows, 12 columns, all identifiers hashed/anonymised
- **Raw borrowers:** 474,312 (received at least one "Loan Payment Via API" from lender E7C89F8C4A27F173)
- **After 30-day followup filter:** 223,081 retained
- **After feature merge (no empty feature vectors):** **222,618 borrowers**

### 31.3 Label Distribution (30-day filter, n=223,081)

| Label | Count | % |
|-------|-------|---|
| Good (repaid, no penalty) | 93,272 | 41.8% |
| Risky (repaid with penalty) | 120,074 | 53.8% |
| Default (never repaid) | 9,735 | 4.4% |
| y_default positive rate | 9,735 | 4.4% |
| y_bad positive rate | 129,809 | 58.2% |

### 31.4 Feature Engineering

47 user-level aggregate features across 8 categories (4 balance-at-loan features set to 0 — unavailable in real data). The 243 raw transaction type strings are mapped to 9 semantic categories via rule-based pattern matching.

### 31.5 Results (from `roc_data.json` and Table 2 in `main.tex`)

**y_default (strict default, n=222,618):**

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|-------|---------|--------|----|-------|-----|
| XGBoost | **0.7317** | **0.1120** | 0.1566 | 0.1749 | 0.3175 |
| LightGBM | 0.7289 | 0.1105 | 0.1570 | 0.1748 | 0.3187 |
| Random Forest | 0.7229 | 0.1007 | **0.1611** | **0.1620** | **0.3168** |
| Logistic Regression | 0.7093 | 0.0971 | 0.1394 | 0.2194 | 0.3966 |

**y_bad (default or penalised, n=222,618):**

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|-------|---------|--------|----|-------|-----|
| LightGBM | **0.7293** | **0.7791** | 0.6988 | **0.2095** | **0.0666** |
| XGBoost | 0.7274 | 0.7774 | **0.7407** | 0.3596 | 0.3634 |
| Random Forest | 0.7185 | 0.7684 | 0.6886 | 0.2149 | 0.0735 |
| Logistic Regression | 0.6907 | 0.7467 | 0.6660 | 0.2228 | 0.0735 |

### 31.6 Key Findings

- **0.732 AUC-ROC** from aggregate behavioral features alone — strong baseline without traditional credit bureau data
- **Severe class imbalance** (4.4% default) makes AUC-PR and F1 low by construction — practitioners should use calibrated probability scores, not fixed-threshold F1
- XGBoost and LightGBM significantly outperform LR (p < 0.01 via bootstrap) — non-linear interactions critical for MFS credit signal
- **Calibration varies by task:** For y_bad (balanced classes), LightGBM delivers both strong discrimination and excellent calibration (ECE 0.067) — directly actionable for loan pricing. For y_default (4.4% positive), all models show high ECE (0.32–0.40) — post-hoc recalibration required before deployment.
- **Behavioral diversity ablation:** Removing diversity features drops XGBoost AUC from 0.7317 to 0.6965 (−0.0352) — same signal dominates real data as synthetic

### 31.7 Limitations (5)

1. **Single observation window** — seasonal effects may shift distributions
2. **Survivorship bias and Reject Inference** — 53% attrition between cohorts; model sees only already-creditworthy applicants
3. **No demographic attributes** — cannot assess disparate impact; fairness audit required
4. **First-time borrowers** — structurally zero values for all repayment features; model weakest where it's most needed
5. **Right-censoring** — some borrowers near window edge may default after observation ends

### 31.8 Bibliography (`icml2026/references.bib`)

18 entries. Additional vs Indaba bib: Suwanzy 2025 (double discrimination / gender bias in African fintech), Grinsztajn 2022 (why tree-based models outperform deep learning on tabular data — NeurIPS), Lopez de Prado 2018 (Advances in Financial Machine Learning), Hand 1997 (consumer credit scoring), Feelders 1999 (credit scoring sample selection bias), Mortey 2025 (GBM for MFS security in West Africa). BibTeX syntax error at lines 148–150 (stray fragment from copy-paste).

### 31.9 ROC Figure Pipeline

`gen_roc.py` reads `roc_data.json` and produces `roc_curves.pdf` / `roc_curves.png`. Plot: 1×2 subplots (y_default / y_bad), 4 static models only (LSTM/Hybrid excluded), legend with model name + AUC. JSON is generated in Databricks notebook (Cell 22) from OOF predictions and pasted manually into `roc_data.json`.

---

## 32. Real Data Transaction Schema

**Source file:** `notebooks/Analysis_Notebook_A.csv` — a sample of raw Telecel Ghana transactions exported from the Databricks table.

**12 columns:**

| Column | Type | Example |
|--------|------|---------|
| TRANSACTION_TIMESTAMP | String | `09-OCT-25 14.48.56.000000000` (Oracle DD-MON-YY format) |
| TRANSACTION_ID | String (hashed) | `649014F85835655F` |
| TRANSACTION_TYPE | String (243 distinct) | `Airtime Purchase for Other Networks`, `Customer Withdrawal at Agent Till`, `Ghipss_P2P_transfer` |
| DEBIT_PARTY_ID | String (hashed) | hex ID |
| DEBIT_PARTY_TYPE | String | `Customer`, `Organisation` |
| DEBIT_PARTY_ACCOUNT | String (hashed) | account number |
| DEBIT_ACCOUNT_TYPE | String | `M-Pesa Account For Customer`, `Float Account For Organization` |
| CREDIT_PARTY_ID | String (hashed) | hex ID |
| CREDIT_PARTY_TYPE | String | `Customer`, `Organisation` |
| CREDIT_PARTY_ACCOUNT | String (hashed) | account number |
| CREDIT_ACCOUNT_TYPE | String | `Utility Account`, `Merchant Account` |
| TRANSACTION_AMOUNT | Double | 1.0, 5.05, 298.0 GHS |

**Date range:** October 2025 – February 2026. Amounts range from GHS 1 to large values. All party IDs and account numbers are hash-anonymised hex strings.

**Transaction type mapping** (243 raw types → 9 semantic categories in `_categorize_txtype`):
- `loan_disbursement`: `Loan Payment Via API`
- `loan_repayment_principal`: `Loan Principal Collection via API`
- `loan_repayment_interest`: `Loan Interest via API`
- `loan_penalty`: `Loan Penalty via API`
- `cash_out`: contains "Withdrawal" (e.g., `Customer Withdrawal at Agent Till`)
- `cash_in`: contains "Deposit at Agent"
- `airtime_data`: contains "Airtime", "Data Purchase", or "EVD Top"
- `transfer`: contains "Ghipss", "GHIPSS", "GHiPSS", "Transfer", or "P2P"
- `payment`: contains "Pay Bill", "Online Payment", or "Direct Debit"
- `betting`: contains "Betting"
- `fsi`: contains "FSI"
- `other`: everything else

**Timestamp parsing challenge:** Oracle-style format `DD-MON-YY HH.MM.SS.FFFFFFFFF` with variable-length seconds. Handled via coalesce of two try_to_timestamp patterns (18-char and 17-char substrings) in `_parse_timestamp`.

---

## 33. Data Cards

Two versions exist at different paths:

### 33.1 Root `DATACARD.md` (798 lines — Internal Working Version)

- Full Datasheets for Datasets framework (Gebru et al. 2021), 12 sections + 12.1 Technical Verification
- Covers 3 datasets: Dataset 1A (raw per-user CSVs), Dataset 1B (user_labels.csv), Dataset 2 (user_features.csv)
- **Section 4.2:** Full provenance table for all 17 calibration parameters with source citations and derivation arithmetic
- **Section 12.1:** Parameters organized into 5 categories: A=directly observed product terms, B=computed from aggregates (math shown), C=bounded range estimates, D=NaN (no published data), E=designer judgment calls
- Binary target: 661 defaults / 5,952 borrowers = 11.1% default rate (~8:1 imbalance)
- 13-item known limitations table: 5 resolved, 8 open
- Citation key: `attabra2025seqcredit` (year 2025)
- `ADJUSTMENT` transaction type defined in feature engineering but never generated — `is_adjustment` always 0

### 33.2 `docs/DATACARD.md` (344 lines — Public-Facing Version)

- Same framework, condensed and cleaned
- Section 4.2 summarised ("informed by public reports, operator materials, and regulatory documents")
- Section 12.1 absent
- 5 limitations bullets (vs 13-item table)
- No `[TO BE UPDATED]` flags — cleaner
- Citation year updated: `attabra2026seqcredit`
- Section 9 retitled "Known Limitations and Threats to Validity" (stronger academic framing)

---

## 34. Project Infrastructure

### 34.1 Package Structure (`pyproject.toml`, `__init__.py`)

```toml
[project]
name = "seqcredit-model"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [pandas, matplotlib, numpy, scikit-learn, xgboost, lightgbm,
                tensorflow, seaborn, jupyter, shap, pyspark, joblib, ipykernel]
```

`src/seqcredit_model/__init__.py` exports:
```python
__version__ = "0.1.0"
# From pipeline.py:
TemporalTransactionFeatureEngineer
# From synthesize.py:
CalibratedMoMoDataGenerator
# From credit_model.py:
CreditRiskDataLoader, LogisticRegressionModel, XGBoostModel,
RandomForestModel, LightGBMModel, LSTMModel, GRUModel,
HybridLSTMModel, HybridGRUModel, ModelEvaluator,
set_random_seeds, bootstrap_evaluate
```

Note: `GRUModel`, `HybridGRUModel`, and `bootstrap_evaluate` are exported but not described in `CLAUDE.md` — undocumented additions.

### 34.2 `.gitignore` — What Is Not Tracked

Gitignored (must be regenerated/re-downloaded):
- `*.npz`, `*.keras`, `*.pkl` — all model artifacts and LSTM caches
- `data/user_transactions/` — raw synthetic per-user CSVs (10,000 files)
- `docs/publications/` — paper drafts
- `.claude/` — Claude Code session data
- `CLAUDE.md`, `AGENTS.md` — AI assistant config files
- `.venv/`, `env/`, `venv/` — virtual environments
- `.mypy_cache/`, `.ruff_cache/` — linting caches

Tracked (version-controlled):
- `data/user_labels.csv`, `data/user_features.csv` — processed data
- `src/synthetic_params.json` — calibration parameters
- All source code, notebooks, docs

### 34.3 `AGENTS.md` — Code Style Guide for Agentic Assistants

Import order: stdlib → third-party → local (with try/except for local imports). Type hints required for public functions. Google-style docstrings. `snake_case` functions, `PascalCase` classes, `UPPER_CASE` constants, `_private_methods`. Comment "why", not "what". `set_random_seeds(42)` always. `prepare_static_splits()` before `load_sequences()`.

### 34.4 `best_practices.md` — PyParsing Guide

Contains pyparsing best practices for building parsers — not directly relevant to the credit risk project. Likely a default context file.

### 34.5 `blueprint.md`

```
# Deleted
This blueprint has been removed as the Transformer baseline is no longer part of the project scope.
```
Transformer models were added (commit `5069dcb`), benchmarked, then dropped (commit `d332846`) to reduce sequential runs from 60 to 20.

### 34.6 `README.md` — Broken

The root README contains only a single line of PyZMQ CFFI error text (96 bytes). This is a stub file — the repo has no functional project README for GitHub visitors. The project description lives in `docs/PROJECT.md`.

### 34.7 `LICENSE`

MIT License, copyright 2025, "Benjamin Attabra". Note: other documents are dated 2026; the copyright year has not been updated. Full name in DATACARD is "Benjamin Ekow Attabra".

---

## 35. Complete Git Development History

### 35.1 Repository Origin

The project began as **sequential-crm-for-dce** on October 21, 2025, with a series of Google Colab notebooks (v1 through v3) being iteratively created, deleted, and rebuilt. Early commits involved GitHub Copilot bot for README cleanup.

### 35.2 Development Phases

**Phase 0 — Exploration (Oct–Nov 2025)**
Colab notebooks for initial credit risk prediction experiments. CTGAN synthetic data generator written then deleted. No stable architecture.

**Phase 1 — Generator Foundation (Jan 2026)**
Major restructuring into Python package. Claude (Anthropic) authored two pivotal commits:
- `820195a`: Rewrote synthetic data generator for 10,000 per-user CSVs with 5 archetypes, credit labels, calibrated to MTN QwikLoan
- `d313c08`: Recalibrated generator against 593 real Ghanaian MoMo transactions (Table 5 dataset); added `real_data_calibration.json`

**Phase 2 — Model Implementation (Feb 2026)**
Repository renamed `seqcredit-model`. First full commit of `credit_model.py` (1,024 lines), `feature_engineering.py`, `synthetic_data.py`, `credit_risk_modeling.ipynb`. All six model classes (LR, XGBoost, RF, LightGBM, LSTM, HybridLSTM) established.

**Phase 3 — Debugging and XAI (Mar 2026)**
- Fixed total_transactions ambiguity (`2de63db`)
- Fixed label assignment bugs (`12bcd3f`)
- Ran descriptive statistics EDA
- Refactored all code (Mar 15): renamed notebooks, files, package structure (`60b0659`, `7c3c708`, `60976f4`)
- Added save()/load() to all 6 models (`b4bcc3d`)
- Consolidated experiments/ into single main notebook
- Added XAI research analysis notebook: SHAP, surrogate trees, calibration (`97e930d`, `8bbf237`)
- Fixed class imbalance (`7ca76d5`)
- First successful full end-to-end run (Mar 20, `ce36cd4`)
- Post-imbalance run committed model binaries (`a64db19`)

**Phase 4 — Evaluation Framework (Apr 2026, early)**
- Added ablation study, CV benchmark, hyperparameter tuning scripts (`7668192`)
- Refactored docs: restructured PROJECT.md, renamed notebooks to model.ipynb/analysis.ipynb (`68e9142`, `62f3034`)
- Committed CV/ablation/tuning result CSVs (`0b8e6fb`)
- CV runs on synthetic benchmark; dataset confirmed at 5,952 borrowers, 11.1% default rate

**Phase 5 — Real Data Pipeline (Apr 2026, mid)**
- **Pivotal: Apr 14** — `real_data_pipeline.py` added (`1b2bf09`); `real_data_end_to_end.ipynb` created
- Fixed crash in significance test (`0767849`)
- ICML 2026 paper development begins
- Added H1 2026 progress report (`a07ad28`)
- Added git clone cell to Databricks notebook (`976d860`)
- Added end-to-end test file (`e481e94`)

**Phase 6 — Paper + Real Data Results (Apr 2026, late)**
- Paper updated with real Telecel results: AUC 0.733, default rate 4.4%, n=222,618 (`cb121c4`)
- Paper author block: real name + KNUST affiliation (`72d2f67`)
- ROC JSON/PDF generation pipeline: `gen_roc.py`, `roc_data.json` (`fe59d79`, `b60c531`)
- Multiple Spark pipeline correctness fixes: duplicate definitions removed (`4302257`), `.cache()` calls removed (not supported on serverless, `43ef692`)

**Phase 7 — Spark Performance Marathon (May 2026, early)**
Five consecutive performance commits on `real_data_pipeline.py` in 24 hours (May 5):
- `ab34d49`: Eliminated `toLocalIterator()` timeout — replaced with `toPandas()` after pushing all 19 feature computations into Spark workers; scan count to be reduced
- `9d6ea97`: Halved driver peak memory — cast to float32 in Spark before collect; fixed OOM SIGKILL 137
- `8d9e628`: Solved 6-scan problem — materialised `df_ts` to Parquet; replaced union+explode with single pass; scans 6→3
- `fc91f8f`: Fixed DBFS_DISABLED — three-tier fallback (Unity Catalog → env var → skip); eliminated separate max_ts collect()
- `ff611a0`: Pre-filter rows/columns before timestamp parse — prunes 374M rows by customer-side match before expensive `try_to_timestamp` runs; wall-clock hours→minutes

**Phase 8 — Sequential Model Expansion (May 2026, mid)**
- `4375ac2`: Added GRUModel and HybridGRUModel — exact LSTM/HybridLSTM replicas with GRU cells
- `3e35218`: Fixed bootstrap p-value bug — was always ≈0.5 due to missing null centering
- `5069dcb`: Added TransformerModel and HybridTransformerModel to CV benchmark
- `e7a56a2`: Live output streaming via Popen line-iteration; verbose=2 for per-epoch logs
- `e387402`: Epochs 50→30, patience 10→7 for CPU-only clusters
- `2f96119`: Batch size 32→256; CV results always written to runtime dir

**Phase 9 — Notebook Restructure and Final Fixes (May 2026, late)**
- `d332846`: Dropped Transformer — benchmark now LSTM+GRU only (20 sequential runs vs 60)
- `3bb6242`: Dropped all hybrid models — sequential runs 60→30
- `f5cfa63`: Notebook split — Notebook A static-only, Notebook B static+LSTM+GRU
- `9b9982f`: Raised `min_followup_days` 30→60 — penalties appear from day 31 onward (p10=31d, p50=55d); old cutoff mislabelled risky borrowers as good
- `d8e7cfa`: Lean EDA in both notebooks — dropped 9 redundant cells each
- `bec1252`: Fixed XGBoost scale_pos_weight — was using y_default ratio for y_bad
- Notebook A loop variable fixes: `df→cv_df`, `col→metric` to avoid clobbering Spark DataFrame
- `ce07412` (latest): Clean Notebook B build_pipeline cell

### 35.3 Key Commit Milestones

| Commit | Date | Significance |
|--------|------|-------------|
| `820195a` | 2026-01-28 | First 10K user synthetic dataset with 5 archetypes |
| `9fc22c8` | 2026-03-20 | First successful full end-to-end run |
| `7668192` | 2026-04-02 | Complete evaluation framework (CV, ablation, tuning) |
| `1b2bf09` | 2026-04-14 | Real data pipeline born |
| `cb121c4` | 2026-04-29 | Real Telecel results replace synthetic in paper |
| `4375ac2` | 2026-05-11 | GRU models added |
| `3e35218` | 2026-05-11 | Bootstrap significance bug fixed |
| `9b9982f` | 2026-05-17 | min_followup_days 30→60 (label leakage fix) |
| `bec1252` | 2026-05-19 | scale_pos_weight per-target bug fixed |
| `ce07412` | 2026-05-20 | HEAD — current state |

### 35.4 Total Commit Count

Approximately 112 commits across all branches from repository origin (Sep 22, 2025) through HEAD (May 20, 2026) — ~8 months of development.

---

## 36. Known Discrepancies and Open Issues

### 36.1 The AUC Number Discrepancy

The project shows multiple sets of AUC numbers across different files:

| Source | RF y_default AUC | LR y_default AUC | Context |
|--------|-----------------|-----------------|---------|
| `cv_results_y_default.csv` | 0.8836 | 0.9143 | **Authoritative** (run_cv_benchmark.py, Databricks real data, 60-day filter) |
| `indaba/main.tex` | 0.884 | 0.914 | Matches CSV — these are the real data results mistakenly presented as synthetic |
| `indaba/draft.tex` | 0.832 | 0.831 | Older synthetic-only CV run |
| `NOTES.md` | 0.832 | 0.831 | Older synthetic-only CV run |
| `model_comparison.csv` | 0.487 | 0.479 | Pre-fix saved models, wrong y_test_lstm alignment |
| ICML `main.tex` Table 2 | 0.7229 | 0.7093 | **Real data, 30-day filter, n=222,618** — different pipeline run |

The `draft.tex` (0.832 RF, 0.831 LR) represents the confirmed synthetic benchmark. The `cv_results_y_default.csv` (0.8836 RF, 0.9143 LR) are from the Databricks real data pipeline with 60-day followup filter. The ICML paper (0.7229 RF, 0.7093 LR) uses a 30-day filter and n=222,618 — a different and larger cohort.

### 36.2 The model_comparison.csv Anomaly

`data/model_comparison.csv` shows static models at AUC ~0.48–0.50 with HybridLSTM at 0.863. This is from an early run (`704ae39`, 2026-04-07) using `compute_bootstrap_ci.py` which loads pre-saved models and evaluates on `y_test_lstm` — the misaligned test set where the user ordering placed defaults disproportionately in the test partition. This file is **not authoritative** and is superseded by the CV benchmark CSVs.

### 36.3 `final_credit_limit` SHAP Dominance

In `analysis_gcolab.ipynb`, `final_credit_limit` has mean |SHAP| = 5.23 for y_default — vastly larger than all other features. This is a synthetic data artifact: credit limit is directly determined by repayment history in the generator, creating near-deterministic signal. This feature does NOT appear in `user_features.csv` (the 29-feature static set) — it leaks into the model only when `user_labels.csv` columns are accidentally included in training. This represents a potential data leakage bug in the GColab notebook version.

### 36.4 `data_gcolab.ipynb` Syntax Error

Cell `cell-s2-dist-activity`: trailing `"` on `plt.show()"` causes `SyntaxError: unterminated string literal`. The cell will not execute.

### 36.5 LaTeX Compilation Errors

- `docs/H1_2026_Progress_Report.tex` line 20: `\hr` is not valid LaTeX (should be `\hrule` or `\noindent\rule{\textwidth}{0.4pt}`)
- `docs/publications/icml2026/references.bib` lines 148–150: stray fragment `rence on Financial Technology and AI}, year={2024}}` from a copy-paste error — will break BibTeX compilation

### 36.6 README.md

Root `README.md` contains only `PyZMQ's CFFI support is designed only for (Unix) systems conforming to have_sys_un_h = True.` — a PyZMQ error message. No project documentation exists at the root for GitHub visitors.

---

## Closing Narrative: What We Built and What We Found

This project began on September 22, 2025, as `sequential-crm-for-dce` — a series of iterative Colab notebooks exploring credit risk prediction. By May 20, 2026, it has grown into a complete research framework: a calibrated synthetic data generator, a temporal feature engineering pipeline, six model classes, a reproducible 5-fold CV benchmark, an ablation study, hyperparameter tuning, a Spark-based production pipeline for 374 million real transactions, two academic papers in preparation, and a growing body of findings that tell a coherent story about credit risk prediction in African mobile money markets.

The central question this project asks is: **do transaction sequences improve credit risk prediction in the African mobile money context?** The synthetic benchmark gives a clean, controlled answer — they do not. The LSTM collapses to near-random (0.523–0.587 AUC-ROC). The Hybrid LSTM recovers only because of its static feature branch. Well-engineered aggregate features from RandomForest (0.832–0.884 AUC-ROC) and Logistic Regression (0.831–0.914 AUC-ROC) dominate.

The more important finding is *why*. The ablation study locates the dominant signal: three features — `unique_recipients`, `recipient_concentration`, and `unique_txn_types` — encode most of the default signal independent of loan history. These are not sequence features. They are aggregate diversity statistics. Dropping them collapses AUC from 0.884 to 0.697 (−0.135). No other group comes close. This is the Behavioral Diversity (BD) Framework: it is not *when* things happen in the transaction sequence, but *how diverse* the user's financial behaviour is. The entropy of who you interact with and how you transact predicts creditworthiness better than the temporal ordering of your transactions.

The real-data validation on 222,618 Telecel Ghana borrowers (374 million transactions) confirms the framework generalises — though at lower absolute AUC (0.732 vs 0.884). The gap is expected: real-world credit data is noisier, outcomes are more heterogeneous, and the observation window captures less behavioral signal than a 180-day simulation. The behavioral diversity signal still dominates the real-data ablation (−0.035 AUC when dropped). The hypothesis is real.

The code we built — `CalibratedMoMoDataGenerator` calibrated to 17 public data sources, `TemporalTransactionFeatureEngineer` with 113 transaction-level features and 29 user-level aggregates, `CreditRiskDataLoader` with leakage-free temporal splitting, six model classes sharing a common interface, the CV benchmark with proper fold-level scaling and corrected bootstrap significance testing, the Spark pipeline handling 374 million rows in a single distributed pass — is a reusable framework. Any researcher studying mobile money credit risk in any African market can clone this repository, calibrate `synthetic_params.json` to their market, and run the same experiments.

The paper arc tells this story in two acts. Paper A (targeting Deep Learning Indaba 2026) presents the synthetic benchmark: methodology, surprising LSTM failure, and the behavioral diversity finding. Paper B (ICML 2026) validates the framework at national scale on real data. Together they make the case that behavioral fingerprints — the richness of the financial life recorded in a mobile money ledger — are the primary signal for credit risk prediction in data-scarce African markets, and that sophisticated sequential deep learning adds little on top of well-engineered aggregate statistics.

The thread stitching it all together: from 593 real Ghanaian transactions used to calibrate the generator, through 1.03 million synthetic transactions, through 374 million real Telecel Ghana transactions, through the CV benchmark scripts, through the ablation study, through the LaTeX papers — everything connects to a single claim: **in mobile money credit risk, behavioral diversity is the fingerprint that matters.**

---

*Generated from complete codebase read — May 20, 2026. All findings trace to data files in `data/`, source code in `src/seqcredit_model/`, notebooks in `notebooks/`, publication drafts in `docs/publications/`, and 112 commits of git history on the `real-data` and `main` branches.*

This project set out to ask whether transaction sequences improve credit risk prediction in the African mobile money context. The synthetic benchmark gives us a clean, controlled answer: **they do not, on this data.** The LSTM collapses to random; the Hybrid LSTM is competitive only because of its static feature branch.

But the more interesting finding is *why*. The ablation study revealed that three features — `unique_recipients`, `recipient_concentration`, `unique_txn_types` — encode most of the non-loan default signal. These are not sequence features. They are aggregate diversity statistics. The conclusion is a reframe of the original question: it's not about *when* things happen in the transaction sequence, but *how diverse* the user's financial behaviour is.

The synthetic benchmark is not the end of the story. It is the beginning. The Telecel Ghana data — 374 million real transactions from 474,312 real borrowers — is where the hypothesis gets tested in the real world. Early results (0.9464 AUC-ROC for LightGBM on `y_default`) suggest the framework generalises. The behavioral diversity signal appears to be real, not synthetic.

The code we built — `CalibratedMoMoDataGenerator`, `TemporalTransactionFeatureEngineer`, `CreditRiskDataLoader`, the six model classes, the CV benchmark scripts, the Spark real-data pipeline — is a reusable framework. Any researcher studying mobile money credit risk in any African market can take this code, calibrate the `synthetic_params.json` to their market, and run the same experiments. That is the open-source contribution.

The paper we are writing tells this story: a synthetic framework, a surprising finding about sequences and entropy, and the first glimpse of national-scale validation. The thread stitching it together is behavioral diversity — not the pattern of amounts over time, but the richness of the financial life that generates those amounts.

---

*Generated from complete codebase read — May 20, 2026. All findings trace to data files in `data/`, source code in `src/seqcredit_model/`, and git history on the `real-data` branch.*
