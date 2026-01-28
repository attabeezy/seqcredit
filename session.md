# Research Session Log: Mobile Money Fraud Detection Feature Engineering
**Date**: January 25, 2026
**Researcher**: Benjamin Ekow Attabra
**Institution**: Kwame Nkrumah University of Science and Technology
**Project**: Sequential vs. Static Credit Risk Modeling Using Transaction Data

---

## Session Overview

This session established the complete framework for a two-paper research project on feature engineering and modeling for mobile money fraud detection in Ghana. We progressed from initial planning through real data analysis to generating a fully calibrated synthetic multi-user dataset ready for research experiments.

---

## 1. Research Context

### Research Questions

**Paper A (Feature Engineering Framework):**
1. What features can be extracted from short, irregular transaction histories?
2. How do we validate feature quality in data-constrained environments?
3. What evaluation metrics distinguish good feature engineering frameworks?

**Paper B (Static vs Sequential Modeling):**
1. Do sequential models (LSTM) outperform static models (Logistic Regression)?
2. When does the additional complexity of sequential modeling provide meaningful gains?
3. What are the interpretability and deployment trade-offs?

---

## 2. Real Data Analysis

### Dataset Characteristics
- **User**: Account 71797604 (Benjamin Ekow Attabra)
- **Period**: February 18 - September 6, 2024 (200 days)
- **Transactions**: 482 total
- **Frequency**: 2.41 transactions per day (every 10 hours)
- **Total Volume**: GHS 15,810.40

### Key Patterns

**Transaction Amounts:**
```
Mean: GHS 32.80
Median: GHS 18.50
Distribution: Lognormal(mu=2.8421, sigma=1.0034)
```

**Transaction Types:**
```
TRANSFER: 52.9%
DEBIT: 27.2%
PAYMENT_SEND: 10.8%
CASH_OUT: 5.0%
PAYMENT: 4.1%
```

**Temporal Patterns:**
```
Weekend transactions: 32.2%
Night transactions (22:00-24:00): 8.1%
Afternoon (12:00-18:00): 36.1%
```

**Balance Behavior:**
```
Mean: GHS 305.88
Std: GHS 302.55
Range: GHS 4.50 - GHS 1,886.09
```

---

## 3. Feature Engineering Implementation

### Features Extracted: 113 temporal features across 8 categories

| Category | Count | Examples |
|----------|-------|----------|
| Transaction-level static | 10 | log_amount, is_large_txn, fee_to_amount_ratio |
| Categorical encodings | 8 | is_transfer, is_debit, is_payment_send |
| Temporal extraction | 17 | hour, day_of_week, is_weekend, hour_sin/cos |
| Balance dynamics | 23 | balance_change, amount_to_balance_ratio |
| Sequence features | 31 | last_5_avg_amount, cumulative_volume |
| Rolling windows | 28 | rolling_7d_count, rolling_30d_mean |
| Behavioral patterns | 4 | unique_recipients_so_far, is_self_transfer |
| Risk indicators | 8 | unusual_hour, rapid_transaction, risk_score |

### Usage

```python
import sys
sys.path.append('src')
from feature_engineering import TemporalTransactionFeatureEngineer

# Initialize and extract features
engineer = TemporalTransactionFeatureEngineer()
df_features = engineer.extract_all_features(df)

# Create user-level summary
user_summary = engineer.create_user_level_summary(df)
```

---

## 4. Synthetic Data Generation

### Dataset Statistics

```
Total Users: 2,000
├── Legitimate: 1,900 (95.0%)
└── Fraudulent: 100 (5.0%)

Total Transactions: 29,994
├── Legitimate: 28,513 (95.1%)
└── Fraud: 1,481 (4.9%)
    ├── Account takeover: 545 (36.8%)
    ├── Social engineering: 506 (34.2%)
    └── SIM swap: 430 (29.0%)

Temporal Coverage: 48 days (Jan 1 - Feb 18, 2024)
```

### Calibration Validation

| Metric | Real Data | Synthetic | Match % |
|--------|-----------|-----------|---------|
| Mean amount | GHS 32.80 | GHS 29.32 | 89% |
| Transfer % | 52.9% | 49.0% | 93% |
| Debit % | 27.2% | 25.0% | 92% |
| Weekend % | 32.2% | 29.8% | 93% |
| Night % | 8.1% | 9.5% | 85% |
| Cash-out % | 5.0% | 4.9% | 98% |

**Overall Calibration Grade: A (Excellent)**

### Usage

```python
import sys
sys.path.append('src')
from synthetic_data import CalibratedMoMoDataGenerator

generator = CalibratedMoMoDataGenerator(
    n_users=2000,
    avg_transactions_per_user=15,
    fraud_rate=0.05,
    start_date='2024-01-01',
    duration_days=180
)

df, users = generator.generate_dataset()
```

### Fraud Simulation

**Legitimate Users:**
- Consistent patterns based on real data
- Transaction types: 53% transfers, 27% debits, 11% payments
- Amounts: Lognormal distribution (mu=2.84, sigma=1.00)
- Frequency: ~2.4 transactions/day
- Night activity: 8%

**Fraudulent Users (3 Types):**
- **Account Takeover**: Sudden behavior change, 40% cash-outs, 30% night activity
- **Social Engineering**: Rapid sequences of large transfers
- **SIM Swap**: Rapid balance depletion via multiple cash-outs

---

## 5. Project Files

### Code
```
src/
├── __init__.py
├── feature_engineering.py    # Temporal feature extraction
└── synthetic_data.py         # Calibrated data generation
```

### Data
```
data/
├── transactions.xlsx - Table 1.csv    # Real data (105 transactions)
├── transactions.xlsx - Table 5.csv    # Real data (482 transactions)
├── engineered_features_real_data.csv  # 482 x 113 features
├── user_level_summary.csv             # 40+ aggregates
├── synthetic_momo_calibrated.csv      # 29,994 transactions
├── synthetic_user_profiles.csv        # 2,000 user profiles
├── real_data_calibration.json         # Calibration parameters
└── synthetic-momo-data.csv            # Legacy synthetic data
```

### Notebooks
```
notebooks/
├── credit_risk_prediction_v1a.ipynb
├── credit_risk_prediction_v1b.ipynb
├── credit_risk_prediction_v1c.ipynb
├── ctgan_syn_data_gen.ipynb
└── syn_data_gen.ipynb
```

---

## 6. Research Workflow

### Paper A: Feature Engineering Framework

**Weeks 1-2: Feature Extraction & Validation**
- Apply feature engineering to all 2,000 synthetic users
- Compute discriminative power metrics (mutual information, correlation)
- Redundancy analysis (correlation matrix, PCA)

**Weeks 3-4: Scenario Testing**
- Short History (5-10 transactions)
- Irregular Transaction Patterns
- Balance-Constrained Users
- High-Frequency Users
- Early Fraud Detection

**Weeks 5-6: Baseline Comparisons**
- Raw features only
- Simple statistics
- TSFresh automated extraction
- Proposed framework

**Weeks 7-8: Writing Paper A**

### Paper B: Static vs Sequential Comparison

**Weeks 9-10: Static Model Development**
- User-level aggregates (40+ features)
- Logistic Regression, Random Forest, XGBoost

**Weeks 11-12: Sequential Model Development**
- LSTM on transaction sequences (20 transactions x 25 features)
- Architecture: LSTM(64) -> Dropout -> LSTM(32) -> Dense(16) -> Dense(1)

**Weeks 13-14: Comparative Analysis**
- Performance comparison (AUC-ROC, Precision-Recall)
- Data regime analysis (performance vs sequence length)
- Interpretability analysis (SHAP, attention weights)
- Cost-benefit analysis

**Weeks 15-16: Writing Paper B**

---

## 7. Evaluation Scenarios

### Scenario 1: Short History Users (5-10 transactions)
```python
short_history_users = df.groupby('FROM ACCT').size()
short_users = short_history_users[short_history_users <= 10].index
```

### Scenario 2: Performance vs Sequence Length
```python
sequence_lengths = [5, 10, 15, 20, 30]
for seq_len in sequence_lengths:
    X_seq, y = prepare_sequences(df, user_ids, sequence_length=seq_len)
    auc = evaluate_lstm(X_seq, y)
```

### Scenario 3: Time-Aware Train/Test Split
```python
split_date = pd.to_datetime('2024-02-01')
train_df = df[df['TRANSACTION DATE'] < split_date]
test_df = df[df['TRANSACTION DATE'] >= split_date]
```

---

## 8. Key Decisions

1. **Two-Paper Strategy**: Paper A (feature engineering) + Paper B (model comparison)
2. **Real Data as Primary Validation**: 482 real transactions provide credibility
3. **113 Features Across 8 Categories**: Interpretable, efficient features
4. **5% Fraud Rate, 3 Types**: Matches typical mobile money fraud rates (3-7%)
5. **2,000 Users, ~15 Transactions Each**: Manageable size, matches "short history" focus

---

## 9. Current Status

### Completed
- Research planning (two-paper strategy)
- Literature review (12 exemplar papers)
- Real data analysis (482 transactions)
- Feature engineering framework (113 features)
- Synthetic data generation (2,000 users, 30K transactions)
- Calibration validation (85-98% match)

### Next Steps
1. Apply feature engineering to all synthetic users
2. Generate feature quality report
3. Create evaluation scenarios
4. Train baseline models
5. Implement LSTM architecture
6. Write papers

---

## 10. Quick Start

```python
import pandas as pd
import sys
sys.path.append('src')

# Load synthetic dataset
df = pd.read_csv('data/synthetic_momo_calibrated.csv')
df['TRANSACTION DATE'] = pd.to_datetime(df['TRANSACTION DATE'])

# Apply feature engineering
from feature_engineering import TemporalTransactionFeatureEngineer
engineer = TemporalTransactionFeatureEngineer()

# Process one user
user_df = df[df['FROM ACCT'] == 'USER_000001']
features = engineer.extract_all_features(user_df)

# Check fraud distribution
print(df['is_fraud'].value_counts())
print(df['fraud_type'].value_counts())
```

---

## 11. Citations for Papers

**Paper A (Feature Engineering):**
> "We validate our framework on a calibrated synthetic dataset (N=2,000 users,
> 29,994 transactions) generated from real mobile money transaction patterns
> in Ghana. Calibration parameters include transaction frequency (mu=10.01 hours),
> amount distribution (lognormal mu=2.84, sigma=1.00), and behavioral patterns
> (52.9% transfers, 27.2% debits, 32.2% weekend activity)."

**Paper B (Static vs Sequential):**
> "Our experimental dataset comprises temporal transaction sequences from 2,000
> synthetic users, calibrated against 482 real mobile money transactions from
> Ghana. The dataset includes 1,900 legitimate users and 100 fraudulent users
> (5% fraud rate) across three fraud types: account takeover, social engineering,
> and SIM swap attacks."

---

## Summary

**Key Achievements:**
- Working temporal feature engineering pipeline (113 features)
- Real transaction data analyzed (482 txns)
- Calibrated synthetic dataset (2,000 users, 30K transactions, 5% fraud)
- Excellent calibration match (85-98% across all metrics)
- Clear path to two publications

**Estimated Timeline:**
- Paper A: 8 weeks (feature evaluation + writing)
- Paper B: 16 weeks (modeling + analysis + writing)

---

*Session End Time: January 25, 2026*
