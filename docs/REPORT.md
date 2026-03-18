# Project Report: Sequential Credit Risk Modeling

**Project:** Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments  
**Author:** Attabra Benjamin Ekow  
**Last Updated:** March 2026

---

## Executive Summary

This project develops a credit risk prediction system using mobile money transaction data. It implements both traditional static models (Logistic Regression, XGBoost, Random Forest, LightGBM) and sequential deep learning models (LSTM, Hybrid LSTM) to predict loan default risk. A key feature is the synthetic data generator calibrated to real Ghanaian mobile money patterns, enabling model development without privacy concerns.

**Critical Bug Fixed (March 2026):** Discovered that models were training on stale label data (`summary_extended.csv`) that didn't match the current transaction files. This has been corrected by using `user_labels.csv` which contains the correct, current labels.

---

## Project Structure

```
seqcredit-model/
├── src/seqcredit_model/          # Source package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Path constants
│   ├── synthetic_data.py         # Synthetic data generation
│   ├── feature_engineering.py    # Transaction feature extraction
│   └── credit_model.py           # Models + data loader + evaluator
├── notebooks/                    # Jupyter notebooks
│   ├── credit_risk_model.ipynb   # Primary modeling notebook (all 6 models, saves models/arrays)
│   ├── credit_risk_analysis.ipynb# Research analysis: y_bad, SHAP, surrogate tree, calibration
│   └── data_analysis.ipynb       # Descriptive statistics notebook
├── data/                         # Generated data
│   ├── synthetic_params.json    # Calibration parameters
│   ├── user_features.csv        # Aggregated user features
│   ├── user_labels.csv          # Credit risk labels
│   └── user_transactions/       # Per-user CSVs (10,000 users, regenerable)
├── models/                       # Persisted trained models (created by model.save())
├── docs/
│   ├── REPORT.md                # This file
│   ├── SESSION.md               # Development log
│   └── DATACARD.md              # Data documentation
├── AGENTS.md                     # AI coding agent guide
├── CLAUDE.md                     # Claude Code guide
└── requirements.txt              # Python dependencies
```

---

## Data Pipeline

### 1. Synthetic Data Generation

`CalibratedMoMoDataGenerator` creates realistic mobile money transaction data:

- **10,000 users** with 100 transactions on average
- **5 credit archetypes:**
  - Non-borrower (40%)
  - Responsible borrower (35%)
  - Occasional borrower (15%)
  - Risky borrower (8%)
  - Defaulter (2%)
- **Transaction types:** TRANSFER, DEBIT, PAYMENT, PAYMENT_SEND, CASH_OUT, CASH_IN, ADJUSTMENT, CREDIT (loan disbursement), LOAN_REPAYMENT
- **Calibrated to real Ghanaian mobile money patterns**

Output: `data/user_transactions/USER_XXXXXX.csv` and `data/user_labels.csv`

### 2. Feature Engineering

`TemporalTransactionFeatureEngineer` extracts 8 feature categories per transaction:

1. Transaction-level static features (amount transforms, categories)
2. Categorical encodings (transaction type one-hots)
3. Temporal features (hour, day, cyclical encodings)
4. Balance dynamics (change, ratios, depletion indicators)
5. Sequence lookback features (rolling counts, averages)
6. Rolling window statistics (time-based)
7. Behavioral patterns (recipient diversity, repeated transactions)
8. Derived risk indicators (unusual timing, rapid transactions)

Output: `data/user_features.csv` (one row per user with ~50 aggregate features)

### 3. Target Definition

- `credit_risk_label`:
  - `-1` = Non-borrower (excluded from training)
  - `0` = Good (repaid on time)
  - `1` = Late (repaid after term)
  - `2` = Default (failed to repay)
- **Primary target (`y_default`):** `1` if default (label=2), else `0`
- **Secondary target (`y_bad`):** `1` if late or default (label ∈ {1,2}), else `0`

---

## Models

### Model Interface

All models share a common interface:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
results = model.cross_validate(X, y, n_splits=5)
```

### Logistic Regression (`LogisticRegressionModel`)

- Input: Scaled user-level static features (29 features)
- Uses `class_weight='balanced'` for imbalance handling
- Provides feature coefficients for interpretability

### XGBoost (`XGBoostModel`)

- Input: Same static features as LR
- Uses `scale_pos_weight` for class imbalance
- Hyperparameters: 200 estimators, max_depth=5, learning_rate=0.1

### Random Forest (`RandomForestModel`)

- Input: Same static features
- Uses `class_weight='balanced'` for imbalance handling
- Hyperparameters: 200 estimators, max_depth=10, min_samples_split=5

### LightGBM (`LightGBMModel`)

- Input: Same static features
- Uses early stopping with validation set
- Hyperparameters: 200 estimators, max_depth=5, num_leaves=31

### LSTM (`LSTMModel`)

- Input: Padded transaction sequences (max_len=50, 38 features per transaction)
- Architecture: Masking → LSTM(64) → Dropout → LSTM(32) → Dense(16) → Dropout → Dense(1, sigmoid)
- Uses pre-padding so recent transactions are at sequence end

### Hybrid LSTM (`HybridLSTMModel`)

- Input: Both padded transaction sequences AND static user features
- Architecture: Dual-branch - LSTM processes sequences, Dense processes static features → Concatenate → Dense → sigmoid
- Combines sequential patterns with static user characteristics for improved prediction

---

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `CalibratedMoMoDataGenerator` | `synthetic_data.py` | Generate realistic transaction data |
| `TemporalTransactionFeatureEngineer` | `feature_engineering.py` | Extract features from transactions |
| `CreditRiskDataLoader` | `credit_model.py` | Load/merge/split data for models |
| `LogisticRegressionModel` | `credit_model.py` | Baseline static classifier |
| `XGBoostModel` | `credit_model.py` | Gradient boosting classifier |
| `RandomForestModel` | `credit_model.py` | Random Forest classifier |
| `LightGBMModel` | `credit_model.py` | LightGBM gradient boosting |
| `LSTMModel` | `credit_model.py` | Sequential deep learning model |
| `HybridLSTMModel` | `credit_model.py` | Hybrid LSTM + static features |
| `ModelEvaluator` | `credit_model.py` | Compare models, generate plots |

---

## Evaluation Metrics

- ROC-AUC (Area Under ROC Curve)
- PR-AUC (Area Under Precision-Recall Curve)
- F1 Score
- Precision / Recall
- Accuracy
- Confusion matrices
- Threshold analysis
- Brier Score (probability calibration quality)
- ECE — Expected Calibration Error (15-bin weighted mean |confidence − accuracy|)

---

## Research Analysis Notebook (`credit_risk_analysis.ipynb`)

`notebooks/credit_risk_analysis.ipynb` implements the full publishable research pipeline on top of trained models. It requires running `credit_risk_model.ipynb` first to generate saved models and LSTM arrays.

### Section 1 — Setup & Model Loading
Re-creates the same deterministic data splits (`random_state=42`) and loads all 6 saved models from `models/`. LSTM train/test arrays are loaded from `data/lstm_test_arrays.npz`.

### Section 2 — Secondary Target: `y_bad`
Evaluates all 6 models against `y_bad = 1[label ∈ {1,2}]`. Produces a side-by-side ROC-AUC / PR-AUC bar chart comparing `y_default` and `y_bad` performance across models.

### Section 3 — Permutation Importance
`sklearn.inspection.permutation_importance` (n_repeats=10, scored by AUC-ROC) applied to all 4 static models. Top-15 features plotted per model with error bars showing variability across repeats.

### Section 4 — SHAP Analysis
- `shap.TreeExplainer` on XGBoost and LightGBM
- Global beeswarm summary plots for `y_default` models
- Waterfall plots for one high-risk and one low-risk test user
- A second LightGBM is trained on `y_bad` for comparison; top-10 SHAP features are compared side-by-side between tasks

### Section 5 — Surrogate Decision Tree
- Teacher model: LightGBM (`y_default`)
- Fidelity curve: `DecisionTreeRegressor` trained at depths 2–8; R² and MAE to teacher plotted
- Final surrogate at depth 3: `plot_tree` visualisation + `export_text` human-readable rules

### Section 6 — Model Calibration
- Brier score and ECE (15-bin) for all 6 models
- Reliability diagrams (calibration curves) before calibration
- Post-hoc isotonic calibration for XGBoost and LightGBM: calibration split carved from training data, `IsotonicRegression` fitted, before/after curves and Brier/ECE improvement table

---

## Critical Bug Fix (March 2026)

### Issue Discovered

The models were training on `summary_extended.csv` which contained **stale labels** from a previous data generation run. This file had incorrect transaction counts and credit risk labels that didn't match the actual transaction files in `data/user_transactions/`.

**Evidence:**
- `summary_extended.csv`: USER_000000 had 21 transactions
- `user_transactions/USER_000000.csv`: Actually has 114 transactions
- Labels were from an older synthetic data generation

### Resolution

1. Identified that `user_summaries.csv` (output from synthetic data generator) contained **correct current labels**
2. Renamed `user_summaries.csv` → `user_labels.csv`
3. Deleted stale `summary_extended.csv` from the repository
4. Updated `config.py` to use `USER_LABELS_FILE` constant

---

## Usage

### Generate Synthetic Data

```bash
python -m seqcredit_model.synthetic_data
```

### Build Features

```bash
python -m seqcredit_model.feature_engineering
```

### Run Experiments

```bash
jupyter notebook notebooks/credit_risk_model.ipynb
```

---

## Dependencies

- Python 3.10+
- pandas, numpy
- scikit-learn
- xgboost
- lightgbm
- tensorflow
- matplotlib, seaborn
- jupyter
- shap (for SHAP analysis in `credit_risk_analysis.ipynb`)

Install: `pip install -r requirements.txt`

---

## Descriptive Statistics Notebook (`data_analysis.ipynb`)

`notebooks/data_analysis.ipynb` provides a systematic statistical summary of the synthetic dataset. It is organized into six sections:

### Section 1 — Dataset Overview
Dimensions of each data file (features, labels, transactions), column listings, and null value counts.

### Section 2 — Numerical Feature Statistics
- Full `describe()` table for all numeric user-level features, augmented with Coefficient of Variation (CV = std / |mean|)
- Top-15 features ranked by CV (relative variability)
- Violin plots for balance and activity features annotated with per-archetype mean (μ) and standard deviation (σ)

### Section 3 — Categorical Feature Statistics
Frequency tables (count + percentage) for every categorical variable:

| Variable | Source |
|----------|--------|
| `credit_archetype` | user_labels.csv |
| `credit_risk_label` | user_labels.csv |
| `TRANS. TYPE` | transaction sample |
| `LOAN_PROVIDER` | CREDIT transactions only |
| Day of week | transaction sample |
| Hour of day | transaction sample |

Each variable also has a bar chart showing its distribution.

### Section 4 — Key Numerical Distributions
- Transaction amount statistics (mean, std, CV, skew, kurtosis) and log-scale histogram
- IQR box plots by transaction type
- Loan disbursement and repayment summary stats separately

### Section 5 — Feature Correlations
Pearson correlation heatmap for 15 interpretable features plus the binary default target. Top-10 features ranked by absolute correlation with default are printed as a table.

### Section 6 — Per-Archetype Summary
- Mean ± std table for 8 key features across all five archetypes
- Styled gradient table of means for easy scanning

---

## Future Work

- [ ] Add hyperparameter tuning (GridSearch/RandomSearch/Optuna)
- [ ] Implement cross-validation with proper time-series splits
- [ ] Experiment with attention mechanisms for LSTM
- [ ] Add unit tests

---

## References

- Mobile money transaction patterns calibrated to Ghanaian data (MTN QwikLoan)
- Loan parameters: GHS 25-1,000, 6.9% interest, 30-day term