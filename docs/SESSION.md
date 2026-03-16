# Project Log: Sequential CRM for DCE - Credit Risk Modeling
**Date**: February 16, 2026
**Author**: Attabra Benjamin Ekow
**Project**: Building a credit risk prediction system using mobile transaction data.

---

## Project Overview

This project aims to explore how transaction data can be used to predict credit risk, focusing on both traditional (static) and time-aware (sequential) modeling approaches. I'm especially interested in how much value sequential models add given their complexity. The work involves a few key steps: understanding real transaction patterns, generating realistic synthetic data with different borrowing behaviors, engineering useful features from this data, and finally building and comparing different predictive models.


---

## Understanding the Data

My journey started by looking at some real-world mobile money transaction data. This helped me get a feel for how people transact, what kinds of transactions are common, and how balances typically behave. This initial exploration was crucial for understanding the raw material I'd be working with. I focused on patterns like:

*   **Transaction Amounts:** How much money moves around? (e.g., typical amounts, range)
*   **Transaction Types:** What are the most common activities? (e.g., transfers, debits, payments)
*   **Temporal Patterns:** Are there specific times of day or week when activity is higher?
*   **Balance Behavior:** How do user balances change over time?

This foundational understanding informed how I designed the feature engineering and synthetic data generation processes.

---

## Feature Engineering

To make sense of the raw transaction data, I developed a feature engineering pipeline. This process extracts meaningful information from transaction histories, transforming them into features that predictive models can use. These features capture various aspects of user behavior, such as transaction frequency, amount patterns, and balance dynamics.

The core logic for this is in `src/seqcredit_model/feature_engineering.py`.

### Usage

```python
import sys

sys.path.append('../src')
from seqcredit_model.feature_engineering import TemporalTransactionFeatureEngineer

# Initialize and extract features
engineer = TemporalTransactionFeatureEngineer()
df_features = engineer.extract_all_features(df)

# Create user-level summary
user_summary = engineer.create_user_level_summary(df)
```

## Synthetic Data Generation

Since real-world financial data is sensitive and often scarce, I built a synthetic data generator. This tool creates realistic transaction datasets that mimic real mobile money user behavior, allowing me to develop and test models without privacy concerns.

Initially, the generator focused on general transaction patterns. However, to specifically address credit risk, I had to significantly enhance it. This involved:

*   **Introducing Credit Transactions:** Adding `CREDIT` (loan disbursements) and `LOAN_REPAYMENT` transaction types.
*   **Modeling Borrowing Behavior:** Simulating different user archetypes (e.g., responsible borrowers, risky borrowers, defaulters) to create a diverse dataset for credit risk prediction.
*   **Generating Individual User Files:** Instead of one large file, the generator now creates separate CSV files for each user's transactions, stored in `data/user_transactions/`. This design is better suited for sequential modeling.

The primary script for this is `src/seqcredit_model/synthetic_data.py`.

### Usage

```python
import sys
sys.path.append('src')
from seqcredit_model.synthetic_data import CalibratedMoMoDataGenerator

# Generate individual user datasets
generator = CalibratedMoMoDataGenerator(
    n_users=10000,
    avg_transactions_per_user=15,
    start_date='2024-01-01',
    duration_days=180,
    output_dir='data/user_transactions'
)

summary_df = generator.generate_dataset()
```






## Credit Risk Modeling

With the data prepared and features engineered, the next step was to build models to predict credit default. I explored both traditional static models and more advanced sequential models (LSTMs) to see which performed better and when the complexity of sequential models was justified.

The goal is a binary classification: predicting whether a borrower will default on a loan. Users who haven't taken loans are excluded from this specific prediction task.

### Core Components:

The main modeling logic resides in `src/seqcredit_model/credit_model.py` and is orchestrated through `notebooks/credit_risk_model.ipynb`.

*   **Data Loading and Splitting:** A `CreditRiskDataLoader` handles loading the various datasets, merging user features with summaries, and preparing sequences for the LSTM model. It also ensures a consistent train/test split across all models for fair comparison.
*   **Model Implementations:** I implemented several models:
    *   **Logistic Regression:** A good baseline static model.
    *   **XGBoost:** A powerful gradient boosting model, also static.
    *   **LSTM Model:** A recurrent neural network designed for sequential data, capable of learning patterns over time in transaction histories.
*   **Model Evaluation:** A `ModelEvaluator` helps compare the performance of these different models using metrics like ROC curves, precision-recall curves, and confusion matrices.

### Completed Tasks:

- [x] All dependencies installed and verified
- [x] Notebooks tested and working
- [x] Model results analyzed
- [x] File naming standardized
- [x] Documentation created (AGENTS.md, CLAUDE.md)

---

## March 2026: Data Pipeline Restructuring

### Critical Bug Fix: Stale Labels Issue

**Discovered:** Models were training on `summary_extended.csv` which contained labels from a previous data generation run. This file didn't match the current transaction files:

| File | USER_000000 Transactions | Status |
|------|-------------------------|--------|
| `summary_extended.csv` | 21 | STALE (old generation) |
| `user_transactions/USER_000000.csv` | 114 | Current (actual data) |
| `user_summaries.csv` | 114 | Correct (matches transactions) |

**Impact:** Models may have been trained on mismatched labels, potentially affecting accuracy.

**Resolution:** Used `user_summaries.csv` (renamed to `user_labels.csv`) which is the correct output from synthetic data generation.

### File Renaming

Standardized data file names for clarity:

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `calibration.json` | `synthetic_params.json` | Parameters for synthetic data generation |
| `features.csv` | `user_features.csv` | User-level aggregated features |
| `user_summaries.csv` | `user_labels.csv` | Credit risk labels per user |
| `summary_extended.csv` | → `legacy/` | Stale data, moved out of active pipeline |

### Legacy Directory Created

Moved unused/test files to `data/legacy/`:

- `raw_part1.csv`, `raw_part2.csv` — Test data for single user
- `features_engineered.csv` — Transaction-level features from raw files
- `transactions.csv` — Old fraud detection dataset (different schema)
- `transactions_calibrated.csv` — Calibrated variant of above
- `profiles.csv` — User profiles for simulation
- `summary.csv` — Old aggregate stats
- `summary_extended.csv` — Stale labels (the bug)

All legacy files added to `.gitignore`.

### Config Changes

Updated `src/seqcredit_model/config.py`:

```python
# Before
CALIBRATION_FILE = DATA_DIR / 'calibration.json'
FEATURES_FILE    = DATA_DIR / 'features.csv'
SUMMARIES_FILE   = DATA_DIR / 'summary_extended.csv'  # BUG

# After
SYNTHETIC_PARAMS_FILE = DATA_DIR / 'synthetic_params.json'
USER_FEATURES_FILE     = DATA_DIR / 'user_features.csv'
USER_LABELS_FILE      = DATA_DIR / 'user_labels.csv'  # FIXED
LEGACY_DIR            = DATA_DIR / 'legacy'
```

### Code Updates

Updated all source files to use new config constants:

- `synthetic_data.py`: `CALIBRATION_FILE` → `SYNTHETIC_PARAMS_FILE`, output path → `user_labels.csv`
- `feature_engineering.py`: `FEATURES_FILE` → `USER_FEATURES_FILE`
- `credit_model.py`: `FEATURES_FILE` → `USER_FEATURES_FILE`, `SUMMARIES_FILE` → `USER_LABELS_FILE`

### Documentation Created

- **AGENTS.md**: Guide for AI coding agents (build/lint commands, code style, architecture)
- **CLAUDE.md**: Guide for Claude Code (setup, architecture, key patterns)
- **docs/REPORT.md**: Comprehensive project report
- **docs/SESSION.md**: This development log

### LSTM Cache Cleanup

Deleted `data/lstm_sequences.npz` (was built with stale labels). Needs regeneration after running feature engineering.

### Final Data Structure

```
data/
├── synthetic_params.json     # Calibration parameters
├── user_features.csv         # Aggregated user features (10,000 users)
├── user_labels.csv           # Credit risk labels (correct)
├── model_comparison.csv      # Model evaluation output
├── lstm_sequences.npz        # Cache (regenerate)
├── user_transactions/        # Per-user CSVs (10,000 files)
│   └── USER_XXXXXX.csv
└── legacy/                   # Old/unused files (gitignored)
    ├── summary_extended.csv  # Stale labels
    ├── transactions.csv
    └── ...
```

### Next Steps (after March 2026 pipeline fix)

- [x] Regenerate LSTM sequences: delete `lstm_sequences.npz` and re-run training
- [x] Re-train all models with corrected labels
- [x] Compare performance metrics before/after fix
- [x] Consider adding data validation checks in `CreditRiskDataLoader` to catch mismatches

---

## March 15, 2026: Descriptive Statistics Notebook

### Goal

Deliver a dedicated descriptive statistics notebook covering numerical variability (mean, std, CV) and categorical distributions (unique value counts, frequencies) for the synthetic dataset.

### What Was Done

Restructured `notebooks/data_analysis.ipynb` from an EDA/narrative notebook into a formal descriptive statistics notebook. The previous version was visualization-first (user stories, radar charts, behavioral narrative); the new version leads with statistical tables and uses plots only to support them.

**Removed:**
- Individual user story section (balance trajectory plots for 3 users)
- Archetype radar chart
- "Feature Engineering Preview" framing (correlation-with-default bar chart)
- All EDA narrative markdown (Part 1–6 story structure)

**Added:**

| Section | Content |
|---------|---------|
| 1 — Dataset Overview | Shape, column listings, null counts for features/labels/transactions |
| 2 — Numerical Stats | `describe()` + CV table for all numeric features; top-15 CV ranking chart; violin plots annotated with μ and σ per archetype |
| 3 — Categorical Stats | Frequency tables (count + %) for `credit_archetype`, `credit_risk_label`, `TRANS. TYPE`, `LOAN_PROVIDER`, day of week, hour of day; bar charts for each |
| 4 — Distributions | Amount stats (CV, skew, kurtosis); log-scale histogram; IQR box by type; loan disbursement vs repayment stats |
| 5 — Correlations | Pearson heatmap + printed top-10 correlations with default target |
| 6 — Summary | Mean ± std table; styled gradient mean table by archetype |

### Files Changed

- `notebooks/data_analysis.ipynb` — full rewrite
- `README.md` — updated notebook description
- `docs/REPORT.md` — added Descriptive Statistics Notebook section
- `docs/SESSION.md` — this entry

---

## March 16, 2026: Model Expansion & Hybrid LSTM

### Goal

Expand the model portfolio with additional classifiers and improve the LSTM architecture by combining sequential with static features.

### What Was Done

**1. Added New Model Classes to `src/seqcredit_model/credit_model.py`:**

| Class | Description |
|-------|-------------|
| `RandomForestModel` | sklearn RandomForest with balanced class weights |
| `LightGBMModel` | LightGBM with early stopping support |
| `HybridLSTMModel` | Dual-branch LSTM + static features (early fusion) |

**2. Created `/experiments` Directory:**

```
experiments/
├── lr_model.ipynb      # Logistic Regression
├── xgb_model.ipynb     # XGBoost
├── rf_model.ipynb      # Random Forest
├── lgbm_model.ipynb    # LightGBM
└── lstm_model.ipynb    # Hybrid LSTM (sequence + static)
```

Each notebook follows a consistent structure:
- Setup (imports, paths, seeds)
- Data Preparation (CreditRiskDataLoader)
- Model Training (cross-validation + fit)
- Evaluation (AUC-ROC, AUC-PR, classification report)
- Visualization (ROC curves, PR curves, confusion matrices)

**3. Hybrid LSTM Architecture:**

The new `HybridLSTMModel` combines:
- **Sequential branch**: LSTM layers processing transaction sequences
- **Static branch**: Dense layers processing user-level features
- **Fusion**: Concatenate → Dense → sigmoid output

This addresses the poor performance of pure LSTM (AUC ~0.50) by incorporating the highly predictive static features.

**4. Updated Documentation:**

- `AGENTS.md` — Added experiments section, updated model list
- `CLAUDE.md` — Added experiments, updated architecture
- `README.md` — Updated project structure
- `docs/REPORT.md` — Updated model descriptions, key classes table

### Files Changed

- `src/seqcredit_model/credit_model.py` — Added 3 new model classes (~250 lines)
- `experiments/lr_model.ipynb` — New notebook
- `experiments/xgb_model.ipynb` — New notebook
- `experiments/rf_model.ipynb` — New notebook
- `experiments/lgbm_model.ipynb` — New notebook
- `experiments/lstm_model.ipynb` — New notebook (hybrid)
- `AGENTS.md` — Documentation update
- `CLAUDE.md` — Documentation update
- `README.md` — Documentation update
- `docs/REPORT.md` — Documentation update

### Lint Check

```bash
ruff check src/          # ✅ All checks passed
ruff format --check src/  # ✅ All files formatted
```

### Next Steps

- [ ] Run individual model notebooks and verify they work
- [ ] Compare model performance across all 5 models
- [ ] Evaluate if hybrid LSTM outperforms static models

---

## March 16, 2026: Model Implementation Bug Fixes

### Goal

Scan the model implementation for bugs and fix all issues found across `credit_model.py` and the experiment notebooks.

### Bugs Found & Fixed

**`src/seqcredit_model/credit_model.py`**

| # | Bug | Fix |
|---|-----|-----|
| 1 | `XGBoostModel` passed deprecated `use_label_encoder=False` param — raises a warning in XGBoost ≥1.6 | Removed the key from the params dict |
| 2 | `TF_ENABLE_ONEDNN_OPTS = 0` was set as a Python variable, not an environment variable — had no effect on TensorFlow | Replaced with `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` before all imports |
| 3 | `HybridLSTMModel.fit()` built `validation_data = ([X_val_seq, X_val_static], y_train)` — used `y_train` instead of `y_val`, causing early stopping to monitor the wrong labels | Changed `y_train` → `y_val` in the tuple |
| 4 | `HybridLSTMModel.fit()` referenced `y_val` in the body but `y_val` was not in the method signature — `NameError` at runtime whenever a validation set was passed | Added `y_val=None` parameter to the signature |
| 5 | `HybridLSTMModel` had no `cross_validate()` method — inconsistent with every other model class (`LogisticRegressionModel`, `XGBoostModel`, `RandomForestModel`, `LightGBMModel`, `LSTMModel`) | Added `cross_validate(self, X_seq, X_static, y, n_splits=5, epochs=100, batch_size=32, class_weight=None)` using `StratifiedKFold`, mirroring `LSTMModel.cross_validate` |

**`experiments/lstm_model.ipynb`**

| # | Bug | Fix |
|---|-----|-----|
| 6 | Cell-7 created a second independent `train_test_split` — the LSTM was trained and evaluated on a different user partition than all static models, making comparisons invalid | Replaced `train_test_split(...)` block with `loader._train_user_ids` / `loader._test_user_ids` (set by `prepare_static_splits()`) |

### Files Changed

- `src/seqcredit_model/credit_model.py` — Bugs 1–5
- `experiments/lstm_model.ipynb` — Bug 6 (cell-7)

### Outstanding

- [ ] Add `save()`/`load()` methods to all model classes (low priority, deferred)

