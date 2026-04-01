# Sequential Credit Risk Modeling: Project Documentation

**Project:** Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments  
**Author:** Attabra Benjamin Ekow  
**Last Updated:** April 2026

This document consolidates all project documentation: overview, research framework, development history, and publication plan.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Results](#current-results)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Models](#models)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Research Framework](#research-framework)
8. [Research Analysis Notebook](#research-analysis-notebook)
9. [Development History](#development-history)
10. [Publication Plan](#publication-plan)
11. [Usage](#usage)
12. [Dependencies](#dependencies)
13. [Future Work](#future-work)

---

## Executive Summary

This project develops a credit risk prediction system using mobile money transaction data. It implements both traditional static models (Logistic Regression, XGBoost, Random Forest, LightGBM) and sequential deep learning models (LSTM, Hybrid LSTM) to predict loan default risk. A key feature is the synthetic data generator calibrated to real Ghanaian mobile money patterns, enabling model development without privacy concerns.

**Key Finding:** Hybrid LSTM achieves 53% higher ROC-AUC than the best static model (0.81 vs 0.53), demonstrating that transaction sequences contain predictive signal lost in aggregation.

---

## Current Results

Model performance metrics are stored in `data/model_comparison.csv` and include 95% bootstrap confidence intervals. Results are generated with `RANDOM_SEED = 42`.

**Key Finding:** Hybrid LSTM achieves 53% higher ROC-AUC than the best static model (0.81 vs 0.53), demonstrating that transaction sequences contain predictive signal lost in aggregation.

To regenerate results with bootstrap CIs:
```bash
python src/seqcredit_model/compute_bootstrap_ci.py
```

### Dataset Statistics

- **Users:** 10,000
- **Total transactions:** 1,030,198 (~103 per user average)
- **Borrowers:** 5,952 (60%)
- **Default rate:** 11.1% among borrowers (661 defaults)

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
│   ├── credit_risk_model.ipynb   # Primary modeling notebook (all 6 models)
│   ├── credit_risk_analysis.ipynb# Research analysis: SHAP, surrogate tree, calibration
│   ├── data_analysis.ipynb       # Descriptive statistics notebook
│   └── *_gcolab.ipynb            # Google Colab variants (3 files)
├── data/                         # Generated data
│   ├── synthetic_params.json     # Calibration parameters
│   ├── user_features.csv         # Aggregated user features
│   ├── user_labels.csv           # Credit risk labels
│   ├── model_comparison.csv      # Latest model results
│   └── user_transactions/        # Per-user CSVs (10,000 users, regenerable)
├── models/                       # Persisted trained models
├── docs/
│   ├── PROJECT.md                # This file (consolidated documentation)
│   └── DATACARD.md               # Data documentation
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
- **Secondary target (`y_bad`):** `1` if late or default (label in {1,2}), else `0`

---

## Models

### Model Interface

All models share a common interface:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
results = model.cross_validate(X, y, n_splits=5)
model.save("models/model_name")
model = ModelClass.load("models/model_name")
```

### Static Models

| Model | Description |
|-------|-------------|
| `LogisticRegressionModel` | Scaled user-level features (29), `class_weight='balanced'` |
| `XGBoostModel` | Static features, `scale_pos_weight` for imbalance |
| `RandomForestModel` | 200 estimators, max_depth=10, balanced weights |
| `LightGBMModel` | Early stopping, num_leaves=31 |

### Sequential Models

| Model | Description |
|-------|-------------|
| `LSTMModel` | Padded sequences (max_len=50, 38 features), stacked LSTM layers |
| `HybridLSTMModel` | Dual-branch: LSTM (sequences) + Dense (static) -> Concatenate -> sigmoid |

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `CalibratedMoMoDataGenerator` | `synthetic_data.py` | Generate realistic transaction data |
| `TemporalTransactionFeatureEngineer` | `feature_engineering.py` | Extract features from transactions |
| `CreditRiskDataLoader` | `credit_model.py` | Load/merge/split data for models |
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
- ECE — Expected Calibration Error (15-bin weighted mean |confidence - accuracy|)

---

## Research Framework

**Research Question:** *In a data-constrained, fully synthetic mobile money setting, can sequential credit risk models outperform strong static baselines **and** produce outputs that are interpretable, auditable, and reliable enough for decision-making?*

### The Five-Part Pipeline

1. **Benchmark Performance (the "What")** - Train and compare 6 models on `y_default` and `y_bad`
2. **Feature Contributions / XAI (the "Why")** - Permutation importance, SHAP analysis
3. **Surrogate Decision Tree (Auditable Rules)** - Interpretable approximation of best model
4. **Model Calibration (the "Trust")** - Reliability diagrams, Brier score, ECE
5. **Causal Boundary (the "Action")** - Clear statement that XAI explains model behavior, not causation

### Completed Deliverables

- [x] Results tables for both tasks (`y_default`, `y_bad`) across all models
- [x] SHAP summary + global importance table (default vs bad outcome)
- [x] Surrogate tree rule set + fidelity vs depth plot
- [x] Calibration curves + Brier score (before/after calibration)
- [x] Clear limitations: synthetic-only + predictive-not-causal

---

## Research Analysis Notebook

`notebooks/credit_risk_analysis.ipynb` implements the full publishable research pipeline on top of trained models. It requires running `credit_risk_model.ipynb` first to generate saved models and LSTM arrays.

### Sections

| Section | Content |
|---------|---------|
| 1 - Setup & Loading | Deterministic splits, loads all 6 models + LSTM arrays |
| 2 - Secondary Target: `y_bad` | All 6 models evaluated, side-by-side AUC bar chart |
| 3 - Permutation Importance | Top-15 features for all 4 static models with error bars |
| 4 - SHAP Analysis | TreeExplainer on XGBoost/LightGBM, beeswarm + waterfall plots |
| 5 - Surrogate Decision Tree | Fidelity curve depths 2-8, depth-3 tree plot + rules |
| 6 - Individual Tree Visualisations | RF, XGBoost, LightGBM actual tree splits |
| 7 - Model Calibration | Brier/ECE, reliability diagrams, isotonic calibration |

---

## Development History

### March 2026: Critical Bug Fix

**Issue:** Models were training on `summary_extended.csv` which contained stale labels from a previous data generation run.

**Resolution:** 
1. Switched to `user_labels.csv` (correct output from synthetic data generator)
2. Deleted stale `summary_extended.csv`
3. Updated `config.py` with new constants
4. Regenerated all data with `RANDOM_SEED = 42`

### March 2026: Model Expansion

- Added `RandomForestModel`, `LightGBMModel`, `HybridLSTMModel`
- Fixed LSTM data alignment bug (now uses same train/test split as static models)
- Added `save()`/`load()` methods to all model classes
- Created unified comparison notebook with all 6 models

### March 2026: Research Analysis

- Created `credit_risk_analysis.ipynb` with full XAI pipeline
- Added SHAP analysis for XGBoost and LightGBM
- Implemented surrogate decision tree with fidelity analysis
- Added calibration analysis (Brier, ECE, isotonic correction)

### April 2026: Documentation Consolidation

- Updated all documentation to reflect current project state
- Created consolidated `docs/MAIN.md` (this file)
- Added Google Colab notebook variants
- Ensured data consistency (100% match between features and labels)

---

## Publication Plan

**Goal:** Complete the project to publication-ready quality with reproducible experiments, robust validation, and finalized manuscript artifacts.

**Target:** ML Conference (NeurIPS/ICML/ICLR/AISTATS)  
**Core Contribution:** Sequential models (LSTM/Hybrid LSTM) outperform static models for credit risk prediction from mobile money transaction sequences.

### Success Criteria

- [ ] 5-fold stratified CV results completed for all key models and both targets (`y_default`, `y_bad`)
- [ ] Final benchmark tables include 95% CIs and significance comparisons
- [ ] Medium-priority experiments (tuning, ablations, Transformer baseline) completed and documented
- [ ] Paper draft completed end-to-end with final figures/tables
- [ ] Metrics synchronized across `data/model_comparison.csv`, `docs/PROJECT.md`, `docs/DATACARD.md`, and `README.md`
- [ ] Reproducibility verified with `RANDOM_SEED = 42` and a scripted run path

### Phase 1 - Validation Core (Publication-blocking)

**Timeline:** Week 1

#### 1.1 Implement 5-Fold Stratified CV

- [ ] Add CV pipeline for `LogisticRegressionModel`, `XGBoostModel`, `RandomForestModel`, `LightGBMModel`, `LSTMModel`, and `HybridLSTMModel`
- [ ] Run CV on both targets: `y_default` and `y_bad`
- [ ] Track fold-level metrics: ROC-AUC, PR-AUC, F1, Precision, Recall, Brier, ECE
- [ ] Persist fold results and aggregate means/std/CIs
- [ ] Output `data/cv_results_default.csv`
- [ ] Output `data/cv_results_bad.csv`

#### 1.2 Statistical Comparison Layer

- [ ] Add pairwise tests for core claims (best static vs LSTM, best static vs HybridLSTM)
- [ ] Use bootstrap/permutation delta testing for ROC-AUC and PR-AUC
- [ ] Report effect sizes with p-values and/or confidence intervals
- [ ] Output `data/significance_tests.csv`

#### 1.3 Reproducibility Guardrails

- [ ] Standardize seed handling (`RANDOM_SEED = 42`) across scripts/notebooks
- [ ] Ensure deterministic split logic and cache behavior are documented
- [ ] Add a single run path to regenerate final benchmark artifacts
- [ ] Output `scripts/run_full_benchmark.py` (or equivalent documented notebook flow)

### Phase 2 - Strengthening Experiments

**Timeline:** Week 2

#### 2.1 Hyperparameter Tuning

- [ ] Run bounded-budget tuning (Optuna/GridSearch) for static models
- [ ] Run constrained tuning for sequence models (units, dropout, learning rate, sequence length)
- [ ] Keep validation protocol consistent with Phase 1
- [ ] Output `data/tuning_results.csv`

#### 2.2 Ablation Studies

- [ ] Run feature ablations: loan-only, temporal-only, behavioral-only, full feature set
- [ ] Run sequence length sensitivity analysis (e.g., max_len in {20, 50, 100})
- [ ] Output `data/ablation_features.csv`
- [ ] Output `data/ablation_sequence_length.csv`

#### 2.3 Transformer Baseline

- [ ] Implement a lightweight Transformer/attention baseline
- [ ] Train/evaluate under the same splits and metrics stack
- [ ] Compare against LSTM/Hybrid and best static model
- [ ] Output `data/transformer_results.csv`

### Phase 3 - Paper and Artifact Finalization

**Timeline:** Week 3

#### 3.1 Manuscript Draft Completion

- [ ] Write full draft: Introduction, Related Work, Data, Methods, Results, Interpretability, Calibration, Ablations, Discussion, Conclusion
- [ ] Include predictive-not-causal boundary statement
- [ ] Include synthetic-data and external validity limitations
- [ ] Output `paper/draft_v1.md` (or LaTeX equivalent)

#### 3.2 Figure/Table Freeze

- [ ] Freeze benchmark tables with CIs and significance results
- [ ] Freeze SHAP/permutation/surrogate/calibration/ablation/Transformer figures
- [ ] Output `paper/figures/*` and `paper/tables/*`
- [ ] Add reproducibility manifest (`paper/repro_manifest.md`)

#### 3.3 Documentation Sync

- [ ] Update `docs/PROJECT.md`, `docs/DATACARD.md`, and `README.md` after final rerun
- [ ] Confirm all reported metrics match `data/model_comparison.csv`

### Phase 4 - Engineering Quality (Parallel, Recommended)

**Timeline:** Ongoing

#### 4.1 Minimal Test Suite

- [ ] Add tests for data integrity and label alignment
- [ ] Add tests for split reproducibility
- [ ] Add tests for model interface consistency (`fit`, `predict`, `predict_proba`)
- [ ] Add metric computation sanity checks

#### 4.2 Execution Hygiene

- [ ] Add a one-command experiment runner with clear CLI docs
- [ ] Ensure core experiments do not depend on hidden notebook-only state

### Week-by-Week Snapshot

#### Week 1

- [ ] CV complete for all models and both targets
- [ ] Significance tests complete
- [ ] Reproducibility run path validated

#### Week 2

- [ ] Hyperparameter tuning complete
- [ ] Ablation studies complete
- [ ] Transformer baseline complete

#### Week 3

- [ ] Manuscript draft complete
- [ ] Final figures/tables frozen
- [ ] Documentation synchronized and release-ready

### Risks and Mitigations

- [ ] Mitigate compute overrun by constraining tuning budgets and prioritizing HybridLSTM + best static baselines
- [ ] Mitigate result instability with fixed seeds, deterministic splits, and CI reporting
- [ ] Mitigate scope creep by separating submission-critical tasks from optional engineering improvements

### Definition of Done

- [ ] All publication-blocking tasks in this plan are completed
- [ ] Final benchmark claims are supported by CV + CIs + significance checks
- [ ] Paper draft is complete with reproducible figures/tables
- [ ] Documentation and result artifacts are fully consistent across the repository

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

### Run Analysis

```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
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
- shap

Install: `pip install -r requirements.txt`

---

## Future Work

- [ ] Add hyperparameter tuning (GridSearch/RandomSearch/Optuna)
- [ ] Implement cross-validation with proper time-series splits
- [ ] Experiment with attention mechanisms (Transformer)
- [ ] Add unit tests
- [ ] Validate on real mobile money data (with appropriate permissions)

---

## References

- Mobile money transaction patterns calibrated to Ghanaian data (MTN QwikLoan)
- Loan parameters: GHS 25-1,000, 6.9% interest, 30-day term
