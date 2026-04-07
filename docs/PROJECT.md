# Sequential Credit Risk Modeling: Project Documentation

**Project:** Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments  
**Author:** Attabra Benjamin Ekow  
**Last Updated:** April 2, 2026

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
11. [Open Questions](#open-questions)
12. [Usage](#usage)
13. [Dependencies](#dependencies)
14. [Limitations](#limitations)
15. [Future Work](#future-work)

---

## Executive Summary

This project is a **preliminary investigation and proof-of-concept framework paper** for credit risk modeling in African fintech contexts. It makes two primary contributions:

1. **Open synthetic benchmark** — `CalibratedMoMoDataGenerator` produces a 10,000-user mobile money dataset calibrated to real Ghanaian patterns (MTN MoMo / QwikLoan), providing a privacy-safe, reproducible testbed for African credit risk research where real data is scarce or inaccessible.

2. **Temporal feature engineering framework** — `TemporalTransactionFeatureEngineer` formalizes 8 feature groups (38 features) extracted from transaction sequences, offering a reusable pipeline for practitioners and researchers.

Preliminary evaluation on this benchmark compares six models (Logistic Regression, XGBoost, Random Forest, LightGBM, LSTM, Hybrid LSTM) under rigorous 5-fold cross-validation. The central finding is that **well-engineered static user-level features capture most of the available default signal** (RF: 0.832 AUC-ROC); the Hybrid LSTM is comparable in discrimination (0.813) but significantly worse in calibration (ECE 0.22 vs 0.04); and the standalone LSTM collapses (0.523), confirming that transaction sequences alone carry insufficient signal.

**Scope:** These are preliminary findings on a controlled synthetic benchmark. External validity on real-world mobile money data is the subject of Paper B (pending Telecel Ghana data access). This paper does not claim universal generalizability — it establishes a framework, an open benchmark, and motivating results for the larger study.

---

## Current Results

All benchmark results are stored in the data files below. `RANDOM_SEED = 42` throughout.

| File | Contents |
|------|----------|
| `data/cv_results_y_default.csv` | 5-fold CV — all 6 models, `y_default` target |
| `data/cv_results_y_bad.csv` | 5-fold CV — all 6 models, `y_bad` target |
| `data/significance_tests.csv` | Pairwise bootstrap significance tests |
| `data/ablation_features.csv` | Feature group ablation — drop-one + single-group conditions |
| `data/tuning_results.csv` | Hyperparameter tuning results (RF, XGBoost, LightGBM) |

Generated with:
```bash
python src/seqcredit_model/run_cv_benchmark.py       # primary benchmark
python src/seqcredit_model/run_ablation_study.py     # ablation study
python src/seqcredit_model/run_hyperparameter_tuning.py  # tuning
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
│   ├── synthesize.py             # Synthetic data generation
│   ├── pipeline.py               # Transaction feature extraction
│   └── credit_model.py           # Models + data loader + evaluator
├── notebooks/                    # Jupyter notebooks
│   ├── model.ipynb               # Primary modeling notebook (all 6 models)
│   ├── analysis.ipynb            # Research analysis: SHAP, surrogate tree, calibration
│   ├── data.ipynb                # Descriptive statistics notebook
│   └── *_gcolab.ipynb            # Google Colab variants (3 files)
├── src/
│   └── synthetic_params.json     # Calibration parameters
├── data/                         # Generated data
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
| `LSTMModel` | Padded sequences (max_len=50, 38 features), stacked LSTM (32→16 units), dropout=0.4, L2 reg |
| `HybridLSTMModel` | Dual-branch: LSTM (sequences) + Dense (static) -> Concatenate -> sigmoid |

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `CalibratedMoMoDataGenerator` | `synthesize.py` | Generate realistic transaction data |
| `TemporalTransactionFeatureEngineer` | `pipeline.py` | Extract features from transactions |
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

**Research Question:** *Can a calibrated synthetic mobile money dataset and a temporal feature engineering framework serve as a reusable open benchmark for African fintech credit risk research? And on this controlled benchmark, what does a comparative preliminary evaluation reveal about the relative value of sequential deep learning versus well-engineered static features for default prediction?*

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

`notebooks/analysis.ipynb` implements the full publishable research pipeline on top of trained models. It requires running `model.ipynb` first to generate saved models and LSTM arrays.

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

- Created `analysis.ipynb` with full XAI pipeline
- Added SHAP analysis for XGBoost and LightGBM
- Implemented surrogate decision tree with fidelity analysis
- Added calibration analysis (Brier, ECE, isotonic correction)

### April 2026: Documentation Consolidation

- Updated all documentation to reflect current project state
- Created consolidated `docs/PROJECT.md` (this file)
- Added Google Colab notebook variants
- Ensured data consistency (100% match between features and labels)

### April 2026: CV Benchmark + Bug Fixes

- Implemented `run_cv_benchmark.py`: 5-fold stratified CV for all 6 models, both targets, with bootstrap significance tests and OOF predictions
- **Bug fix — double-scaling:** `run_cv_benchmark.py` was passing pre-scaled `X_train_scaled` from the data loader to CV functions that scaled again per-fold. Fixed to pass raw `X_train` so each fold's scaler is the only scaler applied.
- **Bug fix — LSTM overfitting:** Standalone LSTM (val AUC ≈0.53) was severely overfitting due to excess capacity (64→32 units) relative to available signal. Reduced to 32→16 units, increased dropout to 0.4, added L2 kernel regularization (1e-4) and higher recurrent_dropout (0.3). Awaiting re-run.
- **Investigation resolved:** The single-split results (static ~0.53 AUC) that contradicted CV results (static ~0.83 AUC) were traced to a class-imbalance handling bug in earlier runs. CV results are authoritative.

---

## Publication Plan

**Goal:** Complete the project to publication-ready quality with reproducible experiments, robust validation, and finalized manuscript artifacts.

**Targets:**
- **Deep Learning Indaba 2026** — abstract deadline April 15, 2026 (Lagos, Nigeria, August 2–7)
- **IEEE ICAST 2026** — abstract deadline ~April 30, 2026 (AI track or Digital Innovation track)

**Framing:** Preliminary Investigation / Novel Framework + Open Benchmark. This paper does not claim to present production-ready or universally generalizable results. It contributes a reusable framework and synthetic benchmark, and provides preliminary motivating findings that justify a larger real-data study (Paper B).

**Core Contributions:**
1. Calibrated synthetic Ghanaian mobile money dataset (open benchmark, 10k users, 5 archetypes)
2. Temporal feature engineering framework (8 groups, 38 features) — reusable pipeline for African fintech
3. Preliminary comparative evaluation: static features dominate sequences on this benchmark; LSTM alone is insufficient; Hybrid LSTM is comparable in AUC but worse in calibration
4. Ablation evidence: `behavioural_diversity` and `loan_history` carry most default signal (drop-one: −0.135 and −0.045 AUC respectively)

### Success Criteria

- [ ] 5-fold stratified CV results completed for all key models and both targets (`y_default`, `y_bad`)
- [ ] Final benchmark tables include 95% CIs and significance comparisons
- [ ] Medium-priority experiments (tuning, ablations, Transformer baseline) completed and documented
- [ ] Paper draft completed end-to-end with final figures/tables
- [ ] Metrics synchronized across `data/model_comparison.csv`, `docs/PROJECT.md`, `docs/DATACARD.md`, and `README.md`
- [ ] Reproducibility verified with `RANDOM_SEED = 42` and a scripted run path

### Phase 1 - Validation Core (Publication-blocking)

**Timeline:** Week 1  
**Status:** Completed (with unexpected findings requiring investigation)

#### 1.1 Implement 5-Fold Stratified CV

- [x] Add CV pipeline for `LogisticRegressionModel`, `XGBoostModel`, `RandomForestModel`, `LightGBMModel`, `LSTMModel`, and `HybridLSTMModel`
- [x] Run CV on both targets: `y_default` and `y_bad`
- [x] Track fold-level metrics: ROC-AUC, PR-AUC, F1, Precision, Recall, Brier, ECE
- [x] Persist fold results and aggregate means/std/CIs
- [x] Output `data/cv_results_y_default.csv`
- [x] Output `data/cv_results_y_bad.csv`

**CV Results:** See `data/cv_results_y_default.csv`.

#### 1.2 Statistical Comparison Layer

- [x] Add pairwise tests for core claims (best static vs LSTM, best static vs HybridLSTM)
- [x] Use bootstrap/permutation delta testing for ROC-AUC and PR-AUC
- [x] Report effect sizes with p-values and/or confidence intervals
- [x] Output `data/significance_tests.csv`

**Note:** No statistically significant differences found between top models (RandomForest, LogisticRegression, HybridLSTM) in CV results.

#### 1.3 Reproducibility Guardrails

- [x] Standardize seed handling (`RANDOM_SEED = 42`) across scripts/notebooks
- [x] Ensure deterministic split logic and cache behavior are documented
- [x] Add a single run path to regenerate final benchmark artifacts
- [x] Output `src/seqcredit_model/run_cv_benchmark.py`
- [x] Output `data/cv_manifest.json` (reproducibility params + timings)

#### 1.4 Investigation (Resolved)

- [x] Investigate discrepancy between CV and single-split results
- [x] Determine root cause: class-imbalance handling bug in pre-fix runs caused static models to predict all-negative → artificially low AUC
- [x] Fix double-scaling bug in `run_cv_benchmark.py`
- [x] Re-evaluate thesis — revised to: static models competitive with Hybrid LSTM; sequences alone insufficient
- [x] Update documentation and paper claims

### Phase 2 - Strengthening Experiments

**Timeline:** Week 2
**Status:** Unblocked — Phase 1.4 resolved

#### 2.1 Hyperparameter Tuning

- [x] Script: `src/seqcredit_model/run_hyperparameter_tuning.py` (40 trials RandomizedSearchCV, 3-fold search → 5-fold final, RF/XGBoost/LightGBM)
- [x] Output `data/tuning_results.csv`

**Tuning Results (y_default):** XGBoost tuned is marginally best (0.8303 AUC-ROC vs default 0.8183). Tuned RF (0.8271) performs slightly *worse* than default RF (0.8319) — likely due to reduced depth (max_depth=5 vs default 10). LightGBM tuned (0.8227) also below default (0.8123→0.8227). HybridLSTM excluded from tuning (compute cost). Conclusion: default hyperparameters are near-optimal for this dataset; gains from tuning are marginal.

#### 2.2 Ablation Studies

- [x] Drop-one-group ablations: 8 feature groups, RandomForest, 5-fold CV each
- [x] Single-group-only ablations: standalone signal of each group
- [x] Script: `src/seqcredit_model/run_ablation_study.py`
- [x] Output `data/ablation_features.csv`

**Key ablation findings (y_default, RandomForest):**
- `behavioural_diversity` (unique_recipients, recipient_concentration, unique_txn_types) is the dominant group: dropping it cuts AUC-ROC from 0.832 → 0.697 (−0.135). Alone it achieves AUC 0.777.
- `loan_history` is second most important: dropping it costs −0.045 AUC-ROC.
- All other groups (amount_stats, balance_dynamics, fee_behaviour, temporal_patterns, txn_type_mix, activity_intensity) have negligible individual impact (delta ≤ 0.005); some marginally improve performance when dropped.
- Implication: behavioral diversity and loan history encode most of the default signal; the other 6 feature groups add robustness but not discriminative power.

#### 2.3 Transformer/Attention Baseline (CRITICAL)
- [ ] Implement a lightweight Transformer/attention baseline (e.g., 2-layer encoder + global pooling)
- [ ] Train/evaluate under the same 5-fold CV stack as LSTM/static models
- [ ] **Goal:** Determine if the "sequence collapse" (0.52 AUC) is an LSTM limitation or a data property.

#### 2.4 Formalized Reproducibility Script
- [ ] Create `run_full_experiment_suite.py` that executes: synthesize → pipeline → cv_benchmark → ablation → tuning → transformer.
- [ ] Ensure it generates a final `benchmark_summary_report.md` for easy copy-pasting into the paper draft.

### Phase 3 - Paper and Artifact Finalization

**Timeline:** Week 3  
**Status:** Pending

#### 3.1 Manuscript Draft Completion
- [ ] Write full draft: Introduction, Related Work, Data, Methods, Results, Interpretability, Calibration, Ablations, Discussion, Conclusion
- [ ] **Revise core claims based on CV findings:** Static features dominate MoMo sequences for default prediction.
- [ ] Output `paper/draft_v1.md` (or LaTeX equivalent)

#### 3.2 Figure/Table Freeze
- [ ] Freeze benchmark tables with CIs and significance results (use CV results as primary)
- [ ] Freeze SHAP/permutation/surrogate/calibration/ablation/Transformer figures
- [ ] Output `paper/figures/*` and `paper/tables/*`
- [ ] Add reproducibility manifest (`paper/repro_manifest.md`)

#### 3.3 Documentation Sync

- [ ] Update `docs/PROJECT.md`, `docs/DATACARD.md`, and `README.md` after final rerun
- [ ] Confirm all reported metrics match `data/cv_results_y_default.csv` (CV as primary)
- [ ] Reconcile single-split vs CV results in documentation

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

#### Week 1 (Completed)

- [x] CV complete for all models and both targets
- [x] Significance tests complete (no significant differences found between top models)
- [x] Reproducibility run path validated
- [x] CV vs single-split discrepancy investigated and resolved
- [x] Double-scaling bug fixed
- [x] LSTM regularization applied (re-run pending)

#### Week 2

- [x] Re-run CV benchmark after LSTM fix — LSTM still 0.5231 AUC-ROC on y_default even after regularization (32→16 units, dropout 0.4, L2, recurrent_dropout 0.3). Confirms: sequences alone carry insufficient default-predictive signal. This is the publishable finding.
- [x] Ablation study complete (`data/ablation_features.csv`)
- [x] Hyperparameter tuning complete (`data/tuning_results.csv`)
- [ ] Transformer baseline

#### Week 3

- [ ] Manuscript draft complete (with revised claims if needed)
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

## Open Questions

### 1. ~~Why do static models perform much better in CV vs single-split?~~ (Resolved)

**Root cause:** Class-imbalance handling bug in early runs caused static models to predict all-negative, yielding artificially low AUC (~0.53). The "post-imbalance fix" commit corrected `class_weight='balanced'` and `scale_pos_weight` settings. CV results (static ~0.83) are the true performance.

### 2. Is the "sequential models beat static models" thesis still valid? (Resolved — revised)

**Conclusion:** No. With proper imbalance handling and CV evaluation:
- Top static models (RF 0.832, LR 0.831) match HybridLSTM (0.813) in AUC-ROC
- Static models are significantly better calibrated (ECE 0.04 vs 0.22)
- No statistically significant pairwise differences between any top-5 models
- **Revised framing:** The Hybrid provides comparable discriminative performance to static models, but static tree models should be preferred when calibration matters (e.g., loan pricing, risk scoring)

### 3. ~~Single-split notebook consistency~~ (Superseded)

Single-split results from the notebook are no longer used as a primary benchmark. CV results from `run_cv_benchmark.py` are authoritative.

### 4. Why does standalone LSTM collapse in CV? (Resolved)

**Root cause identified:** Model too large (64→32 units) relative to available positive examples (~529 defaults in training). Train AUC reaches 0.95 while val AUC stagnates at 0.50–0.57 — classic overfitting.

**Fix applied:** Reduced to 32→16 units, dropout 0.4, L2 kernel regularization (1e-4), recurrent_dropout 0.3. Re-run completed.

**Outcome:** LSTM still scores 0.5231 AUC-ROC on y_default after regularization (y_bad: 0.5071) — essentially random. This confirms the finding is not a capacity/overfitting artifact but a data property: transaction sequences alone lack sufficient signal to predict default.

**Conclusion (publishable):** *Well-engineered static user-level features capture the relevant predictive signal; the temporal order of individual transactions adds negligible marginal value for default prediction. The Hybrid model's 0.813 AUC is entirely attributable to its static feature branch.*

---

## Usage

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
pip install -e .              # editable install — required for all commands below
```

> **Important:** The `pip install -e .` step registers `seqcredit_model` as an importable package. Without it, `python -m` invocations and notebook imports will fail with `ModuleNotFoundError`. Source edits take effect immediately (no reinstall needed).

### Full Rebuild (from scratch)

```bash
python -m seqcredit_model.synthesize       # 1. Generate synthetic data (~few min)
python -m seqcredit_model.pipeline         # 2. Build user-level features
jupyter notebook notebooks/model.ipynb     # 3. Train all 6 models
jupyter notebook notebooks/analysis.ipynb  # 4. Research analysis (SHAP, calibration)
jupyter notebook notebooks/data.ipynb      # 5. Descriptive statistics
```

### Benchmark Scripts

```bash
python src/seqcredit_model/run_cv_benchmark.py          # 5-fold CV (~75 min)
python src/seqcredit_model/run_ablation_study.py        # Feature group ablation (~30 min)
python src/seqcredit_model/run_hyperparameter_tuning.py # Tuning: RF, XGB, LGBM (~45 min)
```

### Google Colab

The `*_gcolab.ipynb` notebooks are self-contained — they install dependencies and generate data automatically. No local setup needed.

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

Install: `pip install -r requirements.txt && pip install -e .`

---

## Limitations

This work is an explicitly scoped **preliminary investigation**. Reviewers and readers should note:

1. **Synthetic data only.** All results are on a calibrated synthetic dataset. While the generator is calibrated to real Ghanaian MoMo patterns, it cannot capture the full complexity of real transaction behavior. Findings may not transfer directly to live credit portfolios.

2. **Single geography and product type.** The synthetic parameters are tuned for Ghanaian mobile money (MTN QwikLoan structure). Generalizability to other markets or products is untested.

3. **No prospective or deployment validation.** Models are evaluated on a held-out test split within the synthetic dataset. No live A/B test or shadow deployment has been conducted.

4. **Binary default framing.** The primary target (`y_default`) collapses late payment and default into non-default. Real credit risk applications typically require probability calibration and decision thresholds tuned to deployment costs — partially addressed by calibration analysis but not fully resolved.

5. **Sequences may carry more signal in real data.** The finding that standalone LSTM collapses (0.523 AUC) reflects a property of the synthetic dataset's generation process, where credit outcomes are partly determined by user archetype labels rather than purely by transaction sequence dynamics. This motivates — but does not substitute for — real-data validation.

**Paper B** (Hybrid LSTM on real Telecel Ghana data, pending data access) addresses limitations 1–3 directly and is the intended real-world validation of this framework.

---

## Future Work

- [ ] **Transformer/Attention Baseline:** Evaluate if attention mechanisms can extract more signal from sequences than LSTM (Critical).
- [ ] **One-Command Reproducibility:** Create `run_full_experiment_suite.py` for "push-button" result generation (High).
- [ ] **Unit Tests:** Add integrity checks for data processing and label alignment (Moderate).
- [ ] **Paper B:** Validate framework on real Telecel Ghana mobile money data (Separate Project).

---

## References

- Mobile money transaction patterns calibrated to Ghanaian data (MTN QwikLoan)
- Loan parameters: GHS 25-1,000, 6.9% interest, 30-day term
