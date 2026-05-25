# seqcredit-model Project Reference

**Project:** Sequential Credit Risk Modeling in Data-Constrained Environments  
**Branch:** `real-data`  
**Package:** `seqcredit-model`  
**Python:** `>=3.10`

## Summary

This repository contains two connected code paths:

1. a local synthetic mobile-money credit-risk benchmark
2. a Spark/Databricks pipeline for real Telecel Ghana analysis

The current branch contains both. The most important documentation rule for this repo is simple: describe the current codebase as it exists, and distinguish that from older checked-in artifacts when they do not match the current script lineup exactly.

## Repository Surface

### Core modules

- `src/seqcredit_model/config.py`
  Central path constants and runtime override helpers.
- `src/seqcredit_model/synthesize.py`
  Synthetic transaction generator.
- `src/seqcredit_model/pipeline.py`
  Transaction feature engineering and synthetic user-level feature table build path.
- `src/seqcredit_model/credit_model.py`
  Data loader, metrics, evaluator, and model implementations.
- `src/seqcredit_model/real_data_pipeline.py`
  Spark-based feature/label generation for the real-data workflow.

### Benchmark and utility scripts

- `src/seqcredit_model/run_cv_benchmark.py`
- `src/seqcredit_model/run_ablation_study.py`
- `src/seqcredit_model/run_hyperparameter_tuning.py`
- `src/seqcredit_model/run_full_benchmark.py`
- `src/seqcredit_model/compute_bootstrap_ci.py`
- `src/seqcredit_model/reset.py`

## Workflows

### Local synthetic workflow

```bash
python -m seqcredit_model.synthesize
python -m seqcredit_model.pipeline
jupyter notebook notebooks/model.ipynb
```

Supporting commands:

```bash
python -m seqcredit_model.run_cv_benchmark
python -m seqcredit_model.run_ablation_study
python -m seqcredit_model.run_hyperparameter_tuning
python -m seqcredit_model.run_full_benchmark
python -m seqcredit_model.compute_bootstrap_ci
python -m seqcredit_model.reset --data --models --yes
```

### Databricks real-data workflow

`real_data_pipeline.py` exposes:

```python
from seqcredit_model.real_data_pipeline import build_pipeline
```

Its public entrypoint is:

```python
build_pipeline(df, output_dir=None, min_followup_days=60)
```

The pipeline is designed to:

- parse raw transaction timestamps in Spark
- identify each borrower’s most recent loan disbursement as the index loan
- build labels strictly after the index loan
- build features strictly before the index loan
- write runtime-compatible `user_features.csv`, `user_labels.csv`, and sequence artifacts

This is the branch’s main leakage-control design.

## Models In Code

### Exported package models

- `LogisticRegressionModel`
- `XGBoostModel`
- `RandomForestModel`
- `LightGBMModel`
- `LSTMModel`
- `GRUModel`
- `HybridLSTMModel`
- `HybridGRUModel`

### Additional models implemented in `credit_model.py`

- `TransformerModel`
- `HybridTransformerModel`

### Important current-state distinction

The checked-in benchmark artifacts in `data/` now match the current `run_cv_benchmark.py` configuration: `LSTMModel` and `GRUModel` as sequential models, four static baselines (LogisticRegression, XGBoost, RandomForest, LightGBM), 5-fold CV, 1000-bootstrap significance tests. `HybridLSTM` no longer appears in either the live script or the checked-in manifest.

## Data and Runtime Paths

`config.py` defines repo-local defaults and runtime override helpers.

### Default paths

- `DATA_DIR`
- `TRANSACTIONS_DIR`
- `USER_FEATURES_FILE`
- `USER_LABELS_FILE`
- `LSTM_CACHE_FILE`
- `RAW_SEQ_FILE`
- `MODELS_DIR`

### Runtime override helpers

- `get_runtime_lstm_cache_file()`
- `get_runtime_raw_seq_file()`
- `get_runtime_user_features_file()`
- `get_runtime_user_labels_file()`

Related environment variables:

- `SEQCREDIT_DATA_DIR`
- `SEQCREDIT_LSTM_CACHE_FILE`
- `SEQCREDIT_RAW_SEQ_FILE`
- `SEQCREDIT_USER_FEATURES_FILE`
- `SEQCREDIT_USER_LABELS_FILE`
- `SEQCREDIT_EPHEMERAL`

These matter for notebook and Databricks runs that write data outside the repo’s default `data/` directory.

## Synthetic Pipeline Facts

### Synthetic outputs

- `data/user_transactions/USER_XXXXXX.csv`
- `data/user_labels.csv`
- `data/user_features.csv`

### Label conventions

- `-1`: non-borrower
- `0`: good
- `1`: late / risky
- `2`: default

### Synthetic target definitions

- `y_default`: default only, `credit_risk_label == 2`
- `y_bad`: risky or default, `credit_risk_label in {1, 2}`

### Checked-in synthetic table shapes

- `data/user_features.csv`: 30 columns total, including `user_id`
- `data/user_labels.csv`: 6 columns

`user_features.csv` contains the 29 aggregate synthetic user features plus `user_id`. `CreditRiskDataLoader` adds loan-history features from raw transaction CSVs at load time, which is why the feature count used in model training is higher than the table width alone suggests.

## Current Checked-In Results

These are the authoritative tracked result artifacts currently present in `data/`.

### CV results

Files:

- `data/cv_results_y_default.csv`
- `data/cv_results_y_bad.csv`
- `data/significance_tests.csv`
- `data/cv_manifest.json`

Mean metrics from the checked-in CV CSVs:

#### `y_default`

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GRU | 0.7565 | 0.1160 | 0.1452 | 0.2130 | 0.3671 |
| LSTM | 0.7478 | 0.1070 | 0.1409 | 0.2151 | 0.3668 |
| XGBoost | 0.7303 | 0.1075 | 0.1566 | 0.1612 | 0.2966 |
| LightGBM | 0.7292 | 0.1032 | 0.1566 | 0.1621 | 0.2993 |
| RandomForest | 0.7275 | 0.1018 | 0.1599 | 0.1466 | 0.2947 |
| LogisticRegression | 0.7089 | 0.0920 | 0.1322 | 0.2194 | 0.3982 |

#### `y_bad`

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GRU | 0.7159 | 0.8145 | 0.7069 | 0.2166 | 0.1406 |
| LSTM | 0.7112 | 0.8109 | 0.7104 | 0.2170 | 0.1381 |
| XGBoost | 0.7056 | 0.8116 | 0.7000 | 0.2171 | 0.1300 |
| LightGBM | 0.7051 | 0.8115 | 0.7012 | 0.2171 | 0.1301 |
| RandomForest | 0.6965 | 0.8063 | 0.6942 | 0.2186 | 0.1299 |
| LogisticRegression | 0.6855 | 0.7967 | 0.6824 | 0.2250 | 0.1415 |

### Ablation

File:

- `data/ablation_features.csv`

Key checked-in findings:

- baseline `ALL_FEATURES`: AUC-ROC `0.8836`
- `DROP_behavioural_diversity`: AUC-ROC `0.7941`
- `DROP_loan_history`: AUC-ROC `0.7796`
- `ONLY_loan_history`: AUC-ROC `0.7890`
- `ONLY_behavioural_diversity`: AUC-ROC `0.7358`

### Tuning

File:

- `data/tuning_results.csv`

Checked-in tuned static-model results:

| Model | Final AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| RandomForest | 0.8855 | 0.7174 | 0.6685 | 0.0684 | 0.1006 |
| XGBoost | 0.8870 | 0.7266 | 0.5918 | 0.1128 | 0.2197 |
| LightGBM | 0.8880 | 0.7286 | 0.5919 | 0.0912 | 0.1436 |

## Known Inconsistencies To Keep In Mind

- `README.md` was previously broken and is now restored as a real entrypoint.
- `run_full_benchmark.py` and `reset.py` exist and should be documented.
- `credit_model.py` contains GRU and transformer families that older docs omitted.
- `cv_manifest.json` now lists `GRU` and matches the current live benchmark lineup.
- Some script docstrings still refer to older output names such as `cv_results_bad.csv`; documentation in this repo should use the actual tracked filenames in `data/`.

## Limits

- There is no formal automated test suite in the repo yet.
- Real-data results are not represented by a clean, authoritative tracked CSV set in this repo in the same way the synthetic benchmark is.
- Notebook outputs and paper drafts should not be treated as authoritative unless they match tracked code and tracked result artifacts.
