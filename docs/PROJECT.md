# seqcredit-model Project Reference

**Project:** Sequential Credit Risk Modeling in Data-Constrained Environments  
**Branch:** `real-data`  
**Package:** `seqcredit-model`  
**Python:** `>=3.10`

## Summary

This repository contains two connected code paths:

1. a local synthetic mobile-money credit-risk benchmark
2. a Spark/Databricks pipeline for real Telecel Ghana analysis

The current branch contains both. Keep this project reference focused on code surface, workflows, and data conventions. Current research results and hardcoded benchmark facts live in `docs/research.md`.

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

For current model-comparison results and research direction, see `docs/research.md`.

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

## Research Results

Current benchmark metrics, ablation findings, and research direction live in `docs/research.md`.
