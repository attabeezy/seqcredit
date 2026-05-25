# Research Notes for `real-data` Branch

## Summary

This document is a compact research-facing record of what the current branch actually contains and what the tracked artifacts currently support.

The repository serves two research purposes:

- synthetic benchmark development
- real-data pipeline development for Databricks

The tracked result artifacts in `data/` currently support the synthetic benchmark much more cleanly than they support the real-data workflow.

## Synthetic Benchmark Findings Backed by Tracked Artifacts

The checked-in benchmark files are:

- `data/cv_results_y_default.csv`
- `data/cv_results_y_bad.csv`
- `data/significance_tests.csv`
- `data/ablation_features.csv`
- `data/tuning_results.csv`
- `data/cv_manifest.json`

### Main synthetic result

On the checked-in synthetic benchmark artifacts:

- static models dominate the standalone `LSTM`
- `HybridLSTM` performs better than `LSTM`, but the strongest checked-in metrics are still from static models
- `behavioural_diversity` and `loan_history` are the two most important feature groups in the tracked ablation file

### Backed metric snapshot

`y_default` mean AUC-ROC:

- `LogisticRegression`: `0.9143`
- `RandomForest`: `0.8836`
- `XGBoost`: `0.8809`
- `LightGBM`: `0.8777`
- `HybridLSTM`: `0.8501`
- `LSTM`: `0.5872`

`y_bad` mean AUC-ROC:

- `RandomForest`: `0.7269`
- `LogisticRegression`: `0.7254`
- `LightGBM`: `0.7152`
- `XGBoost`: `0.7098`
- `HybridLSTM`: `0.7054`
- `LSTM`: `0.5334`

### Backed ablation snapshot

From `data/ablation_features.csv`:

- baseline `ALL_FEATURES`: `0.8836`
- `DROP_behavioural_diversity`: `0.7941`
- `DROP_loan_history`: `0.7796`
- `ONLY_loan_history`: `0.7890`
- `ONLY_behavioural_diversity`: `0.7358`

These tracked artifacts support a synthetic-benchmark claim that aggregate behavioral features matter more than transaction order alone.

## Current Code Surface Beyond the Tracked Benchmark

The current codebase extends beyond the checked-in synthetic benchmark artifacts.

### Present in code

- `GRUModel`
- `HybridGRUModel`
- `TransformerModel`
- `HybridTransformerModel`
- `real_data_pipeline.py`
- runtime path overrides in `config.py`
- `run_full_benchmark.py`
- `reset.py`

### Important distinction

The tracked benchmark artifacts still describe a `HybridLSTM` benchmark configuration. Current `run_cv_benchmark.py` imports `LSTMModel` and `GRUModel` for sequential evaluation. This branch therefore contains a broader current code surface than the tracked result set alone shows.

## Real-Data Research Status

The real-data pipeline is implemented in `src/seqcredit_model/real_data_pipeline.py` and built around a leakage-aware temporal design:

- identify the most recent loan disbursement as the index loan
- derive labels only from events after the index loan
- derive features only from events before the index loan

Public API:

```python
build_pipeline(df, output_dir=None, min_followup_days=60)
```

What is not cleanly represented in tracked artifacts today:

- a single authoritative real-data benchmark CSV set analogous to the synthetic CV files
- a branch-faithful, tracked metrics table for the real-data path

That means this repository currently supports stronger documentation claims for the synthetic benchmark than for the real-data results.

## Current Research Questions Still Mapped to Code

- How much of the signal comes from `behavioural_diversity` versus `loan_history`?
- How should current GRU and transformer implementations be evaluated relative to the older tracked `HybridLSTM` artifacts?
- What tracked artifact format should become the authoritative result layer for the real-data workflow?
- Which benchmark outputs should be regenerated so the CSVs match the current script lineup?

## Documentation Rule

For this branch, use the following ordering of trust:

1. Python code in `src/seqcredit_model/`
2. tracked `data/*.csv` and `data/*.json`
3. notebooks or external paper drafts only when they do not conflict with 1 or 2
