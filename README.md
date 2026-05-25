# seqcredit-model

`seqcredit-model` is a credit-risk research codebase with two connected workflows:

- a local synthetic mobile-money benchmark
- a Databricks-oriented real-data pipeline for Telecel Ghana analysis

The current branch is `real-data`. The checked-in Python package contains both tracks.

## What Is In This Repo

- `src/seqcredit_model/synthesize.py`
  Generates synthetic per-user transaction CSVs and `data/user_labels.csv`.
- `src/seqcredit_model/pipeline.py`
  Engineers transaction features and builds `data/user_features.csv`.
- `src/seqcredit_model/credit_model.py`
  Contains the shared data loader, evaluation utilities, and model classes.
- `src/seqcredit_model/real_data_pipeline.py`
  Builds leakage-aware user features and labels from a Spark DataFrame in Databricks.
- `src/seqcredit_model/run_cv_benchmark.py`
  Runs the checked-in 5-fold benchmark artifacts.
- `src/seqcredit_model/run_ablation_study.py`
  Runs RandomForest feature-group ablations.
- `src/seqcredit_model/run_hyperparameter_tuning.py`
  Runs bounded-budget tuning for the top static models.
- `src/seqcredit_model/run_full_benchmark.py`
  Clears stale benchmark outputs and re-runs the CV benchmark.
- `src/seqcredit_model/compute_bootstrap_ci.py`
  Computes bootstrap intervals for saved notebook-trained models.
- `src/seqcredit_model/reset.py`
  Removes generated models and data artifacts for a fresh run.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is required. The package is imported as `seqcredit_model`, and several commands use `python -m`.

## What Runs Where

### Local synthetic workflow

Use this path for the synthetic benchmark and local notebooks:

```bash
python -m seqcredit_model.synthesize
python -m seqcredit_model.pipeline
jupyter notebook notebooks/model.ipynb
```

Additional local scripts:

```bash
python -m seqcredit_model.run_cv_benchmark
python -m seqcredit_model.run_ablation_study
python -m seqcredit_model.run_hyperparameter_tuning
python -m seqcredit_model.run_full_benchmark
python -m seqcredit_model.reset --data --models --yes
```

### Databricks real-data workflow

`src/seqcredit_model/real_data_pipeline.py` is designed for a Spark/Databricks runtime. Its public entrypoint is:

```python
from seqcredit_model.real_data_pipeline import build_pipeline
```

Typical usage in a notebook:

```python
df = spark.sql("SELECT * FROM melodatabricks616.default.yara_dump_table")
build_pipeline(df, output_dir=None, min_followup_days=60)
```

This path writes runtime `user_features.csv`, `user_labels.csv`, and raw sequence artifacts compatible with `CreditRiskDataLoader`.

## Models In Code

Exported model classes:

- `LogisticRegressionModel`
- `XGBoostModel`
- `RandomForestModel`
- `LightGBMModel`
- `LSTMModel`
- `GRUModel`
- `HybridLSTMModel`
- `HybridGRUModel`

Additional classes currently present in `credit_model.py`:

- `TransformerModel`
- `HybridTransformerModel`

The current checked-in CV benchmark artifacts were produced by the same benchmark configuration that the current `run_cv_benchmark.py` uses: `LSTMModel` and `GRUModel` as sequential models against four static baselines, 5-fold CV, 1000-bootstrap significance tests.

## Runtime Paths

`src/seqcredit_model/config.py` resolves repo-local paths by default and also supports runtime overrides with environment variables:

- `SEQCREDIT_DATA_DIR`
- `SEQCREDIT_LSTM_CACHE_FILE`
- `SEQCREDIT_RAW_SEQ_FILE`
- `SEQCREDIT_USER_FEATURES_FILE`
- `SEQCREDIT_USER_LABELS_FILE`
- `SEQCREDIT_EPHEMERAL`

This is used primarily to support Databricks or temporary runtime outputs without rewriting repo constants.

## Checked-In Results

The committed CSVs in `data/` currently describe the synthetic benchmark artifacts checked into this branch.

### 5-fold CV means

`y_default`

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GRU | 0.7565 | 0.1160 | 0.1452 | 0.2130 | 0.3671 |
| LSTM | 0.7478 | 0.1070 | 0.1409 | 0.2151 | 0.3668 |
| XGBoost | 0.7303 | 0.1075 | 0.1566 | 0.1612 | 0.2966 |
| LightGBM | 0.7292 | 0.1032 | 0.1566 | 0.1621 | 0.2993 |
| RandomForest | 0.7275 | 0.1018 | 0.1599 | 0.1466 | 0.2947 |
| LogisticRegression | 0.7089 | 0.0920 | 0.1322 | 0.2194 | 0.3982 |

`y_bad`

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GRU | 0.7159 | 0.8145 | 0.7069 | 0.2166 | 0.1406 |
| LSTM | 0.7112 | 0.8109 | 0.7104 | 0.2170 | 0.1381 |
| XGBoost | 0.7056 | 0.8116 | 0.7000 | 0.2171 | 0.1300 |
| LightGBM | 0.7051 | 0.8115 | 0.7012 | 0.2171 | 0.1301 |
| RandomForest | 0.6965 | 0.8063 | 0.6942 | 0.2186 | 0.1299 |
| LogisticRegression | 0.6855 | 0.7967 | 0.6824 | 0.2250 | 0.1415 |

Source files:

- `data/cv_results_y_default.csv`
- `data/cv_results_y_bad.csv`
- `data/significance_tests.csv`
- `data/ablation_features.csv`
- `data/tuning_results.csv`
- `data/cv_manifest.json`

### Ablation highlights

From `data/ablation_features.csv`:

- Baseline RandomForest with all 38 features: AUC-ROC `0.8836`
- Dropping `behavioural_diversity` lowers AUC-ROC to `0.7941`
- Dropping `loan_history` lowers AUC-ROC to `0.7796`
- `ONLY_loan_history` reaches `0.7890`
- `ONLY_behavioural_diversity` reaches `0.7358`

### Tuning highlights

From `data/tuning_results.csv`:

- Tuned `LightGBM`: AUC-ROC `0.8880`
- Tuned `XGBoost`: AUC-ROC `0.8870`
- Tuned `RandomForest`: AUC-ROC `0.8855`

## Data Outputs

Synthetic pipeline outputs:

- `data/user_transactions/`
- `data/user_labels.csv`
- `data/user_features.csv`
- `data/lstm_sequences.npz`

Checked-in result artifacts:

- `data/cv_results_y_default.csv`
- `data/cv_results_y_bad.csv`
- `data/significance_tests.csv`
- `data/ablation_features.csv`
- `data/tuning_results.csv`
- `data/cv_manifest.json`

Notebook/model artifacts used by `compute_bootstrap_ci.py`:

- `data/lstm_test_arrays.npz`
- `models/*.pkl`
- `models/*.keras`
- `models/*.json`

## Notes

- Python requirement is `>=3.10` from `pyproject.toml`.
- The repo has no formal automated test suite yet.
- `docs/PROJECT.md` is the canonical project reference.
- `docs/DATACARD.md` documents the synthetic datasets.
