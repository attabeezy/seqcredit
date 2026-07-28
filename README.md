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
- `src/seqcredit_model/build_poster.py`
  Generates a PowerPoint research poster from checked-in results.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

Or, with [uv](https://docs.astral.sh/uv/) (uses the checked-in `uv.lock`):

```bash
uv sync
```

An editable install of the package is required either way. The package is imported as `seqcredit_model`, and several commands use `python -m`.

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

For current research results, benchmark direction, and ablation findings, see `docs/research.md`.

## Runtime Paths

`src/seqcredit_model/config.py` resolves repo-local paths by default and also supports runtime overrides with environment variables:

- `SEQCREDIT_DATA_DIR`
- `SEQCREDIT_LSTM_CACHE_FILE`
- `SEQCREDIT_RAW_SEQ_FILE`
- `SEQCREDIT_USER_FEATURES_FILE`
- `SEQCREDIT_USER_LABELS_FILE`

This is used primarily to support Databricks or temporary runtime outputs without rewriting repo constants.

## Research Results

Current benchmark tables, real-data ablation findings, and research direction are maintained in `docs/research.md`. Keep this README focused on setup, workflows, and repository orientation rather than duplicating result metrics.

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
