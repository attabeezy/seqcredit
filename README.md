# seqcredit research

Local credit-risk experiments using **synthetic transactions calibrated to aggregate statistics from archived telco analyses**. Generated records are simulated observations, not anonymized customer records. The calibration ledger, v2 contract and legacy leakage audit are complete. See [`docs/project.md`](docs/project.md) for the consolidated project map,
protocol, validation status, and archive boundary. The smoke run, controlled
pilots, full-size gate, and all 30 preregistered matrix runs pass independent
validation.

This `seqcredit` checkout on `default` is the sole active research workspace. Legacy
source is retained for reference under
`archive/code/legacy-model/`; Databricks is not required.

## Setup

Use Python 3.13:

```powershell
uv sync --group dev
uv run python -m pytest
```

Alternatively, in a Python 3.13 environment: `python -m pip install -r requirements.txt`. Dependencies are defined in `pyproject.toml`. The active `.venv` uses Python 3.13.14.

## Code and next steps

The research root contains:

```text
data/                  Research datasets and generated data
docs/                  Design, calibration evidence, and project notes
models/                Saved model artifacts
analyze_data.ipynb     Historical synthetic-data exploration
train_model.ipynb      Historical model training and comparison
analyze_result.ipy     Historical model interpretation (notebook JSON)
src/
  seqcredit/           Python package
  scripts/             Utilities and standalone verification scripts
  tests/               Pytest suite
  configs/             Experiment design configuration
  synthetic_params.json
```

`archive/` is a gitignored reference collection: `code/` contains retired source,
`notebooks/` contains historical notebooks and calibration exports, `evidence/`
contains historical data and publication artifacts, and `provenance/` records
original paths and hashes. Archived environments and repository metadata have
been removed. Pytest discovers `src/tests/` and script checks;
only `seqcredit` is included in the installed package.

The root notebooks use the older shared-data workflow. Run `train_model.ipynb`
before `analyze_result.ipy`; `analyze_data.ipynb` independently explores existing
data. The `.ipy` filename retains notebook JSON as requested. Use the scripts
below for the frozen v2 experiments.

## End-to-end smoke test

```powershell
.\src\scripts\test-smoke.ps1
```

Runs 30 focused checks and a 1,000-borrower diagnostic with logistic regression,
LightGBM and GRU. Each run saves complete artifacts in a new gitignored `data/runs/` directory.

- `src/seqcredit/synthesize.py`: existing generator.
- `src/synthetic_params.json`: legacy parameters awaiting provenance review.
- `src/seqcredit/pipeline.py`: transaction and borrower features.
- `src/seqcredit/credit_model.py`: loaders, models and evaluation.
- `src/tests/`: active local research checks.
- [Project map and research summary](docs/project.md).

The dashboard, Spark integration, publication tooling, historical tables, and Databricks exports are preserved locally in gitignored `archive/`. Exports remain available for calibration extraction; a fresh clone will not contain this local archive.

Existing synthetic notebooks, data, models and caches are historical artifacts. Legacy entrypoints remain unapproved for v2. The smoke run, controlled pilots, full-size gate, and preregistered matrix have passed the documented validation checks.

Run the 5,000-borrower pilot with `.\src\scripts\run-pilot.ps1`.

The v2 runner now accepts the frozen main profile and diagnostic controls directly:

```powershell
.\src\scripts\test-smoke.ps1 -Profile main -Scenario aggregate_signal -Seed 42
.\src\scripts\test-smoke.ps1 -Profile pilot -Control label_permutation
.\.venv\Scripts\python.exe src\scripts\bootstrap_predictions.py --run data\runs\<run>
```

Main-profile settings are frozen by the v2 contract; the resource-check pilot
was completed before the full-size gate and matrix.

The frozen matrix planner creates 18 headline runs (three scenarios, three
seeds, two targets) plus 12 separately labeled retrained controls:

```powershell
.\.venv\Scripts\python.exe src\scripts\run_matrix.py
```

## Citation

```bibtex
@misc{seqcredit,
  author       = {Attabra, Benjamin Ekow},
  title        = {seqcredit},
  year         = {2025},
  howpublished = {\url{https://github.com/attabeezy/seqcredit}}
}
```
