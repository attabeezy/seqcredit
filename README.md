# seqcredit

Reconstruct a synthetic sample from notebook-derived parameters, train static and sequence models, and analyze their results.

The research workflow is:

1. Set parameters from the evidence already extracted from Analysis Notebooks A/B/C, identifying assumptions where evidence is incomplete.
2. Generate one synthetic sample and check it against those parameters.
3. Train the static and sequence models on the same sample and borrower splits.
4. Analyze which performs better, by how much, and what the results mean, including uncertainty and limitations.

The working references are `src/configs/experiment_design.json` and the existing calibration files. You validate the parameters and sample; normal work does not revisit the notebooks. The sample reconstructs supported characteristics rather than the original customer records. Models compete without a required winner or score. See the [methodology](docs/PROJECT.md#parameter-driven-research-workflow).

Requires Python 3.13. Install dependencies with `uv sync --group dev`.

```powershell
& .\.venv\Scripts\python.exe dataset.py
& .\.venv\Scripts\python.exe train.py
```

No flags, profile or scenario selection. Seed defaults to 42. `dataset.py` creates or verifies **data/yara_telecel_dump.csv.gz**. `train.py` builds it if missing, trains logistic regression, LightGBM and GRU on the same borrower splits and history, and prints their held-out scores. Results and fitted models go under `artifacts/`.

`train.py` now fits unweighted sigmoid calibrators on validation predictions and prints calibrated average precision, ROC-AUC, Brier score, log loss and ECE. Each run saves coefficients, five-fold out-of-fold validation results, calibrated test predictions, reliability-bin counts and paired loss-difference intervals in `calibration/`. Apply a model's saved calibration JSON to its raw probability using `calibration.apply`; the base model alone still returns raw scores. Original `metrics.json`, `predictions.csv` and ranking bootstrap files retain raw scores for compatibility with controls. Validation rechecks both raw and calibrated results.

Calibration uses seed 42 and does not change the accepted data, base-model fitting or fixed 0.5 threshold. The already-inspected test set makes this a follow-up analysis. Small ECE values depend on binning and do not establish deployment validity. The [research report](docs/RESEARCH_REPORT.md#validation-fitted-calibration-follow-up) includes the verified calibration results.

The current parameters and dataset have been accepted and frozen in [accepted_dataset.json](src/configs/accepted_dataset.json). Training verifies the exact parameter, generator and dataset hashes before loading model inputs. Changed bytes require explicit new acceptance; a missing dataset can only be rebuilt into the same accepted bytes.

Borrower activity varies randomly around an expected 150 background events per borrower. Every eligible pre-loan event is used by both model types; there is no 64- or 150-event cap. The GRU uses variable-length batches with explicit padding masks. Seed 42 reproduces the same activity draws.

The data is simulated, not observed or anonymized Telecel customer records. Noisy spending levels and trends can carry predictive information; no model is guaranteed to win. Generator settings are frozen before evaluation and are not adjusted to force a ranking or AUC.

Tests are separate:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe tests/smoke.py
& .\.venv\Scripts\python.exe tests/pilot.py
```

Smoke checks a temporary 1,000-borrower pipeline. Pilot uses 5,000 temporary borrowers and exercises label-permutation and order-shuffle controls. Both clean up their temporary data and models.

See [the project guide](docs/PROJECT.md) for provenance and evaluation details.

To regenerate research figures from the recovered comparison artifacts, install the reporting extra with `uv sync --group dev --extra reporting`, then run `python src/scripts/research_report.py`.

Most code runs through WSL and google-colab-cli. `src/colab/` contains the setup, transfer, execution, monitoring and recovery helpers; keep these as part of the main workflow. The project commands remain `python dataset.py` and `python train.py` inside the prepared runtime.

The [research report](docs/RESEARCH_REPORT.md) documents the full frozen-sample comparison on Colab, its temporal-order and label-permutation controls, uncertainty and limitations. Result files, figures and required Colab transfer inputs stay in `artifacts/`. Retired extraction tools, completed-session logs and superseded CPU runs are preserved in `bulk-clean/` with their original relative paths.
