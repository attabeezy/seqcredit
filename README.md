# seqcredit

Develop a reproducible framework for credit risk prediction from alternative mobile-money transaction data.

## Project Structure
```text
seqcredit/
├── data/               # dataset (yara_telecel_dump.csv.gz)
├── docs/              # Project guide (PROJECT.md) and linear research report (REPORT.md)
├── src/                # Core logic (configs, colab infrastructure, and scripts)
├── artifacts/          # Run outputs, fitted models, metrics, and figures
├── tests/              # Unit and integration tests (smoke/pilot)
├── dataset.py          # Entrypoint: Data generation and validation
├── train.py            # Entrypoint: Model training and evaluation
└── AGENTS.md           # Operational guidance for AI agents
```

## Research Workflow
The study follows a deductive, four-step process:
1. **Set Parameters**: Define generator settings from evidence extracted from Analysis Notebooks A/B/C.
2. **Generate Sample**: Produce and validate a single synthetic sample (Seed 42).
3. **Train Models**: Compare grounded static (Logistic Regression and LightGBM), temporal (GRU, LSTM, TCN, Transformer), and, where feasible, relational (GNN) representations on the shared sample.
4. **Analyze Results**: Evaluate held-out performance, probability calibration, and temporal dependencies.

## Quick Start
Requires Python 3.13. Install dependencies with `uv sync --group dev`.

```powershell
# Generate/Verify Dataset
& .\.venv\Scripts\python.exe dataset.py

# Train Models & Evaluate
& .\.venv\Scripts\python.exe train.py
```

## Data & Frozen Contract
The current dataset and parameters are frozen in `src/configs/accepted_dataset.json`. Training verifies the exact parameter, generator, and dataset hashes before loading.

## Evaluation & Calibration
Models are evaluated using Average Precision (primary) and ROC-AUC (secondary). A sigmoid calibration layer is applied to raw scores to ensure probability reliability (Brier score, ECE).

## Testing
```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe tests/smoke.py
& .\.venv\Scripts\python.exe tests/pilot.py
```

## Infrastructure
Most code runs via WSL and `google-colab-cli`. `src/colab/` manages the remote execution pipeline. 

See the [Experiment Plan](docs/EXPERIMENT.md) for the data-to-results framework, the [Project Guide](docs/PROJECT.md) for operational scope, and the [Report](docs/REPORT.md) for historical results from the original representation.
