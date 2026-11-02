# seqcredit

Reconstruct a synthetic sample from notebook-derived parameters, train static and sequence models, and analyze their results to evaluate the "sequence advantage" in credit risk prediction.

## Project Structure
```text
seqcredit/
├── data/               # Frozen synthetic dataset (yara_telecel_dump.csv.gz)
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
3. **Train Models**: Compare static (Logistic Regression, LightGBM) and sequence (GRU, and planned LSTM/TCN/Transformer) models.
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
The current dataset and parameters are frozen in `src/configs/accepted_dataset.json`. Training verifies the exact parameter, generator, and dataset hashes before loading. Any revision requires explicit user authorization and a new acceptance record.

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

See the [Project Guide](docs/PROJECT.md) for a detailed roadmap and the [Report](docs/REPORT.md) for full results.
