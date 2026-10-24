# seqcredit

Credit-risk modelling with synthetic transaction data, comparing borrower-level features with transaction sequences. Generated data is calibrated to aggregate telco statistics and does not represent real customer records.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --group dev
```

## Run

Run the tests:

```powershell
uv run python -m pytest
```

Run a small end-to-end experiment:

```powershell
.\src\scripts\test-smoke.ps1
```

See [the project guide](docs/PROJECT.md) for the workflow, configuration, and data boundaries.
