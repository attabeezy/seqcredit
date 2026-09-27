# SeqCredit Presentation MVP

This isolated project creates a small, fully synthetic mobile-money transaction
dataset whose **broad benchmark pattern** resembles the archived real-data
study: order-aware features should modestly outperform aggregate features for
strict-default discrimination. It does not reconstruct Telecel records, reproduce
the real experiment, or produce probabilities suitable for lending decisions.

## Current milestone: synthetic calibration

The calibration target is intentionally approximate:

| Quantity | Archived real reference | Synthetic acceptance band |
|---|---:|---:|
| Strict-default prevalence | 4.13% | 3.5%-5.0% |
| Strongest static AUC-ROC | 0.721 | 0.70-0.74 |
| Order-aware AUC-ROC | 0.752 | 0.735-0.77 |
| Order-aware gain | about 0.03 | 0.015-0.05 |

The synthetic model is an order-aware sklearn surrogate, not a GRU or LSTM.
The dashboard must display the archived benchmark and synthetic demonstration
as separate evidence.

## Run

From this directory:

```powershell
python -m pip install -e .
python -m seqcredit_mvp.calibrate
python -m seqcredit_mvp.scoring data/sample_transactions.csv
streamlit run src/seqcredit_mvp/streamlit_app.py
python -m pytest
```

After dependencies are installed, launch the **Streamlit dashboard** with:

```powershell
.\run_streamlit.ps1
```

The calibration command writes:

- `artifacts/static_model.joblib`
- `artifacts/sequence_model.joblib`
- `artifacts/manifest.json`
- `data/benchmark_results.csv`
- `data/sample_transactions.csv`

All generated identifiers and transactions are fabricated.

## Input CSV schema

Each row is one fabricated transaction. The required columns are:

| Column | Meaning |
|---|---|
| `borrower_id` | Fabricated batch identifier |
| `transaction_id` | Unique fabricated transaction identifier |
| `timestamp` | Parseable transaction date and time |
| `transaction_type` | Transaction category such as `TRANSFER` or `CASH_OUT` |
| `amount` | Non-negative transaction amount |
| `balance_before` | Non-negative balance before the event |
| `balance_after` | Non-negative balance after the event |
| `fee` | Non-negative fee amount |
| `recipient_id` | Fabricated recipient identifier |

Extra columns are accepted but ignored by the model contract.

The scorer returns both model scores, a **demo-only** descriptive risk band,
the artifact version, and the scoring time. It never returns an approve/decline
decision.

## Dashboard experience

The local dashboard is an upload-first demonstration workspace with a clean
white-and-violet theme. It includes:

- drag-and-drop CSV upload with visible validation and scoring stages;
- a one-click, single-borrower fabricated demonstration;
- an overview tab with aggregate and order-aware scores, activity, and neutral
  transaction-pattern observations;
- a loan-scenario tab plotting requested amount against an explicitly labeled
  synthetic risk index;
- a transactions tab with type mix, profile indicators, and recent records; and
- a model-comparison tab separating archived real evidence from the synthetic
  benchmark on a clearly labeled focused AUC scale.

The result view also provides a plain-language, non-causal interpretation of
the selected fabricated history, batch score context, and CSV export. Pattern
scores are shown on a 0-100 index for readability and are explicitly not
probabilities, judgments about a person, or lending recommendations.

The loan-size curve is a transparent what-if illustration based on observed
synthetic capacity. Requested loan amount is not an input to the frozen model,
so the curve is not a calibrated default probability or lending recommendation.

Open <http://localhost:8501> if Streamlit does not open automatically in your browser.
The dashboard runs locally, makes no external network requests, and supports the
supplied sample or another CSV using the same documented schema.

## Development Roadmap

See [TODO.md](TODO.md) for planned features, including explainability and driver breakdowns, scenario stress testing, executive reporting, and synthetic model extensions.

