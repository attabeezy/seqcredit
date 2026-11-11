# Project guide

## Rescoped study

Seqcredit develops a reproducible framework for creditworthiness assessment, operationalized as credit-risk prediction from alternative mobile-money transaction data in a controlled synthetic setting. The framework compares theory- and literature-grounded static, temporal, and relational representations for simulated future nonrepayment prediction using one accepted synthetic sample and one fixed experimental design. It is not a programme for finding the best possible credit-risk architecture or for establishing deployment readiness.

The methodology takes structural inspiration from *A Comparative Analysis of Few-Shot Learning Models for Text-Based Diagnosis of Crop Pests and Diseases in Ghana*: state a practical question, document the data and preparation, compare a small set of meaningful approaches fairly, assess distinct kinds of performance, and let the conclusion follow the evidence. The reference does not prescribe our models, data design, or claims.

## Research question

Within the accepted simulated population, how can grounded static, temporal, and relational representations be evaluated fairly for predicting future nonrepayment from the same alternative mobile-money borrower histories?

The study has four steps:

1. Set and document simulator parameters from the preserved evidence, explicitly distinguishing evidence from assumptions.
2. Generate and verify one frozen synthetic sample.
3. Construct a consolidated, literature-backed feature representation and fit the prespecified models on identical borrower partitions, with preprocessing learned only from inner-training borrowers.
4. Compare held-out ranking and probability-quality results, then interpret them with paired uncertainty and stated limitations.

## Fixed comparative methodology

| Element | Decision |
| --- | --- |
| Data | One accepted synthetic sample, seed 42 |
| Comparators | Logistic regression, LightGBM, GRU, LSTM, TCN, Transformer encoder, and one graph neural network subject to the graph-feasibility gate below |
| Fairness of comparison | Identical eligible histories and borrower partitions; no test-informed preprocessing or selection |
| Primary outcome | Average precision |
| Supporting outcomes | ROC-AUC, Brier score, log loss, ECE, and descriptive fixed-threshold metrics |
| Model selection | Training/validation only; the existing test set has already been inspected, so expanded-model results are exploratory unless a separately accepted untouched evaluation is created |
| Interpretation aids | Validation-fitted calibration and one temporal-order shuffle for temporal models; prior controls remain historical evidence rather than an expanding control programme |
| Uncertainty | Paired borrower bootstrap intervals conditional on the fitted models and frozen sample |

Ranking, calibration, and a decision threshold answer different questions. A stronger ranking score does not establish calibrated risks or an operational lending policy.

## Model and feature rationale

The model suite is a comparison of distinct representational assumptions, not a catalogue of fashionable architectures.

The data-to-results workflow, initial sources, and evidence required before implementation are recorded in [the experiment plan](EXPERIMENT.md).

| Family | Models | Question answered |
| --- | --- | --- |
| Static | Logistic regression; LightGBM | Do aggregates suffice, and do nonlinear interactions improve on a linear baseline? |
| Gated sequence | GRU; LSTM | Does learned recurrent memory improve on static summaries, and does the LSTM memory cell change the result? |
| Causal convolution | TCN | Can a finite receptive-field temporal representation capture predictive transaction patterns without recurrence? |
| Attention | Transformer encoder | Does content-dependent attention over events add value beyond recurrence or convolution? |
| Relational | One GNN | Does a causal transaction-counterparty graph add information unavailable in each borrower's event sequence? |

The GNN is conditional, not automatic. Before implementation, a short feasibility note must define nodes, pre-index edges, temporal cutoff, borrower-to-counterparty linkage, and anti-leakage checks. It must also identify a relevant graph-learning or financial-network source. If the accepted data cannot support that representation without identifiers, post-index information, or an implausible graph construction, the GNN is excluded and the reason is reported.

Advanced features replace redundant aggregates; they are not appended to create a larger, less interpretable feature list. The feature dictionary must map every retained feature to its source events, time window, direction of aggregation, and literature or domain rationale. The intended consolidated families are:

- recency, frequency, monetary level, velocity, and stability;
- transaction rhythm and cyclical regularity;
- payment discipline and liquidity stress; and
- relationship or counterparty diversity only where the graph-feasibility gate is met.

Feature definitions, transformations, and the removal of redundant existing aggregates must be completed from literature and domain rationale before fitting final models. Test metrics cannot choose among feature families or model designs.

## Proportionate validation

Validation is deliberately adequate rather than exhaustive:

1. Use the shared stratified train/validation/test borrower partition and fit all preprocessing, class weights, early stopping, and calibration on the non-test partitions only.
2. Report AP as primary, ROC-AUC as secondary, and Brier score, log loss, and ECE separately for probability quality.
3. Compare models on the same held-out borrowers with paired bootstrap intervals.
4. Apply one prespecified temporal-order shuffle to the temporal models as a representation check. Do not automatically run label permutations, repeated random seeds, feature ablations, slice analyses, or stress-test programmes.
5. Preserve the existing label-permutation result as a limitation of the historical GRU comparison; do not treat it as a reason to create an open-ended control pipeline.

## Scope boundaries

The active deliverable is a clear report of the recorded study. It must accurately describe the synthetic data, the fair comparison, the results, and their limits.

The following are out of scope for this study and must not be added as routine follow-up work:

- feature families without a documented rationale or a feature dictionary;
- model families outside the prespecified comparison, including hybrids or additional graph variants;
- SHAP, Fairlearn, deployment reason codes, or lending-policy design;
- fresh stress-test, slice-analysis, label-permutation, or random-seed programmes;
- new remote orchestration or audit tooling beyond what is needed to reproduce the recorded evidence.

The existing label-permutation result is an unresolved limitation. It should be reported plainly. Any new diagnostic requires a separate, explicit decision and should be described as exploratory because the test set has already been inspected.

## Project state and frozen contract

On 2026-10-24, the user accepted the parameter set and generated sample. The acceptance record is `src/configs/accepted_dataset.json`.

- **Inputs:** `experiment_design.json`, `docs/calibration/ledger.json`, and `docs/calibration/category_mapping.json` are frozen.
- **Data:** the 92,503,821-byte accepted dataset (`SHA-256: 6ec8bfc5...`) is fixed.
- **Policy:** normal loading and training reject changed parameters or dataset bytes. A revision requires explicit authorization, a new audit, and a new acceptance record.

## Reproducibility and execution

- **Environment:** Python 3.13; `uv sync --group dev`.
- **Entrypoints:** `python dataset.py` and `python train.py`.
- **Checks:** `python -m pytest -q`, `tests/smoke.py`, and `tests/pilot.py`.
- **Remote execution:** existing WSL/Google Colab helpers are retained only to reproduce the recorded comparison or perform an explicitly authorized follow-up. They are infrastructure, not a new research objective.

## Archive and retirement

- **archive/** contains private historical evidence. Do not execute or publish it.
- **bulk-clean/** holds retired tools and superseded runs. It is excluded from Git.
