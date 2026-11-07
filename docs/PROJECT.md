# Project guide

## Core Research Workflow
The objective is to reconstruct a synthetic sample from evidence-derived parameters, train models, and analyze their results to evaluate the "sequence advantage" in credit risk prediction.

1. **Set Parameters from Evidence**: Define generator parameters based on preserved evidence from Analysis Notebooks A/B/C. Identify assumptions where evidence is incomplete.
2. **Generate Sample**: Use `dataset.py` (Seed 42) to produce `data/yara_telecel_dump.csv.gz`. Validate the sample against parameters.
3. **Train Models**: Train static and sequence models on shared borrower splits. Fit preprocessing only on inner-training borrowers.
4. **Analyze Results**: Compare held-out performance, paired uncertainty, and relevant limitations.

## Project State & Frozen Contract
On 2026-10-24, the user accepted the current parameters and sample. This acceptance is recorded in `src/configs/accepted_dataset.json`. 

**The Frozen Contract**:
- **Inputs**: `experiment_design.json`, `docs/calibration/ledger.json`, and `docs/calibration/category_mapping.json` are frozen.
- **Data**: The specific 92,503,821-byte dataset (`SHA-256: 6ec8bfc5...`) is fixed.
- **Policy**: Main loading and training reject changed parameters or dataset bytes. Any revision requires explicit user authorization, a new audit, and a new acceptance record.

## Execution Infrastructure
- **Environment**: Python 3.13, `uv sync --group dev`.
- **Runtime**: WSL Ubuntu + `google-colab-cli`.
- **Entrypoints**: `python dataset.py` and `python train.py`.
- **Validation**: `python -m pytest -q` for unit checks; `tests/smoke.py` and `tests/pilot.py` for integration.
- **Framework Strategy**: To maintain baseline stability while enabling research novelty, the project uses a hybrid stack:
    - **Keras/TensorFlow**: Used for baseline models (LR, LightGBM, GRU) to preserve existing results.
    - **PyTorch**: Used for advanced architectures (TCN, Transformers, GNNs) to leverage SOTA libraries like PyTorch Geometric.

## Future Development Roadmap (Linear Order)
To evolve the study into a literature-grounded comparative analysis, development will proceed in the following phases:

### Priority: Resolve the GRU–logistic regression comparison
Investigate the existing comparison before attributing a sequence advantage or expanding the model suite:

- **Recheck existing evidence**: Verify matched borrower IDs, predictions, preprocessing, paired AP/ROC-AUC differences and the original label-permutation implementation. Document whether any error explains the result.
- **Characterize the negative control**: Prespecify multiple label-permutation seeds and compare the resulting score distribution with the original control. Test the influence of genuine-label early stopping with a control whose stopping rule uses no genuine labels. Retain logistic regression as a comparator without early stopping, and include the planned random-noise feature baseline.
- **Assess training variability**: Prespecify matched model initialization/training seeds and repeat GRU and logistic regression fits on the same frozen sample, splits, histories and backend. Report individual results and variability alongside paired borrower uncertainty; distinguish training variability from uncertainty about new populations or simulator settings.
- **Set interpretation criteria in advance**: State the primary AP contrast, planned comparisons, uncertainty summaries and stopping rule before running follow-up experiments. Report a persistent inconclusive interval or unexplained control as unresolved rather than forcing a winner. Assess sensitivity to the early-stopping policy separately from the original fixed comparison.
- **Protect interpretation**: Keep accepted data and generator hashes fixed, retain the original outputs, and use development data for any choices. Since the held-out test results have already been inspected, label follow-up results exploratory; an independent untouched evaluation would be needed for a new confirmatory claim. Summarize what each check resolves and what remains uncertain in `REPORT.md`.

### Phase 1: Enriched Foundation (Feature Engineering)
Upgrade the 30 baseline features using **Strategy A (Replacement)** to avoid multicollinearity:
- **RFM+ Evolution**: Velocity ratios (short vs long term) and Stability indices (Spend CV, Balance Floor Decay).
- **Social Capital**: Network-based features (Reciprocity Index, Centrality, Counterparty Diversity).
- **Cyclical Patterns**: Fourier transforms for payday/weekly cycles and Harvest/Holiday ratios.
- **Behavioral Discipline**: Airtime-to-Money ratios and Payment Latency.
- **Liquidity Stress**: Zero-Balance Frequency and Time-to-Recovery (TTR).

### Phase 2: Architectural Expansion (Model Suite)
Benchmark the enriched data against a sequence of increasingly complex architectures:
- **Sequential Baseline**: LSTM (Deep memory) and TCN (Causal convolutions).
- **Attention Mechanism**: Transformer Encoders for critical event identification.
- **Data Fusion**: Tabular-Sequence Hybrids (e.g., Cross-Attention) to optimize static-temporal integration.
- **Structural Learning**: Graph Neural Networks (GNNs) via PyTorch Geometric to learn community-based risk embeddings.

### Phase 3: Rigorous Validation & Stress Testing
- **Refined Negative Control**: Implement a zero-information baseline (random noise features).
- **Temporal Saliency**: Use saliency mapping to explain the "sequence advantage."
- **Activity Stratification**: Slice results by borrower activity quantiles.
- **Tail-Risk Calibration**: Validate probability reliability for the top 5% highest-risk borrowers.

### Phase 4: Responsible AI & Final Reporting
- **Fairness & Explainability**: Integrate SHAP and Fairlearn into the core pipeline for transparent "Reason Codes."
- **Linear Synthesis**: Populate `REPORT.md` following the deductive flow: Data $\rightarrow$ Features $\rightarrow$ Architecture $\rightarrow$ Results $\rightarrow$ Inference $\rightarrow$ Conclusion.

## Archive & Retirement
- **archive/**: Contains private historical evidence. Do not execute or publish.
- **bulk-clean/**: Retirement folder for obsolete tools and superseded runs. Excluded from Git.
