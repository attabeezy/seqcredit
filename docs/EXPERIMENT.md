# Experiment plan

## Purpose and study boundary

Seqcredit develops a reproducible framework for credit risk prediction from alternative mobile-money transaction data in a controlled synthetic setting. It asks how grounded static, temporal, and, where feasible, relational representations compare for future nonrepayment prediction on one accepted synthetic population. It does not seek a universally best credit-risk model, validate lending deployment, or infer performance for real customers.

The study takes structural inspiration from *A Comparative Analysis of Few-Shot Learning Models for Text-Based Diagnosis of Crop Pests and Diseases in Ghana*: motivate a practical question, document data preparation, compare meaningful approaches fairly, evaluate distinct kinds of performance, and let the conclusion follow the evidence. It does not adopt that paper's few-shot design, data regime, or deployment claims.

The workflow proceeds from the data to the results. A model or feature is included only when the accepted data and a documented rationale support it.

## 1. Data audit and frozen design

The accepted source is one synthetic event dataset generated from the frozen simulator inputs and acceptance record in `src/configs/accepted_dataset.json`. The audit must establish:

- the pre-index event window, borrower partitions, and simulated nonrepayment outcome;
- which quantities are supported by preserved notebook evidence and which are simulator assumptions;
- that all candidate model inputs are derived from pre-index events only; and
- that post-index events, outcomes, identifiers, and simulator-private state are excluded.

The accepted data and generator assumptions cannot be altered to improve model results. Any change requires explicit authorization, a new audit, and renewed acceptance.

## 2. Consolidated feature dictionary

The existing aggregates are reviewed for redundancy and replaced, where justified, with a smaller feature dictionary. Features are not added merely to enlarge the representation.

The candidate families are:

| Family | Intended information |
| --- | --- |
| Recency, frequency, monetary level, velocity, and stability | Activity level, recent change, and variability over defined pre-index windows |
| Transaction rhythm and cyclical regularity | Timing and regularity of event activity |
| Payment discipline and liquidity stress | Pre-index balance and repayment-related behaviour available in the accepted events |
| Relationship and counterparty diversity | Pre-index relational structure, only if the graph-feasibility gate passes |

Before fitting final models, every retained feature requires a dictionary entry containing its source events, lookback window, transformation, aggregation direction, rationale, potential redundancy, and leakage risk. The feature dictionary is chosen from literature and domain reasoning on training/validation data; test metrics cannot choose features.

## 3. Representation feasibility

The data audit determines which representations are defensible.

- **Static representation:** consolidated borrower-level features.
- **Sequence representation:** ordered, pre-index event histories with explicit padding masks.
- **Graph representation:** a GNN is permitted only after a short feasibility note defines nodes, causal pre-index edges, temporal cutoff, counterparty linkage, and anti-leakage checks.

If a causal graph cannot be built without identifiers, post-index information, or an implausible construction, the GNN is excluded and the reason is reported. This is a legitimate outcome, not a failure to complete the study.

## 4. Model suite and evidence base

The suite compares distinct representational hypotheses rather than collecting architectures without a purpose.

| Family | Models | Question |
| --- | --- | --- |
| Static | Logistic regression; LightGBM | Do consolidated aggregates suffice, and do nonlinear interactions add value over a linear baseline? |
| Gated sequence | GRU; LSTM | Does learned recurrent memory improve on static summaries, and does the LSTM memory cell change the result? |
| Causal convolution | TCN | Can a finite-receptive-field temporal representation capture predictive patterns without recurrence? |
| Attention | Transformer encoder | Does content-dependent attention over events add value beyond recurrent or convolutional sequence models? |
| Relational | One GNN, conditional | Does a causal transaction-counterparty graph add information unavailable in borrower-local sequences? |

The initial literature rationale is:

| Choice | Rationale to test | Foundational evidence |
| --- | --- | --- |
| LSTM | A gated memory cell is distinct from the existing GRU and may retain longer dependencies. | Hochreiter and Schmidhuber, *Long Short-Term Memory*, 1997, doi:10.1162/neco.1997.9.8.1735. |
| TCN | Causal convolutions provide a non-recurrent temporal representation with a controllable receptive field. | Bai, Kolter, and Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, 2018, arXiv:1803.01271. |
| Transformer encoder | Self-attention provides content-dependent event interactions and a non-recurrent comparison with TCNs and gated models. | Vaswani et al., *Attention Is All You Need*, 2017, arXiv:1706.03762. |
| GNN | Message passing can represent pre-index counterparty or transaction-network structure omitted by borrower-local sequences. | Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, 2017, arXiv:1609.02907. |
| Consolidated features | Credit-scoring feature sets can contain redundant and nonrelevant variables; feature selection should seek informative, less redundant representations. | Larodec, Ghazali, and Larodec, *A Three-Stage Feature Selection Using Quadratic Programming for Credit Scoring*, 2013, doi:10.1080/08839514.2013.823327. |
| Transaction-based features | Transaction histories can provide alternative signals for credit assessment, but their value is context-dependent. | Jiang et al., *Deciphering Big Data in Consumer Credit Evaluation*, 2021, doi:10.1016/j.jempfin.2021.01.009. |

Before implementation, each final model and feature family needs a concise source record: citation and exact claim, representation or definition, required data, leakage or redundancy risk, and comparison hypothesis. This record supports design decisions; it does not predict a performance gain.

## 5. Training protocol

All eligible models use the same stratified borrower train/validation/test partition and the same frozen sample. Preprocessing, class weights, early stopping, feature selection, and calibration are fitted without test data. Model settings are fixed before the final held-out comparison.

The historical test set has already been inspected. Consequently, results for the new feature dictionary and expanded suite are exploratory unless a separately accepted untouched evaluation is created. This does not invalidate the comparison, but it limits confirmatory language.

## 6. Results and interpretation

Average precision is primary. ROC-AUC is secondary. Brier score, log loss, and ECE evaluate probability quality separately; fixed-threshold metrics remain descriptive because no operational decision threshold or cost model has been chosen.

Results must answer distinct questions:

1. Which model ranks nonrepayment cases best on the shared test borrowers?
2. Are predicted probabilities reliable after validation-fitted calibration?
3. Does preserving event order matter for temporal models?
4. What conclusions remain conditional on the simulator, fitted models, and inspected test set?

The existing logistic-regression, LightGBM, and GRU results are historical evidence for the original representation only. They are not results for the consolidated features or expanded suite.

## 7. Proportionate validation

Validation is adequate but intentionally limited:

1. shared train/validation/test borrowers and non-test fitting of preprocessing, calibration, and selection;
2. paired bootstrap intervals for model differences on the held-out borrowers;
3. validation-fitted probability calibration; and
4. one prespecified temporal-order shuffle for temporal models.

There is no automatic repeated-seed, label-permutation, feature-ablation, slice-analysis, or stress-test programme. The earlier label-permutation finding is retained as an unresolved limitation of the historical GRU comparison. It must be reported plainly, but it does not authorize open-ended new controls.

## 8. Paper narrative and scope guardrails

The final paper should follow this order: problem and question; data provenance and construction; feature dictionary and representation feasibility; comparative methodology; results; interpretation and limitations; conclusion.

It must identify the data as synthetic, distinguish measured outcomes from explanations, and avoid claims of a generally best architecture, real-customer validity, or deployment readiness. Further model families, ungrounded features, explainability systems, fairness tooling, lending-policy design, and new validation programmes require separate scope decisions.

## Appendix A. Retired repeated-permutation protocol

The following historical exploratory protocol is retained so the existing orchestration code and past plan remain reproducible. It is not part of the active proportionate-validation workflow and may run only with explicit authorization.

Use the accepted dataset, frozen inputs, original borrower splits, all eligible histories, the same T4 backend, environment, and deterministic settings. Verify recorded hashes before and after execution. Fit preprocessing and class weights on inner-fit borrowers only. Keep model initialization and minibatch seed at 42; do not adjust settings or seeds from results.

For each permutation seed 1051 through 1060, permute only the inner-fit binary target using NumPy `default_rng(seed).permutation`, preserving class counts. Keep inner-stop, validation, and test labels unchanged.

1. Fit balanced logistic regression on permuted inner-fit labels; it has no early stopping.
2. Fit GRU on the same permuted labels with the historical genuine-label inner-stop policy, patience 5, minimum delta 0.0001, and 20-epoch maximum.
3. Fit the same GRU for exactly one epoch, without consulting genuine inner-stop labels.

Also fit one genuine-label headline for each GRU stopping policy and one genuine-label logistic regression. Save borrower-aligned predictions, exact permutation identities, histories, source and environment hashes, fitted artifacts, and failures. Report every seed's AP and ROC-AUC, the original seed's rank, and the paired difference between the two GRU stopping policies. Interpret this only as a limited diagnostic: it is not a permutation-test null distribution, proof of no leakage, or a replacement for training- or generator-seed uncertainty.
