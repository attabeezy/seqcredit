# Research Notes for `real-data` Branch

## Summary

This document records the current research direction for the `real-data` branch.
The branch still contains the synthetic benchmark pipeline, but the current
research narrative should be led by the Databricks real-data workflow.

Use the notebooks as follows:

- **Analysis Notebook B** is the main real-data benchmark source: static
  baselines plus `LSTM` and `GRU`.
- **Analysis Notebook A** is the real-data ablation source: static feature-group
  interpretability on the real-data cohort.

## Real-Data Cohort

Both Notebook A and Notebook B use the same raw Databricks table:

- raw table size: `374,295,424` rows and `12` columns
- eligible borrowers with at least one loan and at least `60` days of follow-up:
  `149,667`
- label distribution:
  - good (`credit_risk_label = 0`): `51,105`
  - risky / late (`credit_risk_label = 1`): `92,392`
  - default (`credit_risk_label = 2`): `6,170`
- matched feature/label users used for modeling: `149,204`

The real-data pipeline remains leakage-aware:

- index loan defines the prediction point
- features are derived from pre-index-loan behavior
- labels are derived from post-index-loan outcomes

Public API:

```python
from seqcredit_model.real_data_pipeline import build_pipeline

build_pipeline(df, output_dir=None, min_followup_days=60)
```

## Main Benchmark Direction

Notebook B is the benchmark source for model comparison. It runs four static
baselines plus recurrent sequence models over the same real-data split:

- `LogisticRegression`
- `XGBoost`
- `RandomForest`
- `LightGBM`
- `LSTM`
- `GRU`

### `y_default`

Target: `credit_risk_label == 2`.

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GRU | 0.7517 | 0.1152 | 0.1462 | 0.2083 | 0.3673 |
| LSTM | 0.7418 | 0.1034 | 0.1430 | 0.2094 | 0.3657 |
| XGBoost | 0.7211 | 0.1003 | 0.1509 | 0.1622 | 0.2987 |
| RandomForest | 0.7207 | 0.0993 | 0.1559 | 0.1484 | 0.2981 |
| LightGBM | 0.7193 | 0.1006 | 0.1503 | 0.1627 | 0.3006 |
| LogisticRegression | 0.7058 | 0.0904 | 0.1308 | 0.2205 | 0.3999 |

### `y_bad`

Target: `credit_risk_label in {1, 2}`.

| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| GRU | 0.7148 | 0.8150 | 0.7232 | 0.2124 | 0.1261 |
| LSTM | 0.7132 | 0.8134 | 0.7181 | 0.2160 | 0.1380 |
| XGBoost | 0.7068 | 0.8141 | 0.7022 | 0.2168 | 0.1318 |
| LightGBM | 0.7063 | 0.8134 | 0.7014 | 0.2169 | 0.1318 |
| RandomForest | 0.6980 | 0.8090 | 0.6947 | 0.2182 | 0.1314 |
| LogisticRegression | 0.6876 | 0.7995 | 0.6824 | 0.2245 | 0.1434 |

### Interpretation

The current direction is to lead with sequential modeling, especially `GRU`,
because it gives the strongest AUC-ROC on both real-data targets in Notebook B.
The static models remain important baselines and calibration comparators, but
they are no longer the headline model family for the real-data workflow.

Notebook B displays archived significance-test output, but those results are not
independently verifiable: comparisons are duplicated and the underlying
out-of-fold prediction archive is missing. Do not use statistical significance,
confidence intervals, or fold variation as central evidence. The preserved mean
benchmark tables and the separate Notebook C rerun support a descriptive
ranking claim only.

For the `y_default` target specifically, there is a clean discrimination-vs-calibration
split: recurrent models (GRU, LSTM) lead on AUC-ROC while tree models (RandomForest,
XGBoost, LightGBM) are substantially better calibrated (lower Brier and ECE). This
framing — *deep learning finds the defaulters, tree models price the risk* — is
defensible for `y_default` but does not hold for `y_bad`, where GRU achieves the
best Brier and ECE of all models. Scope any such claim explicitly to the rare-event
default target.

## Real-Data Ablation

Notebook A is the source for real-data feature-group ablation. It selects
`RandomForest` for ablation on `y_default` using the static-model rank-sum rule.

The ablation should be cited as a separate interpretability analysis, not mixed
into Notebook B's main benchmark table.

Notebook A drop-one-group results:

| Dropped group | Features left | AUC-ROC | Delta vs all features | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| `txn_type_mix` | 44 | 0.717167 | -0.007211 | 0.151800 | 0.303113 |
| `amount_stats` | 40 | 0.719217 | -0.005161 | 0.151288 | 0.303318 |
| `activity_intensity` | 44 | 0.723101 | -0.001277 | 0.148983 | 0.299101 |
| `behavioural_diversity` | 44 | 0.723889 | -0.000489 | 0.148498 | 0.296644 |
| `loan_history` | 38 | 0.724197 | -0.000180 | 0.145401 | 0.290541 |
| `temporal_patterns` | 43 | 0.724402 | 0.000024 | 0.148361 | 0.297696 |

### Ablation inferences

**Largest conditional drops.** `txn_type_mix` and `amount_stats` produce the
largest drop-one AUC changes (−0.0072 and −0.0052 respectively). Drop-one
effects are conditional and non-additive, so they must not be summed or treated
as shares of model performance.

**Composition and amount statistics contain the strongest static signal.** In
this RandomForest run, removing transaction-type mix or amount statistics causes
the largest degradation. This supports a predictive-signal claim, not an
explanation of borrower behavior.

**Temporal patterns add no measurable incremental discrimination in this run.**
Dropping `avg_hours_between_txns`, `pct_weekend_txns`, `pct_night_txns`, and
`pct_early_morning_txns` changes AUC-ROC by +0.000024. This does not establish
that timing is generally irrelevant.

**Loan history adds little incremental static discrimination in this run.** The
drop is −0.0002 AUC-ROC. The surviving evidence does not establish whether this
is caused by first-time borrowing, sparsity, redundancy, or another mechanism.

**Calibration note.** The Brier and ECE columns track AUC-ROC across all groups:
dropping `txn_type_mix` or `amount_stats` also worsens calibration. Dropping
`loan_history` *improves* both Brier (0.1454 vs baseline 0.1484) and ECE
(0.2905 vs 0.2981), suggesting those features add slight miscalibration noise
without contributing to discrimination.

## Current Research Questions

- Do the GRU gains remain stable across reruns and hardware/runtime variation?
- Which calibration strategy should be used for recurrent models, given their
  stronger AUC-ROC but weaker Brier/ECE than the best static baselines?
- If the data is restored, can preprocessing be moved inside each CV fold and
  the untouched 20% test population be evaluated?

## Use of Results

Use this note as the current real-data research summary:

- cite Notebook B for the main real-data benchmark comparison
- cite Notebook A for the real-data feature-group ablation
- keep tracked synthetic artifacts out of this real-data result narrative
