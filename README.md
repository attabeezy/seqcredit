# Sequential Deep Learning for Credit Risk Modeling

A **preliminary investigation and proof-of-concept framework** for credit risk modeling in African fintech contexts using mobile money transaction data.

**Primary contributions:**
1. **Open synthetic benchmark** — calibrated Ghanaian mobile money dataset (10,000 users, 5 credit archetypes) for privacy-safe credit risk research
2. **Temporal feature engineering framework** — 8 feature groups, 38 features extracted from transaction sequences; reusable pipeline for African fintech

Preliminary evaluation on this benchmark compares six models under 5-fold CV. Key finding: well-engineered static features recover most default-predictive signal (RF: 0.832 AUC-ROC); standalone LSTM collapses (0.523); Hybrid LSTM is comparable in discrimination but worse in calibration. Results are scoped to the synthetic benchmark — real-data validation (Paper B, Telecel Ghana) is ongoing.

See [docs/PROJECT.md](docs/PROJECT.md) for full technical details and limitations.

---

## Results

5-fold stratified CV on 5,952 borrowers (11.1% default rate), `RANDOM_SEED = 42`.

### Primary target: `y_default` (default vs non-default)

| Model | AUC-ROC | AUC-PR | F1 | ECE |
|---|---|---|---|---|
| RandomForest | **0.832** | **0.679** | **0.676** | 0.115 |
| LogisticRegression | 0.831 | 0.598 | 0.514 | 0.229 |
| XGBoost | 0.818 | 0.677 | 0.684 | 0.040 |
| LightGBM | 0.812 | 0.673 | 0.669 | 0.049 |
| HybridLSTM | 0.813 | 0.606 | 0.524 | 0.219 |
| LSTM | 0.523 | 0.121 | 0.182 | 0.372 |

No statistically significant pairwise differences between the top-5 models. Static tree models preferred when calibration matters (ECE ≈0.04–0.12 vs 0.22 for HybridLSTM). Standalone LSTM collapses even after regularization — sequences alone carry insufficient default-predictive signal on this synthetic benchmark.

> **Scope note:** These are preliminary findings on a controlled synthetic dataset calibrated to Ghanaian mobile money patterns. They motivate but do not substitute for real-data validation. See [Limitations](docs/PROJECT.md#limitations) in PROJECT.md.

### Secondary target: `y_bad` (late+default vs good)

| Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|
| LogisticRegression | **0.636** | 0.611 | 0.539 |
| RandomForest | 0.635 | **0.640** | 0.492 |
| XGBoost | 0.616 | 0.626 | **0.607** |
| LightGBM | 0.614 | 0.629 | 0.508 |
| HybridLSTM | 0.601 | 0.589 | 0.545 |
| LSTM | 0.507 | 0.464 | 0.461 |

Full results in `data/cv_results_y_default.csv` and `data/cv_results_y_bad.csv`. See [docs/PROJECT.md](docs/PROJECT.md) for ablation, tuning, and significance test details.

---

## Setup

```bash
git clone https://github.com/attabeezy/seqcredit-model.git
cd seqcredit-model
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Usage

Open the main notebook:

```bash
jupyter notebook notebooks/credit_risk_model.ipynb
```

---

## Project Structure

```
seqcredit-model/
├── data/
│   ├── user_transactions/           # Per-user transaction CSVs
│   ├── user_features.csv
│   ├── user_labels.csv
│   ├── synthetic_params.json        # Calibration parameters
│   ├── cv_results_y_default.csv     # 5-fold CV results (primary benchmark)
│   ├── cv_results_y_bad.csv         # 5-fold CV results (secondary target)
│   ├── significance_tests.csv       # Pairwise bootstrap significance tests
│   ├── ablation_features.csv        # Feature group ablation results
│   └── tuning_results.csv           # Hyperparameter tuning results
├── src/
│   └── seqcredit_model/
│       ├── config.py
│       ├── feature_engineering.py
│       ├── synthetic_data.py
│       ├── credit_model.py              # Model classes (LR, XGB, RF, LightGBM, LSTM, HybridLSTM)
│       ├── run_cv_benchmark.py          # 5-fold CV benchmark (primary evaluation)
│       ├── run_ablation_study.py        # Feature group ablation study
│       └── run_hyperparameter_tuning.py # Bounded-budget tuning (RF, XGBoost, LightGBM)
├── notebooks/
│   ├── credit_risk_model.ipynb          # Primary modeling notebook (all 6 models)
│   ├── credit_risk_analysis.ipynb       # Research analysis: SHAP, surrogate tree, calibration
│   ├── data_analysis.ipynb              # Descriptive statistics notebook
│   ├── credit_risk_model_gcolab.ipynb   # Google Colab variant
│   ├── credit_risk_analysis_gcolab.ipynb
│   └── data_analysis_gcolab.ipynb
├── docs/
│   ├── PROJECT.md           # Consolidated project documentation
│   └── DATACARD.md          # Dataset documentation
├── models/                  # Persisted trained models
└── requirements.txt
```

---

## Citation

```bibtex
@software{sequential_crm_2025,
  author       = {Benjamin Ekow Attabra},
  title        = {Sequential Deep Learning for Credit Risk Modeling in Data Constrained Environments(Ghana)},
  year         = {2026},
  url          = {https://github.com/attabeezy/seqcredit-model},
  note         = {Mobile money transaction analysis using temporal deep learning}
}
```

---

## License

MIT License. Copyright 2025 Benjamin Ekow Attabra.
