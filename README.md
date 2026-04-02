# Sequential Deep Learning for Credit Risk Modeling

Temporal feature engineering and sequential deep learning for credit risk prediction using mobile money transaction data.

See [docs/PROJECT.md](docs/PROJECT.md) for full technical details.

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
