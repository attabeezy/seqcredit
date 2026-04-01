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
│   ├── user_transactions/        # Per-user transaction CSVs
│   ├── user_features.csv
│   ├── user_labels.csv
│   ├── model_comparison.csv      # Latest model results
│   └── synthetic_params.json     # Calibration parameters
├── src/
│   └── seqcredit_model/
│       ├── config.py
│       ├── feature_engineering.py
│       ├── synthetic_data.py
│       └── credit_model.py       # Model classes (LR, XGB, RF, LightGBM, LSTM, HybridLSTM)
├── notebooks/
│   ├── credit_risk_model.ipynb         # Primary modeling notebook (all 6 models)
│   ├── credit_risk_analysis.ipynb      # Research analysis: SHAP, surrogate tree, calibration
│   ├── data_analysis.ipynb             # Descriptive statistics notebook
│   ├── credit_risk_model_gcolab.ipynb  # Google Colab variant
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
  year         = {2025},
  url          = {https://github.com/attabeezy/seqcredit-model},
  note         = {Mobile money transaction analysis using temporal deep learning}
}
```

---

## License

MIT License. Copyright 2025 Benjamin Ekow Attabra.
