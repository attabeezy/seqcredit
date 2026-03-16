# Sequential Deep Learning for Credit Risk Modeling

Temporal feature engineering and sequential deep learning for credit risk prediction using mobile money transaction data.

See [docs/REPORT.md](docs/REPORT.md) for full technical details.

---

## Setup

```bash
git clone https://github.com/attabeezy/seqcredit-model.git
cd seqcredit-model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
│   └── user_labels.csv
├── src/
│   └── seqcredit_model/
│       ├── feature_engineering.py
│       ├── synthetic_data.py
│       └── credit_model.py
├── tests/
│   └── lstm_test.py
├── notebooks/
│   ├── credit_risk_model.ipynb           # Primary modeling notebook
│   ├── credit_risk_pred.ipynb            # Replication / reference
│   ├── data_generator.ipynb
│   ├── ctgan_generator.ipynb
│   └── data_analysis.ipynb               # Descriptive statistics notebook
├── docs/
│   ├── REPORT.md
│   └── SESSION.md
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
