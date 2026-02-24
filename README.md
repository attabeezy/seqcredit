# Sequential Deep Learning for Credit Risk Modeling

Temporal feature engineering and sequential deep learning for credit risk prediction using mobile money transaction data.

See [docs/REPORT.md](docs/REPORT.md) for full technical details.

---

## Setup

```bash
git clone https://github.com/Attabeezy/sequential-crm-for-dce.git
cd sequential-crm-for-dce
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Open the main notebook:

```bash
jupyter notebook notebooks/credit_risk_prediction_v1c.ipynb
```

---

## Project Structure

```
sequential-crm-for-dce/
├── data/
│   ├── user_transactions/        # Per-user transaction CSVs
│   ├── user_features_engineered.csv
│   └── user_summary_extended.csv
├── src/
│   ├── data/
│   │   ├── feature_engineering.py
│   │   └── synthetic_data.py
│   └── models/
│       ├── credit_model.py
│       └── lstm_test.py
├── notebooks/
│   ├── credit_risk_prediction_v1c.ipynb  # main
│   ├── syn_data_gen.ipynb
│   └── ctgan_syn_data_gen.ipynb
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
