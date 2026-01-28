# Sequential Deep Learning for Credit Risk Modeling

Temporal feature engineering and sequential deep learning for credit risk prediction using mobile money transaction data.

## Setup

```bash
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn ctgan
```

## Notebooks

- `credit_risk_prediction_v1c.ipynb` - Credit risk prediction
- `credit_risk_prediction_v1b.ipynb` - Enhanced version
- `credit_risk_prediction_v1a.ipynb` - Initial version
- `syn_data_gen.ipynb` - Synthetic data generation
- `ctgan_syn_data_gen.ipynb` - CTGAN synthetic data

## Data

- **Real**: 482 transactions over 200 days (Ghana, Feb-Sep 2024)
- **Synthetic**: 29,994 transactions from 2,000 users

## License

MIT - see [LICENSE](LICENSE)
