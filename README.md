# Sequential Deep Learning for Credit Risk Modeling

**Temporal feature engineering and sequential deep learning for credit risk prediction using mobile money transaction data.**

A comprehensive machine learning pipeline that combines traditional statistical models (Logistic Regression, XGBoost) with advanced sequential neural networks (LSTM) to predict credit default risk from mobile money transaction patterns. This project demonstrates the value of sequential modeling for financial risk assessment while maintaining interpretability and operational efficiency.

---

## 🎯 Key Features

- **Temporal Feature Engineering**: Advanced feature extraction capturing transaction patterns, frequency, amounts, and balance dynamics
- **Synthetic Data Generation**: Realistic mobile money transaction data with diverse borrowing behaviors and credit patterns
- **Multi-Model Architecture**: Compare baseline statistical models (Logistic Regression, XGBoost) with deep learning approaches (LSTM)
- **Individual User Sequences**: Transaction data stored per-user for optimal sequential modeling
- **Real-World Data Foundation**: Models calibrated on authentic mobile money patterns from Ghana (Feb-Sep 2024)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Data](#-data)
- [Usage Examples](#-usage-examples)
- [Models](#-models)
- [Notebooks](#-notebooks)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sequential-crm-for-dce.git
cd sequential-crm-for-dce

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run feature engineering
python -c "
from src.data.feature_engineering import TemporalTransactionFeatureEngineer
engineer = TemporalTransactionFeatureEngineer()
features = engineer.extract_all_features(df)
"

# 4. Open main notebook
jupyter notebook notebooks/credit_risk_prediction_v1c.ipynb
```

---

## 💻 Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scikit-learn` | Machine learning utilities |
| `tensorflow` | Deep learning (LSTM models) |
| `xgboost` | Gradient boosting |
| `matplotlib`, `seaborn` | Visualization |
| `ctgan` | Synthetic data generation (optional) |

---

## 📁 Project Structure

```
sequential-crm-for-dce/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── synthetic_transactions_calibrated.csv
│   │   ├── user_features_engineered.csv
│   │   └── user_summary_extended.csv
│   └── user_transactions/            # Individual user transaction files
│       ├── USER_000000.csv
│       ├── USER_000001.csv
│       └── ... (up to 2,000 users)
├── src/
│   ├── data/
│   │   ├── feature_engineering.py    # Temporal feature extraction
│   │   └── synthetic_data.py         # Synthetic data generator
│   ├── models/
│   │   ├── credit_model.py           # Model implementations
│   │   └── lstm_test.py              # LSTM testing utilities
│   └── utils/                        # Helper functions
├── notebooks/
│   ├── credit_risk_prediction_v1c.ipynb  # Main model (recommended)
│   ├── credit_risk_prediction_v1b.ipynb  # Enhanced version
│   ├── credit_risk_prediction_v1a.ipynb  # Initial exploration
│   ├── syn_data_gen.ipynb                # Synthetic data generation
│   └── ctgan_syn_data_gen.ipynb          # CTGAN variant
├── docs/
│   ├── SESSION.md                   # Detailed project log
│   └── TECHNICAL_REPORT.md          # Technical specifications
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 📊 Data

### Real-World Data
- **Source**: Mobile money transactions from Ghana
- **Period**: February - September 2024 (200 days)
- **Transactions**: 482 unique transactions
- **Users**: Anonymized customer base

### Synthetic Data
- **Generated Users**: 2,000 synthetic profiles
- **Total Transactions**: 29,994 transactions
- **Borrowing Archetypes**: 
  - Responsible borrowers (consistent repayment)
  - Risky borrowers (irregular patterns)
  - Defaulters (payment failures)

### Data Organization
Each user's transaction history is stored as an individual CSV file in `data/user_transactions/` with the following structure:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `transaction_date` | Date of transaction |
| `transaction_type` | Type (DEPOSIT, WITHDRAWAL, TRANSFER, CREDIT, LOAN_REPAYMENT) |
| `amount` | Transaction amount |
| `balance_after` | Account balance post-transaction |

---

## 💡 Usage Examples

### 1. Feature Engineering

Extract temporal features from raw transaction data:

```python
from src.data.feature_engineering import TemporalTransactionFeatureEngineer

# Initialize engineer
engineer = TemporalTransactionFeatureEngineer()

# Extract all features from transaction dataframe
df_features = engineer.extract_all_features(df_transactions)

# Create user-level summary
user_summary = engineer.create_user_level_summary(df_transactions)

print(df_features.head())
```

**Key Features Generated**:
- Transaction frequency and velocity
- Average/median amounts
- Balance statistics
- Seasonal patterns
- User risk indicators

### 2. Synthetic Data Generation

Generate realistic synthetic mobile money datasets:

```python
from src.data.synthetic_data import CalibratedMoMoDataGenerator

# Initialize generator with desired parameters
generator = CalibratedMoMoDataGenerator(
    n_users=2000,
    avg_transactions_per_user=15,
    start_date='2024-01-01',
    duration_days=180,
    output_dir='data/user_transactions'
)

# Generate individual user datasets
summary_df = generator.generate_dataset()

print(f"Generated {len(summary_df)} user profiles")
print(summary_df.head())
```

**Generator Features**:
- Realistic transaction type distributions
- Borrowing behavior modeling
- Calibrated to real-world patterns
- Individual user files for sequential modeling

### 3. Model Training & Prediction

Train and evaluate credit risk models:

```python
from src.models.credit_model import CreditRiskDataLoader, CreditRiskModel

# Load and prepare data
loader = CreditRiskDataLoader(
    features_path='data/raw/user_features_engineered.csv',
    summaries_path='data/raw/user_summary_extended.csv'
)
X_train, X_test, y_train, y_test = loader.load_and_split()

# Train multiple models
models = {
    'logistic_regression': CreditRiskModel.build_logistic_regression(),
    'xgboost': CreditRiskModel.build_xgboost(),
    'lstm': CreditRiskModel.build_lstm(sequence_length=15)
}

# Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    print(f"{name}: {score}")
```

---

## 🤖 Models

This project implements three distinct approaches to credit risk prediction:

### 1. **Logistic Regression** (Baseline)
- **Type**: Linear statistical model
- **Strengths**: Interpretable, fast, good baseline
- **Best for**: Quick predictions, regulatory compliance (interpretability)

### 2. **XGBoost** (Gradient Boosting)
- **Type**: Ensemble tree-based model
- **Strengths**: High accuracy, handles non-linearity, feature importance
- **Best for**: Production environments, best overall performance

### 3. **LSTM** (Long Short-Term Memory)
- **Type**: Recurrent neural network for sequential data
- **Strengths**: Captures temporal dependencies, learns complex patterns
- **Best for**: Understanding transaction sequences, deep pattern discovery

**Model Comparison Strategy**:
All models are trained on the same train/test split to ensure fair comparison and validate whether sequential complexity provides sufficient improvement over simpler baselines.

---

## 📓 Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `credit_risk_prediction_v1c.ipynb` | **Main**: Complete pipeline with all models | ✅ Recommended |
| `credit_risk_prediction_v1b.ipynb` | Enhanced model development | Development |
| `credit_risk_prediction_v1a.ipynb` | Initial exploration | Archive |
| `syn_data_gen.ipynb` | Synthetic data generation workflow | ✅ Active |
| `ctgan_syn_data_gen.ipynb` | CTGAN-based synthetic data | Experimental |

**Start here**: Open `notebooks/credit_risk_prediction_v1c.ipynb` for the complete walkthrough.

---

## 🛠 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- Additional feature engineering approaches
- Model optimization and tuning
- Data visualization improvements
- Documentation enhancements
- Performance benchmarking

---

## 📚 Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{sequential_crm_2025,
  author       = {Benjamin Attabra},
  title        = {Sequential Deep Learning for Credit Risk Modeling},
  year         = {2025},
  url          = {https://github.com/Attabeezy/sequential-crm-for-dce},
  note         = {Mobile money transaction analysis using temporal deep learning}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Copyright © 2025 Benjamin Attabra**

---

## 👤 Author

**Benjamin Attabra** — Machine Learning Engineer  
[GitHub](https://github.com/Attabeezy) | [Email](mailto:attabeezy@gmail.com)

---

## 🤝 Support

For questions, issues, or suggestions:
- **Open an Issue**: [GitHub Issues](https://github.com/Attabeezy/sequential-crm-for-dce/issues)
- **Discussion**: Check [docs/SESSION.md](docs/SESSION.md) for detailed project background
- **Technical Details**: See [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

*Last updated: February 17, 2026*
