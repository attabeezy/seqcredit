# Project Log: Sequential CRM for DCE - Credit Risk Modeling
**Date**: February 16, 2026
**Author**: Attabra Benjamin Ekow
**Project**: Building a credit risk prediction system using mobile transaction data.

---

## Project Overview

This project aims to explore how transaction data can be used to predict credit risk, focusing on both traditional (static) and time-aware (sequential) modeling approaches. I'm especially interested in how much value sequential models add given their complexity. The work involves a few key steps: understanding real transaction patterns, generating realistic synthetic data with different borrowing behaviors, engineering useful features from this data, and finally building and comparing different predictive models.


---

## Understanding the Data

My journey started by looking at some real-world mobile money transaction data. This helped me get a feel for how people transact, what kinds of transactions are common, and how balances typically behave. This initial exploration was crucial for understanding the raw material I'd be working with. I focused on patterns like:

*   **Transaction Amounts:** How much money moves around? (e.g., typical amounts, range)
*   **Transaction Types:** What are the most common activities? (e.g., transfers, debits, payments)
*   **Temporal Patterns:** Are there specific times of day or week when activity is higher?
*   **Balance Behavior:** How do user balances change over time?

This foundational understanding informed how I designed the feature engineering and synthetic data generation processes.

---

## Feature Engineering

To make sense of the raw transaction data, I developed a feature engineering pipeline. This process extracts meaningful information from transaction histories, transforming them into features that predictive models can use. These features capture various aspects of user behavior, such as transaction frequency, amount patterns, and balance dynamics.

The core logic for this is in `src/data/feature_engineering.py`.

### Usage

```python
import sys

sys.path.append('../src')
from data.feature_engineering import TemporalTransactionFeatureEngineer

# Initialize and extract features
engineer = TemporalTransactionFeatureEngineer()
df_features = engineer.extract_all_features(df)

# Create user-level summary
user_summary = engineer.create_user_level_summary(df)
```

## Synthetic Data Generation

Since real-world financial data is sensitive and often scarce, I built a synthetic data generator. This tool creates realistic transaction datasets that mimic real mobile money user behavior, allowing me to develop and test models without privacy concerns.

Initially, the generator focused on general transaction patterns. However, to specifically address credit risk, I had to significantly enhance it. This involved:

*   **Introducing Credit Transactions:** Adding `CREDIT` (loan disbursements) and `LOAN_REPAYMENT` transaction types.
*   **Modeling Borrowing Behavior:** Simulating different user archetypes (e.g., responsible borrowers, risky borrowers, defaulters) to create a diverse dataset for credit risk prediction.
*   **Generating Individual User Files:** Instead of one large file, the generator now creates separate CSV files for each user's transactions, stored in `data/user_transactions/`. This design is better suited for sequential modeling.

The primary script for this is `src/data/synthetic_data.py`.

### Usage

```python
import sys
sys.path.append('src')
from data.synthetic_data import CalibratedMoMoDataGenerator

# Generate individual user datasets
generator = CalibratedMoMoDataGenerator(
    n_users=10000,
    avg_transactions_per_user=15,
    start_date='2024-01-01',
    duration_days=180,
    output_dir='data/user_transactions'
)

summary_df = generator.generate_dataset()
```






## Credit Risk Modeling

With the data prepared and features engineered, the next step was to build models to predict credit default. I explored both traditional static models and more advanced sequential models (LSTMs) to see which performed better and when the complexity of sequential models was justified.

The goal is a binary classification: predicting whether a borrower will default on a loan. Users who haven't taken loans are excluded from this specific prediction task.

### Core Components:

The main modeling logic resides in `src/models/credit_model.py` and is orchestrated through `notebooks/credit_risk_modeling.ipynb`.

*   **Data Loading and Splitting:** A `CreditRiskDataLoader` handles loading the various datasets, merging user features with summaries, and preparing sequences for the LSTM model. It also ensures a consistent train/test split across all models for fair comparison.
*   **Model Implementations:** I implemented several models:
    *   **Logistic Regression:** A good baseline static model.
    *   **XGBoost:** A powerful gradient boosting model, also static.
    *   **LSTM Model:** A recurrent neural network designed for sequential data, capable of learning patterns over time in transaction histories.
*   **Model Evaluation:** A `ModelEvaluator` helps compare the performance of these different models using metrics like ROC curves, precision-recall curves, and confusion matrices.

### Remaining Tasks:

*   Ensure all necessary dependencies are installed (`scikit-learn`, `xgboost`, `tensorflow`, `seaborn`).
*   Run the `credit_risk_modeling.ipynb` notebook end-to-end to verify everything works as expected.
*   Debug any issues that arise during testing.
*   Analyze and interpret the results to understand the strengths and weaknesses of each modeling approach.

