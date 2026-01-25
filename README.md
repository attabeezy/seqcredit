# Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments

This repository contains Jupyter notebooks and resources for a research project investigating the use of sequential deep learning for credit risk modeling, particularly in environments where data is limited.

## Project Overview

The primary goal of this project is to develop and evaluate sequential deep learning models for credit risk prediction. The models are designed to be effective even in data-constrained scenarios, making them suitable for applications where large datasets are not available. The project includes data preprocessing, model training, and evaluation workflows, as well as comparisons with traditional machine learning models like Logistic Regression and XGBoost.

## Getting Started

### Prerequisites

- Python 3.8 or higher with Jupyter Notebook support
- Or access to [Google Colab](https://colab.research.google.com/) (recommended for easy setup)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Attabeezy/sequential-crm-for-dce.git
   cd sequential-crm-for-dce
   ```

2. **Install required packages:**
   
   The notebooks will prompt you to install necessary packages when you run them. Common dependencies include:
   - pandas
   - numpy
   - scikit-learn
   - tensorflow/keras
   - xgboost
   - matplotlib
   - seaborn

   You can install them manually using:
   ```bash
   pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn
   ```

## Usage

This repository contains Jupyter notebooks that can be run locally or directly in Google Colab.

### Notebooks

1. **credit_risk_prediction_v1a.ipynb** - First version of the credit risk prediction workflow
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Attabeezy/sequential-crm-for-dce/blob/main/notebooks/credit_risk_prediction_v1a.ipynb)

2. **credit_risk_prediction_v1b.ipynb** - Enhanced second version with additional metrics
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Attabeezy/sequential-crm-for-dce/blob/main/notebooks/credit_risk_prediction_v1b.ipynb)

3. **credit_risk_prediction_v1c.ipynb** - Latest version of the credit risk prediction workflow
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Attabeezy/sequential-crm-for-dce/blob/main/notebooks/credit_risk_prediction_v1c.ipynb)

4. **syn_data_gen.ipynb** - Synthetic data generator for mobile money (MoMo) transaction data
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Attabeezy/sequential-crm-for-dce/blob/main/notebooks/syn_data_gen.ipynb)

5. **ctgan_syn_data_gen.ipynb** - CTGAN-based synthetic data generation
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Attabeezy/sequential-crm-for-dce/blob/main/notebooks/ctgan_syn_data_gen.ipynb)

### Running the Notebooks

**Option 1: Google Colab (Recommended)**
- Click on any of the "Open in Colab" badges above
- The notebook will open in Google Colab with all dependencies ready to install
- Follow the instructions within each notebook

**Option 2: Local Jupyter**
- Ensure you have Jupyter installed: `pip install jupyter`
- Launch Jupyter: `jupyter notebook`
- Navigate to and open any of the `.ipynb` files
- Run the cells sequentially

### What to Expect

When you run the credit risk prediction notebooks, they will perform the following steps:

1. **Load the dataset:** The notebooks load credit risk datasets (e.g., Lending Club loan data)
2. **Preprocess the data:** Includes cleaning, feature engineering, and splitting data into training/testing sets
3. **Train the models:** Three models are trained and compared:
   - Artificial Neural Network (ANN)
   - Logistic Regression
   - XGBoost
4. **Evaluate the models:** Performance metrics include:
   - Accuracy
   - Mean Squared Error (MSE)
   - Macro-F1 Score
   - Sensitivity/Precision
   - ROC AUC
5. **Generate visualizations:** ROC curves and other plots to visualize model performance
6. **Classification reports:** Detailed reports for each model

The synthetic data generator notebook creates realistic mobile money transaction data for testing and development purposes.

## Repository Structure

```
sequential-crm-for-dce/
├── data/
│   ├── real/                    # Real transaction data from Ghana
│   │   ├── transactions.xlsx - Table 1.csv
│   │   ├── transactions.xlsx - Table 5.csv
│   │   ├── engineered_features_real_data.csv
│   │   └── user_level_summary.csv
│   └── synthetic/               # Calibrated synthetic datasets
│       ├── synthetic_momo_calibrated.csv
│       ├── synthetic_user_profiles.csv
│       ├── synthetic-momo-data.csv
│       └── real_data_calibration.json
├── docs/
│   ├── COMPLETE_ANALYSIS.md     # Feature engineering framework documentation
│   └── SYNTHETIC_DATA_GUIDE.md  # Synthetic dataset documentation
├── notebooks/
│   ├── credit_risk_prediction_v1a.ipynb
│   ├── credit_risk_prediction_v1b.ipynb
│   ├── credit_risk_prediction_v1c.ipynb
│   ├── ctgan_syn_data_gen.ipynb
│   └── syn_data_gen.ipynb
├── src/
│   ├── data_generation/
│   │   └── calibrated_synthetic_generator.py
│   └── feature_engineering/
│       └── real_temporal_feature_engineering.py
├── .gitignore
├── LICENSE                      # MIT License
├── README.md                    # This file
└── SESSION_LOG.md               # Research session documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
