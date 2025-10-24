# Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments

This repository contains the code and resources for a research project investigating the use of sequential deep learning for credit risk modeling, particularly in environments where data is limited.

## Project Overview

The primary goal of this project is to develop and evaluate a sequential deep learning model for credit risk prediction. The model is designed to be effective even in data-constrained scenarios, making it suitable for applications where large datasets are not available. The project includes data preprocessing, model training, and evaluation scripts, as well as a comparison with traditional machine learning models like Logistic Regression and XGBoost.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` for package management

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Attabeezy/sequential-crm-for-dce.git
   cd sequential-crm-for-dce
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The repository includes two main scripts, `main_v1.py` and `main_v1_1.py`, which represent two versions of the credit risk prediction workflow.

### Running the Scripts

To run either version of the model, execute the corresponding Python script from the command line:

```bash
# To run the first version of the model
python main_v1.py

# To run the second version of the model
python main_v1_1.py
```

### What to Expect

When you run either script, it will perform the following steps:

1. **Load the dataset:** The script will load the `lending_club_loan_two.csv` dataset.
2. **Preprocess the data:** This includes cleaning, feature engineering, and splitting the data into training and testing sets.
3. **Train the models:** The script will train three models: an Artificial Neural Network (ANN), a Logistic Regression model, and an XGBoost model.
4. **Evaluate the models:** The performance of each model will be evaluated using metrics such as accuracy and Mean Squared Error (MSE).
5. **Generate a classification report:** A detailed classification report will be printed for each model.

The script will also generate and display a Receiver Operating Characteristic (ROC) curve to visualize the performance of the models.

## Repository Structure

- **main_v1.py:** The main script for the first version of the credit risk prediction workflow.
- **main_v1_1.py:** The main script for the second version of the credit risk prediction workflow.
- **requirements.txt:** A list of all the Python packages required to run the code.
- **LICENSE:** The license for this project.
- **README.md:** This file, providing an overview of the project and instructions on how to use it.
