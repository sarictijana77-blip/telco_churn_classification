# Telco Customer Churn Classification

![Python](https://img.shields.io/badge/Python-3.14+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GitHub last commit](https://img.shields.io/github/last-commit/sarictijana77-blip/telco_churn_classification)

Machine learning project for predicting customer churn in a telecommunications network. The project trains multiple classification models, evaluates their performance, and provides an interactive web application for real-time churn predictions.

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave. This project implements a complete ML pipeline:

- **Data preprocessing** — cleaning, encoding, scaling, and stratified splitting
- **Model training** — multiple classifiers with hyperparameter tuning
- **Model evaluation** — comprehensive metrics and confusion matrices
- **Web application** — interactive Streamlit app for real-time predictions

## Project Structure

```
telco_churn_classification/
├── data/
│   ├── raw/                          Raw input data (telco_data.csv)
│   └── processed/                    Processed datasets (train.csv, val.csv, test.csv)
├── models/                           Trained ML models (.pkl)
├── results/
│   ├── figures/                      Generated plots (EDA, confusion matrices)
│   └── metrics/                      CSV evaluation reports
├── src/
│   ├── data_preparation.py           Data cleaning & preprocessing
│   ├── train.py                      Model training pipeline
│   └── evaluate.py                   Model evaluation & comparison
├── app.py                            Streamlit web application
├── requirements.txt                  Python dependencies
├── pyproject.toml                    Project metadata & configuration
└── README.md                         This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline Usage

Run each script sequentially:

### 1. Data Preparation

```bash
python src/data_preparation.py
```

Loads `data/raw/telco_data.csv`, cleans the data (removes `customerID`, converts `TotalCharges` to numeric, handles missing values), performs exploratory data analysis (churn distribution by contract type and correlation matrix), applies One-Hot Encoding to categorical features, splits the data into train/val/test (70/15/15) with stratification, scales numerical features using `StandardScaler`, and saves the results to `data/processed/`.

### 2. Model Training

```bash
python src/train.py
```

Loads `data/processed/train.csv` and trains three models:

- **Logistic Regression** — baseline model with cross-validation
- **Random Forest** — hyperparameter tuning via `GridSearchCV` (`n_estimators`, `max_depth`, `min_samples_split`)
- **Gradient Boosting** — hyperparameter tuning via `GridSearchCV` (`learning_rate`, `max_depth`)

Uses Gradient Boosting to determine the top 5 most important features and trains a reduced Gradient Boosting model using only those features. All models are saved to the `models/` directory.

### 3. Model Evaluation

```bash
python src/evaluate.py
```

Loads `data/processed/test.csv` and all models from the `models/` directory. Computes **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** for each model. Generates confusion matrices (saved to `results/figures/`) and a comparative CSV report (`results/metrics/final_model_comparison_report.csv`).

### 4. Web Application

```bash
streamlit run app.py
```

Launches a web application that loads the reduced Gradient Boosting model (trained on the top 5 features). Users input customer details (tenure, MonthlyCharges, TotalCharges, InternetService, PaymentMethod) through a form, and the app predicts whether the customer will churn or stay, along with a risk percentage.

## Quick Start

```bash
pip install -r requirements.txt
python src/data_preparation.py
python src/train.py
python src/evaluate.py
streamlit run app.py
```

## Technical Details

- **Python** 3.14+
- **scikit-learn** for ML models
- **Streamlit** for the web application
- **StandardScaler** for feature normalization
- Stratified train/val/test split (70/15/15)

## Results

Model performance metrics and comparison reports are generated automatically during the evaluation step and saved to `results/metrics/`. Confusion matrices are saved to `results/figures/`.

## License

This project is for educational purposes as part of a coursework assignment.
