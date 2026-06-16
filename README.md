# Telco Customer Churn Classification

![Python](https://img.shields.io/badge/Python-3.14+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![GitHub last commit](https://img.shields.io/github/last-commit/sarictijana77-blip/telco_churn_classification)

Machine learning project for predicting customer churn in a telecommunications network. The project trains multiple classification models (6 algorithms + 1 reduced model), evaluates their performance, and provides an interactive web application for real-time churn predictions.

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave. This project implements a complete ML pipeline:

- **Data preprocessing** — cleaning, encoding, scaling, stratified splitting, and EDA visualization
- **Model training** — 6 classifiers with hyperparameter tuning via **GridSearchCV**, **SMOTE** oversampling for class imbalance
- **Model evaluation** — comprehensive metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) and confusion matrices
- **Web application** — interactive Streamlit app for real-time churn predictions using the reduced Gradient Boosting model

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

Loads `data/raw/telco_data.csv` and performs the following:

- **Drops `customerID`** — irrelevant for modeling
- **Drops `gender`** — removed as a non-predictive feature
- **Removes duplicate rows** — ensures data integrity
- **Cleans `TotalCharges`** — converts from string to numeric, replaces empty strings with NaN, fills missing values with median
- **Exploratory Data Analysis (EDA)**:
  - Count plot: churn distribution by contract type (`eda_churn_by_contract.png`)
  - Correlation heatmap: tenure, MonthlyCharges, TotalCharges (`eda_correlation_matrix.png`)
- **Label Encoding** — encodes the target column `Churn` (Yes/No → 1/0)
- **One-Hot Encoding** — converts categorical features, uses `drop_first=True` to avoid dummy variable trap
- **Stratified split** — train/val/test = 70/15/15 with stratification on target
- **StandardScaler** — fits on training set only, transforms all three sets (with `.copy()` to avoid `SettingWithCopyWarning`)
- **Saves** processed datasets to `data/processed/` (train.csv, val.csv, test.csv)

### 2. Model Training

```bash
python src/train.py
```

Loads `data/processed/train.csv` and trains **6 models** using **SMOTE** (Synthetic Minority Oversampling Technique) to handle class imbalance, all with **GridSearchCV** for hyperparameter tuning:

| # | Model | Tuning Details |
|---|-------|---------------|
| 1 | **Logistic Regression** | `C`, `solver`, `penalty` — tuned via GridSearchCV with 5-fold CV, F1 scoring |
| 2 | **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` — GridSearchCV |
| 3 | **Gradient Boosting** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` — GridSearchCV |
| 4 | **K-Nearest Neighbors (KNN)** | Elbow method (K=1..50, both `distance` and `uniform` weights) on validation set, followed by GridSearchCV around optimal K + metric search (`euclidean`, `manhattan`, `minkowski`) |
| 5 | **Support Vector Machine (SVM)** | `C`, `gamma`, `kernel` (`rbf`, `poly`) — GridSearchCV with `CalibratedClassifierCV` for probability estimates |
| 6 | **XGBoost** | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` — GridSearchCV with `scale_pos_weight` for imbalance |

All models are trained via an `ImbPipeline` (SMOTE + classifier) and saved to the `models/` directory.

**Feature Selection:**

After training, the best Gradient Boosting model is used to extract the **top 5 most important features**. These features are saved (`top_features.pkl`) and a **reduced Gradient Boosting model** is trained using only those features, which is then used in the web application for faster inference.

### 3. Model Evaluation

```bash
python src/evaluate.py
```

Loads `data/processed/test.csv` and **all 7 models** from the `models/` directory (6 full + 1 reduced). Computes:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Generates high-resolution (300 dpi) confusion matrices for each model (saved to `results/figures/`) and a comparative CSV report (`results/metrics/final_model_comparison_report.csv`).

### 4. Web Application

```bash
streamlit run app.py
```

Launches a web application that loads the **reduced Gradient Boosting model** (trained on the top 5 features). Users input customer details (tenure, MonthlyCharges, TotalCharges, InternetService, PaymentMethod) through a form, and the app predicts whether the customer will churn or stay, along with a risk percentage.

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
- **SMOTE (imbalanced-learn)** for handling class imbalance
- **XGBoost** for gradient boosting
- **StandardScaler** for feature normalization
- Stratified train/val/test split (70/15/15)
- **GridSearchCV** with 5-fold cross-validation for hyperparameter tuning
- All models trained with `random_state=42` for reproducibility

## Results

Model performance metrics and comparison reports are generated automatically during the evaluation step and saved to `results/metrics/`. Confusion matrices are saved to `results/figures/`.

## License

This project is for educational purposes as part of a coursework assignment.
