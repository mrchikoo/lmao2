<div align="center">

# Telecom Customer Churn Prediction

**Exploratory analysis and model comparison for predicting which telecom customers will churn.**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

</div>

## Overview

An end-to-end churn project on a Telco-style dataset of 7,043 customers and 21 columns (demographics, account information, subscribed services, billing). It covers exploratory data analysis in a notebook, preprocessing, training and comparison of four classifiers, and hyperparameter tuning of XGBoost.

## Features

- **EDA notebook** (`notebooks/churn_analysis.ipynb`): churn distribution, churn by contract type, tenure group, monthly charges and key categorical features, and a feature correlation heatmap.
- **Preprocessing**: numeric coercion and median imputation of `TotalCharges`, binary and one-hot encoding, stratified 80/20 train/test split.
- **Model comparison**: Logistic Regression, Random Forest, Gradient Boosting and XGBoost.
- **Hyperparameter tuning**: `GridSearchCV` (3-fold, ROC-AUC) over XGBoost `max_depth`, `learning_rate`, `subsample` and `colsample_bytree`.
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, classification report and confusion matrix; top feature importances helper.
- **Artifacts**: tuned XGBoost model and fitted scaler saved to `models/`.

## Tech Stack

Python 3.9+, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib (SHAP is listed in `requirements.txt` but is not used by the current code).

## Getting Started

```bash
pip install -r requirements.txt
```

Run the full pipeline (preprocess, train all models, tune XGBoost, save the best model):

```bash
python src/model_training.py
```

Evaluation runs as part of that script; `src/evaluate.py` is a utility module, not a standalone script. To explore the data, open `notebooks/churn_analysis.ipynb` (launch Jupyter from the `notebooks/` directory so its relative imports resolve).

Outputs are written to `models/`: `best_xgboost.pkl` and `scaler.pkl` (git-ignored).

## Dataset

`data/telco_churn.csv` has 7,043 rows with the same schema as the [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset. The bundled file is **synthetic**, produced by `data/generate_dataset.py` (seed 42) to mimic that schema, so patterns and metrics reflect the generator's rules, not real customers. To regenerate it, run from the repository root:

```bash
python data/generate_dataset.py
```

You can also drop in the real Kaggle file at the same path.

## Results

Reference results from the author's run (test split). Exact values depend on the dataset and library versions; running the pipeline prints your own.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80 | 0.66 | 0.54 | 0.59 | 0.84 |
| Random Forest | 0.79 | 0.64 | 0.47 | 0.54 | 0.82 |
| Gradient Boosting | 0.81 | 0.68 | 0.53 | 0.60 | 0.85 |
| **XGBoost** | **0.82** | **0.69** | **0.56** | **0.62** | **0.86** |

## Key Findings

Observations recorded by the author from the analysis:

1. **Month-to-month contracts** have the highest churn rate (~42%) compared to one-year (~11%) and two-year (~3%) contracts.
2. **Tenure** is the strongest predictor; customers in their first year are most likely to churn.
3. **Fiber optic** users churn more, possibly due to pricing or service quality.
4. **Electronic check** payment correlates with higher churn.
5. Customers without **tech support** or **online security** are significantly more likely to churn.

## Project Structure

```
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator
│   └── telco_churn.csv
├── notebooks/
│   └── churn_analysis.ipynb     # EDA
├── src/
│   ├── data_preprocessing.py    # Loading, encoding, train/test split
│   ├── feature_engineering.py   # Derived features and scaling
│   ├── model_training.py        # Training, comparison, XGBoost tuning
│   └── evaluate.py              # Metrics and feature importance
├── models/                      # Created on first run (git-ignored artifacts)
├── requirements.txt
└── README.md
```
