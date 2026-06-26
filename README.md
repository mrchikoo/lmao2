# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn for a telecom company using classification models. The project covers exploratory data analysis, feature engineering, model training & comparison, hyperparameter tuning, and model interpretability.

## Project Highlights

- **EDA & Visualization**: Comprehensive analysis of churn patterns across demographics, services, and billing
- **Feature Engineering**: Encoding, scaling, and creating derived features from raw data
- **Model Comparison**: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Model Interpretability**: SHAP values and feature importance analysis
- **Evaluation**: Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices

## Dataset

The project uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle containing 7,043 customer records with 21 features including demographics, account information, and service subscriptions.

## Project Structure

```
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluate.py
├── models/
│   └── (saved model artifacts)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline
```bash
python src/model_training.py
```

### Run evaluation on a trained model
```bash
python src/evaluate.py
```

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80 | 0.66 | 0.54 | 0.59 | 0.84 |
| Random Forest | 0.79 | 0.64 | 0.47 | 0.54 | 0.82 |
| Gradient Boosting | 0.81 | 0.68 | 0.53 | 0.60 | 0.85 |
| **XGBoost** | **0.82** | **0.69** | **0.56** | **0.62** | **0.86** |

## Key Findings

1. **Month-to-month contracts** have the highest churn rate (~42%) compared to one/two-year contracts (~11%/~3%)
2. **Tenure** is the strongest predictor — customers in their first year are most likely to churn
3. **Fiber optic internet** users churn more, likely due to pricing or service quality issues
4. **Electronic check** payment method correlates with higher churn
5. Customers without **tech support** or **online security** are significantly more likely to churn

## Tech Stack

- Python 3.9+
- pandas, NumPy — data manipulation
- scikit-learn — modeling & evaluation
- XGBoost — gradient boosting
- SHAP — model interpretability
- Matplotlib, Seaborn — visualization
