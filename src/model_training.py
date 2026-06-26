"""Train and compare multiple classification models for churn prediction."""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from data_preprocessing import preprocess_pipeline
from feature_engineering import scale_features
from evaluate import evaluate_model

warnings.filterwarnings("ignore")


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
    }


def tune_xgboost(X_train, y_train):
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb = XGBClassifier(
        n_estimators=300, use_label_encoder=False,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"Best XGBoost params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)

    X_train, X_test, y_train, y_test = preprocess_pipeline()
    X_train, X_test, scaler = scale_features(X_train.copy(), X_test.copy())

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"Training: {name}")
        print(f"{'─' * 40}")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics

    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER TUNING — XGBoost")
    print(f"{'=' * 60}")
    best_xgb = tune_xgboost(X_train, y_train)
    print(f"\n{'─' * 40}")
    print("Tuned XGBoost")
    print(f"{'─' * 40}")
    tuned_metrics = evaluate_model(best_xgb, X_test, y_test, "XGBoost (Tuned)")
    results["XGBoost (Tuned)"] = tuned_metrics

    os.makedirs("models", exist_ok=True)
    from data_preprocessing import PROJECT_ROOT
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_xgb, os.path.join(models_dir, "best_xgboost.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("\nBest model saved to models/best_xgboost.pkl")

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    summary = pd.DataFrame(results).T
    print(summary.to_string())


if __name__ == "__main__":
    main()
