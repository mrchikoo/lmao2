"""Data loading and preprocessing for the churn prediction pipeline."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


def encode_features(df):
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df


def split_data(df, target="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def preprocess_pipeline(filepath=None):
    df = load_data(filepath)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Churn rate — Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_pipeline()
