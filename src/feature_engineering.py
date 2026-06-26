"""Feature engineering utilities for the churn prediction pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_derived_features(df):
    df = df.copy()

    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["ChargeIncrease"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
    ).astype(int)

    num_services = 0
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        if col in df.columns:
            num_services += (df[col] == "Yes").astype(int) if df[col].dtype == object else df[col]

    df["NumServices"] = num_services

    return df


def scale_features(X_train, X_test, cols_to_scale=None):
    if cols_to_scale is None:
        cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharge", "ChargeIncrease"]

    cols_present = [c for c in cols_to_scale if c in X_train.columns]
    scaler = StandardScaler()
    X_train[cols_present] = scaler.fit_transform(X_train[cols_present])
    X_test[cols_present] = scaler.transform(X_test[cols_present])
    return X_train, X_test, scaler
