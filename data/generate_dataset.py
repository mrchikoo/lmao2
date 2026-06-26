"""Generate a synthetic telecom customer churn dataset mimicking the Kaggle Telco dataset."""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043

gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
tenure = np.random.exponential(scale=32, size=N).clip(1, 72).astype(int)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
)

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
internet_deps = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

records = {
    "customerID": [f"C{str(i).zfill(5)}" for i in range(N)],
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
}

for feat in internet_deps:
    records[feat] = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], N, p=[0.40, 0.60])
    )

contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
paperless = np.random.choice(["Yes", "No"], N, p=[0.60, 0.40])
payment = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
monthly_charges = (base_charge + np.random.normal(0, 15, N)).clip(18, 118).round(2)
total_charges = (monthly_charges * tenure + np.random.normal(0, 200, N)).clip(18, 8700).round(2)

records.update({
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
})

churn_score = (
    -0.03 * tenure
    + 1.2 * (contract == "Month-to-month")
    + 0.5 * (internet_service == "Fiber optic")
    + 0.4 * (payment == "Electronic check")
    + 0.3 * (records["OnlineSecurity"] == "No")
    + 0.3 * (records["TechSupport"] == "No")
    + 0.01 * monthly_charges
    + 0.4 * senior_citizen
    - 0.3 * (partner == "Yes")
    + np.random.normal(0, 0.5, N)
)

churn_prob = 1 / (1 + np.exp(-churn_score))
churn = np.where(churn_prob > 0.82, "Yes", "No")
records["Churn"] = churn

df = pd.DataFrame(records)
df.to_csv("data/telco_churn.csv", index=False)
print(f"Dataset generated: {len(df)} rows, churn rate: {(churn == 'Yes').mean():.2%}")
