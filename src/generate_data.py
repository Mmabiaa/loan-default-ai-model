import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000  # Increased size

data = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "income": np.random.randint(200, 10000, n),
    "loan_amount": np.random.randint(100, 5000, n),
    "savings_balance": np.random.randint(0, 10000, n),
    "missed_payments": np.random.randint(0, 8, n),
    "loan_duration": np.random.randint(3, 36, n),
})

# Advanced feature engineering
data["debt_to_income"] = data["loan_amount"] / (data["income"] + 1)
data["savings_ratio"] = data["savings_balance"] / (data["loan_amount"] + 1)
data["payment_stress"] = data["missed_payments"] / (data["loan_duration"] + 1)

# More realistic default logic
data["default"] = (
    (data["debt_to_income"] > 0.7) |
    (data["savings_ratio"] < 0.15) |
    (data["payment_stress"] > 0.3)
).astype(int)

data.to_csv("data/raw.csv", index=False)

print("✅ Large dataset generated!")