import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "income": np.random.randint(200, 5000, n),
    "loan_amount": np.random.randint(100, 3000, n),
    "savings_balance": np.random.randint(0, 5000, n),
    "missed_payments": np.random.randint(0, 6, n),
    "loan_duration": np.random.randint(3, 24, n),
})

# Feature engineering logic for default
data["debt_to_income"] = data["loan_amount"] / (data["income"] + 1)
data["savings_ratio"] = data["savings_balance"] / (data["loan_amount"] + 1)

# Create realistic default label
data["default"] = (
    (data["debt_to_income"] > 0.6) |
    (data["missed_payments"] > 2) |
    (data["savings_ratio"] < 0.2)
).astype(int)

data.to_csv("data/raw.csv", index=False)

print("Dataset generated!")