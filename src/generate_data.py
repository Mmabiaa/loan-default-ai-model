import pandas as pd
import numpy as np

np.random.seed(42)

n = 1_000_000  # 1 million rows

# Generate base features
age = np.random.randint(18, 65, n)
income = np.random.randint(200, 15000, n)
loan_amount = np.random.randint(100, 10000, n)
savings_balance = np.random.randint(0, 20000, n)
missed_payments = np.random.randint(0, 10, n)
loan_duration = np.random.randint(3, 48, n)

# Feature engineering
debt_to_income = loan_amount / (income + 1)
savings_ratio = savings_balance / (loan_amount + 1)
payment_stress = missed_payments / (loan_duration + 1)

# Default logic (vectorized for speed)
default = (
    (debt_to_income > 0.75) |
    (savings_ratio < 0.1) |
    (payment_stress > 0.35)
).astype(int)

# Build DataFrame
data = pd.DataFrame({
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "savings_balance": savings_balance,
    "missed_payments": missed_payments,
    "loan_duration": loan_duration,
    "debt_to_income": debt_to_income,
    "savings_ratio": savings_ratio,
    "payment_stress": payment_stress,
    "default": default
})

# Save efficiently
data.to_csv("data/raw.csv", index=False)

print("✅ 1,000,000-row dataset generated!")