from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import numpy as np

app = FastAPI(title="CrediSense AI")

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "models", "threshold.pkl"))

# -------------------------
# REQUEST MODEL
# -------------------------
class LoanRequest(BaseModel):
    age: int
    income: float
    loan_amount: float
    savings_balance: float
    missed_payments: int
    loan_duration: int
    debt_to_income: float
    savings_ratio: float
    payment_stress: float

# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def home():
    return {"message": "CrediSense AI - Loan Risk API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: LoanRequest):
    features = np.array([[
        data.age,
        data.income,
        data.loan_amount,
        data.savings_balance,
        data.missed_payments,
        data.loan_duration,
        data.debt_to_income,
        data.savings_ratio,
        data.payment_stress
    ]])

    prob = model.predict_proba(features)[0][1]
    pred = int(prob > threshold)

    return {
        "default_prediction": pred,
        "risk_probability": float(prob),
        "risk_level": (
            "HIGH" if prob > 0.75 else
            "MEDIUM" if prob > 0.4 else
            "LOW"
        ),
        "threshold_used": float(threshold)
    }