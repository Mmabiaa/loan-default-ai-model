from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Fix path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

# ✅ Define request schema
class LoanRequest(BaseModel):
    age: int
    income: float
    loan_amount: float
    savings_balance: float
    missed_payments: int
    loan_duration: int
    debt_to_income: float
    savings_ratio: float

@app.get("/")
def home():
    return {"message": "Loan Default Prediction API"}

@app.post("/predict")
def predict(data: LoanRequest):
    input_data = np.array([[
        data.age,
        data.income,
        data.loan_amount,
        data.savings_balance,
        data.missed_payments,
        data.loan_duration,
        data.debt_to_income,
        data.savings_ratio
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return {
        "default_prediction": int(prediction[0]),
        "risk_probability": float(probability[0][1]),
        "risk_level": (
            "HIGH" if probability[0][1] > 0.7 else
            "MEDIUM" if probability[0][1] > 0.4 else
            "LOW"
        )
    }