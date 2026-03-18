from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("../models/model.pkl")

@app.get("/")
def home():
    return {"message": "Loan Default Prediction API"}

@app.post("/predict")
def predict(data: dict):
    values = list(data.values())
    input_array = np.array(values).reshape(1, -1)

    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    return {
        "default": int(prediction[0]),
        "risk_probability": float(probability[0][1])
    }