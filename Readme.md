# 🚀 Loan Default Prediction AI (Susu Digital)

An advanced machine learning system for predicting loan default risk in microfinance institutions, tailored for African financial ecosystems.

---

## 🧠 Overview

This project builds a **production-ready credit risk scoring system** using machine learning.

It is designed to power **Susu Digital**, enabling:
- Intelligent loan approvals
- Risk-based decision making
- Fraud and default prevention

---

## 🎯 Key Features

- 🔥 XGBoost-based classification model
- 📊 Hyperparameter tuning with GridSearchCV
- 📈 ROC-AUC performance optimization
- 📉 Learning curves & model evaluation
- 🧪 API testing via JSON
- 🌍 Synthetic Ghanaian financial dataset
- 🚀 FastAPI deployment-ready backend

---

## 🏗️ Architecture
```
Data → Feature Engineering → Model Training → Evaluation → API → Client App
```


---

## 📊 Dataset

### Sources:
- Synthetic dataset (generated)
- Kaggle: "Give Me Some Credit"

### Features:
- Age
- Income
- Loan Amount
- Savings Balance
- Missed Payments
- Loan Duration
- Debt-to-Income Ratio
- Savings Ratio
- Payment Stress

---

## ⚙️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Matplotlib

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Generate dataset

```bash
python src/generate_data.py
```
### 3. Train model

```bash
python src/train.py
```
### 4. Run API

```bash
cd api
uvicorn main:app --reload
```

### 5. Test API

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d @tests/test.json
```

## 📈Model Accuracy
Metrics used:
![](/images/matrix.png)

Accuracy & Precision
![](/images/Learning_Curve.png)

ROC-AUC (primary metric)
![](/images/ROC_curve.png)

### 🧠 Why XGBoost?

XGBoost is a state-of-the-art gradient boosting algorithm widely used in:

Banking

Credit scoring

Fraud detection

It provides:

High accuracy

Robust performance on tabular data

Feature importance insights

### 🔮 Future Improvements

Deep learning (LSTM for time-series financial behavior)

Real-time streaming predictions

Integration with mobile money APIs

Explainable AI (SHAP values)

React dashboard for loan officers

### 🌍 Vision

To build an AI-powered credit scoring engine for Africa, enabling:

Financial inclusion

Risk reduction

Data-driven lending

### 👨‍💻 Author

Boateng Prince Agyenim (Mmabiaa)
Fullstack Engineer | AI Developer
Founder, Susu Digital