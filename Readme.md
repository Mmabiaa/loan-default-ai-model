# 🚀 CrediSense AI 

An advanced machine learning system for predicting loan default risk in microfinance institutions, tailored for African financial ecosystems. A scalable, explainable AI system for predicting loan default risk and optimizing lending decisions in microfinance environments like Susu Digital.

---

## 🧠 Overview

CrediSense AI is an advanced machine learning system designed to predict loan default risk in microfinance environments.

Built for platforms like Susu Digital, it enables:
- Data-driven lending decisions
- Risk-aware credit scoring
- Explainable AI insights

---

## 🎯 Key Features

- XGBoost model with hyperparameter tuning
- 1M+ synthetic financial dataset
- SHAP explainability (global + local)
- Risk threshold optimization
- REST API with FastAPI
- Production-ready architecture

---

## 📊 Model Capabilities

- Predict default probability
- Classify risk (Low / Medium / High)
- Optimize decision thresholds
- Explain predictions using SHAP

---

## 📊 Dataset

### Sources:
- Synthetic dataset (generated)

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
- SHAP

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

### 5. API Testing

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d @tests/test.json
```
---
## 📈 Risk Threshold Optimization

Instead of fixed 0.5 threshold:
- Model finds optimal threshold using Precision-Recall tradeoff
- Improves recall (catching defaulters)
- Reduces financial risk

---
### 🔍 Explainability (SHAP)

- Feature importance visualization
- Per-user explanation
- Transparent decision-making

---

## 📈Model Accuracy
Metrics used:
![](/images/matrix.png)

Accuracy & Precision
![](/images/Learning_Curve.png)

ROC-AUC (primary metric)
![](/images/ROC_curve.png)

### 🔮 Future Improvements

Deep learning (LSTM for time-series financial behavior)

Real-time streaming predictions

Integration with mobile money APIs

React dashboard for loan officers

### 🌍 Vision

To build an AI-powered credit scoring engine for Africa, enabling:

Financial inclusion

Risk reduction

Data-driven lending
---
### 🤝 Contribution Guide

- Fork the repo
- Create feature branch
- Commit changes
- Submit PR
---
### 👨‍💻 Author

Boateng Prince Agyenim (Mmabiaa)
Fullstack Engineer | AI Developer
Founder, Susu Digital