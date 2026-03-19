import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier
import shap

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/raw.csv")

X = df.drop("default", axis=1)
y = df["default"]

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL + TUNING
# -------------------------
xgb = XGBClassifier(
    eval_metric='logloss',
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    xgb,
    param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

print("🔥 Best Params:", grid.best_params_)

# -------------------------
# EVALUATION
# -------------------------
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_probs))

# -------------------------
# RISK THRESHOLD OPTIMIZATION
# -------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Choose threshold where recall is high but precision still decent
optimal_idx = np.argmax(recall - (1 - precision))
optimal_threshold = thresholds[optimal_idx]

print(f"✅ Optimal Risk Threshold: {optimal_threshold:.2f}")

# -------------------------
# ROC CURVE
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# -------------------------
# SHAP (Sampled)
# -------------------------
sample = X_train.sample(2000, random_state=42)

explainer = shap.Explainer(model)
shap_values = explainer(sample)

shap.summary_plot(shap_values, sample)
shap.plots.bar(shap_values)

# -------------------------
# SAVE EVERYTHING
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(optimal_threshold, "models/threshold.pkl")

print("✅ Model + threshold saved!")