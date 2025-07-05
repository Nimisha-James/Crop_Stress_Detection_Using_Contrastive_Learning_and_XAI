import os, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load model
xgb = joblib.load("../models/xgb_classifier.pkl")
lgb = joblib.load("../models/lgb_classifier.pkl")

# --- Load Data ---
print("Loading labels and embeddings...")
df = pd.read_csv("../cluster_labels.csv")

X = np.stack([
    np.load(f"../data/embeddings/{i:05d}.npy")
    for i in df["id"]
])
y = df["label"].values

# --- Split Dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_pred_xgb = xgb.predict(X_test)
y_pred_lgb = lgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_lgb))