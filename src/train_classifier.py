import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import xgboost as xgb
import lightgbm as lgb

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



# --- XGBoost Training ---
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=(len(y_train) / np.sum(y_train) - 1),  # imbalance handling
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print("\n=== XGBoost Results ===")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb)) 

os.makedirs("outputs", exist_ok=True)
joblib.dump(xgb_model, "../models/xgb_classifier.pkl")
print("\nModel saved to models/xgb_classifier.pkl")



# --- LightGBM Training ---
print("\nTraining LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    is_unbalance=True,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
print("\n=== LightGBM Results ===")
print(classification_report(y_test, y_pred_lgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgb))

os.makedirs("outputs", exist_ok=True)
joblib.dump(lgb_model, "../models/lgb_classifier.pkl")
print("\nModel saved to modelss/lgb_classifier.pkl")


'''
--- Plot Feature Importance ---
ig, axs = plt.subplots(1, 2, figsize=(14, 5))

# --- XGBoost plot
xgb.plot_importance(xgb_model, ax=axs[0], max_num_features=10)
axs[0].set_title("XGBoost Feature Importance")

# --- LightGBM plot
lgb.plot_importance(lgb_model, ax=axs[1], max_num_features=10)
axs[1].set_title("LightGBM Feature Importance")

plt.tight_layout()
plt.show()'''