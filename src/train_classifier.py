import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# Load actual dataset
print("Loading data...")
df = pd.read_csv("../data/csv_data/stress_cluster_labels.csv")
X = np.stack([np.load(f"../data/embeddings/00_{i:05d}.npy") for i in df["id"]])
y = df["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
lgbm = lgb.LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)

# Stacking model
stack_model = StackingClassifier(
    estimators=[('svm', svm), ('lgbm', lgbm)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=False,
    cv=5,
    n_jobs=-1
)

# Parameter grid
param_grid = {
    'svm__svc__C': [0.1, 1, 10],
    'lgbm__n_estimators': [50, 100],
    'lgbm__max_depth': [3, 5],
    'final_estimator__C': [0.1, 1, 10]
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=stack_model,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Train the model
print("Training model...")
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "../models/stress_classifier.pkl")
print("\n✅ Model saved to ../models/stress_classifier.pkl")
