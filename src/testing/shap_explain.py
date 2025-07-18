import os
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import warnings

# Suppress NumPy RNG, scikit-learn, and joblib warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=Warning, module="sklearn")

# CONFIG
MONTHS = ['mar', 'apr', 'may', 'oct', 'nov', 'dec']
CLASSIFIER_PATH = "../../models/stress_classifier.pkl"
PREDICTION_DIR = "../../new_data/output_predictions"
EXPLAIN_DIR = "../../outputs/shap_explanations_2"
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# Load classifier
clf = joblib.load(CLASSIFIER_PATH)
# Handle StackingClassifier
model = clf.named_steps['stackingclassifier'] if isinstance(clf, Pipeline) and 'stackingclassifier' in clf.named_steps else clf

# SHAP explainer
feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
# Use KernelExplainer with a small background dataset
background = np.zeros((1, model.n_features_in_))  # Minimal background for efficiency
explainer = shap.KernelExplainer(model.predict_proba, background)

for month in MONTHS:
    input_path = os.path.join(PREDICTION_DIR, f"{month}_input.npy")
    preds_path = os.path.join(PREDICTION_DIR, f"{month}_preds.npy")
    probs_path = os.path.join(PREDICTION_DIR, f"{month}_probs.npy")

    if not os.path.exists(input_path):
        print(f"[WARN] Missing data for {month}, skipping...")
        continue

    X = np.load(input_path)
    preds = np.load(preds_path)
    probs = np.load(probs_path)

    # Validate input data
    if X.shape[0] != len(preds) or X.shape[0] != len(probs) or np.any(np.isnan(probs)):
        print(f"[WARN] Invalid data for {month}: X shape={X.shape}, preds len={len(preds)}, probs len={len(probs)}, nan in probs={np.any(np.isnan(probs))}")
        continue

    print(f"Processing {month}: X shape={X.shape}, preds shape={preds.shape}, probs shape={probs.shape}, max prob={np.max(probs)}")

    # Get SHAP values for positive class (index 1)
    try:
        shap_values = explainer.shap_values(X, nsamples=200)[:, :, 1]  # Increased nsamples for better accuracy
    except Exception as e:
        print(f"[ERROR] Failed to compute SHAP values for {month}: {e}")
        continue

    # Save SHAP summary plot for top 10 features
    summary_path = os.path.join(EXPLAIN_DIR, f"{month}_summary.png")
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=10, show=False)
        plt.title(f"SHAP Summary - {month.capitalize()}")
        plt.savefig(summary_path, bbox_inches='tight')
        plt.clf()
        print(f"✅ Saved summary plot for {month}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary plot for {month}: {e}")
        continue

    # Force plot for the most stressed patch
    try:
        top_idx = np.argsort(probs)[-1:]  # Select only the most stressed patch
        if len(top_idx) == 0 or probs[top_idx[0]] <= 0:
            print(f"[WARN] No valid most stressed patch for {month}, max prob={np.max(probs)}")
            continue
        idx = top_idx[0]
        force_path = os.path.join(EXPLAIN_DIR, f"{month}_force.png")  # Single force plot per month
        plt.figure(figsize=(12, 6))  # Increase figure size for better spacing
        shap.plots.force(explainer.expected_value[1], shap_values[idx], features=X[idx], feature_names=feature_names, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot - Most Stressed Patch ({month.capitalize()}, Prob={probs[idx]:.3f})")
        plt.xticks(fontsize=8)  # Reduce font size of x-axis labels
        plt.yticks(fontsize=8)  # Reduce font size of y-axis labels
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(force_path, bbox_inches='tight', dpi=100)
        plt.clf()
        print(f"✅ Saved force plot for {month}, index={idx}, prob={probs[idx]:.3f}")
    except Exception as e:
        print(f"[ERROR] Failed to save force plot for {month}: {e}")
        continue

    print(f"✅ SHAP explanations saved for {month}")

print("\n✅ All SHAP plots saved in:", EXPLAIN_DIR)