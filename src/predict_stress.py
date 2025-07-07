import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import joblib

# Ensure training directory in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

from training.model import Encoder3D
from testing.generate_patches_test import extract_patches_from_folder
from testing.timeseries_test import generate_series

# Suppress scikit-learn and joblib warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# CONFIG
MONTHS = ['mar', 'apr', 'may', 'oct', 'nov', 'dec']
RAW_DATA_DIR = "../new_data/2024_multi_data"
PATCH_DIR = "../new_data/new_patches"
SERIES_DIR = "../new_data/new_series/00"
CLASSIFIER_PATH = "../models/stress_classifier.pkl"
ENCODER_PATH = "../models/encoder_simclr.pt"
OUTPUT_PRED_DIR = "../new_data/output_predictions"
os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)

# Step 1: Preprocessing
os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(SERIES_DIR, exist_ok=True)

extract_patches_from_folder(input_folder=RAW_DATA_DIR, output_folder=PATCH_DIR, patch_size=64, stride=64)
generate_series(patch_dir=PATCH_DIR, series_out_dir=SERIES_DIR, months=MONTHS)

# Step 2: Load Encoder & Classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder3D(in_ch=5).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()

clf = joblib.load(CLASSIFIER_PATH)
# Handle StackingClassifier
model = clf.named_steps['stackingclassifier'] if isinstance(clf, Pipeline) and 'stackingclassifier' in clf.named_steps else clf

# Step 3: Predict for all patches
patch_files = [f for f in os.listdir(SERIES_DIR) if f.endswith(".npy")]
print(f"Found {len(patch_files)} .npy files in {SERIES_DIR}: {patch_files[:5]}")
final_results = {}

for month_idx, month in enumerate(MONTHS):
    preds, probs, embs = [], [], []

    for path in tqdm(patch_files, desc=f"[{month}]"):
        patch = np.load(os.path.join(SERIES_DIR, path))
        if patch.shape != (6, 5, 64, 64):
            print(f"Skipping {path}: Expected shape (6, 5, 64, 64), got {patch.shape}")
            continue
        # Extract single month's patch
        month_patch = patch[month_idx]  # Shape: [5, 64, 64]
        # Reshape to [batch_size, T=1, C=5, H=64, W=64]
        tensor = torch.tensor(month_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, 5, 64, 64]
        print(f"Tensor shape for {path}: {tensor.shape}")  # Debug
        with torch.no_grad():
            emb = encoder(tensor).cpu().numpy().flatten()  # Shape: [128]
        pred = clf.predict([emb])[0]
        prob = clf.predict_proba([emb])[0][1]
        embs.append(emb)
        preds.append(pred)
        probs.append(prob)

    if len(preds) == 0:
        final_results[month] = {"label": "Missing", "stressed_pct": 0, "avg_prob": 0, "total_patches": 0}
        continue

    stressed_pct = np.mean(preds)
    final_label = "Stressed" if stressed_pct >= 0.5 else "Healthy"

    # Save SHAP input
    np.save(os.path.join(OUTPUT_PRED_DIR, f"{month}_input.npy"), np.array(embs))
    np.save(os.path.join(OUTPUT_PRED_DIR, f"{month}_preds.npy"), np.array(preds))
    np.save(os.path.join(OUTPUT_PRED_DIR, f"{month}_probs.npy"), np.array(probs))

    final_results[month] = {
        "label": final_label,
        "stressed_pct": round(stressed_pct * 100, 2),
        "avg_prob": round(np.mean(probs), 3),
        "total_patches": len(preds)
    }

# Step 4: Print results
print("\n🧾 Monthly Stress Summary:")
for month, info in final_results.items():
    print(f"{month.capitalize()}: {info['label']} "
          f"(Stressed: {info['stressed_pct']}%, Avg Prob: {info['avg_prob']}, Patches: {info['total_patches']})")