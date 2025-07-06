import numpy as np
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SERIES_DIR = "../data/series/00"
labels = []

# Step 1: Extract feature vectors for each patch
features = []
file_ids = []

for fname in sorted(os.listdir(SERIES_DIR)):
    path = os.path.join(SERIES_DIR, fname)
    if not fname.endswith(".npy"):
        continue
    cube = np.load(path)  # shape: [6, 5, 64, 64]

    # Flatten spatial dimensions and average to get a [6, 5] time series
    time_series = np.nanmean(cube, axis=(2, 3))  # shape: [6, 5]
    feature_vector = time_series.flatten()       # shape: [30]

    if np.isnan(feature_vector).any():
        continue  # skip patches with invalid values

    features.append(feature_vector)
    file_ids.append(int(fname.split('.')[0]))

# Step 2: Run IsolationForest
features = np.array(features)
iso = IsolationForest(contamination=0.15, random_state=42)
preds = iso.fit_predict(features)  # -1 for outliers, 1 for inliers

# Step 3: Map predictions to labels
for idx, pred in zip(file_ids, preds):
    label = 1 if pred == -1 else 0  # 1 = stressed (outlier), 0 = healthy
    labels.append((idx, label))

# Step 4: Save CSV
df = pd.DataFrame(labels, columns=["id", "label"])
df.to_csv("../data/csv_data/stress_cluster_labels.csv", index=False)
print("Saved labels to ../data/csv_data/stress_cluster_labels.csv")

# Step 5: Plot 2D PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
colors = ['green' if l == 0 else 'red' for l in df['label']]
plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7, edgecolor='k')
plt.title("PCA of Patch Time Series")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../outputs/label_pca_scatter.png")
plt.show()
