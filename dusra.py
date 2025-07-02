import os
import numpy as np
import pandas as pd

SERIES_DIR = "data/series"
THRESHOLD = 0.216  # NDVI threshold for stress

labels = []

files = sorted([f for f in os.listdir(SERIES_DIR) if f.endswith(".npy")])
for f in files:
    cube = np.load(os.path.join(SERIES_DIR, f))  # shape: [T,64,64]
    # print(f"{np.min(cube)} , {np.max(cube)}")
    mean_ndvi = np.nanmean(cube)                 # mean over time and space
    label = int(mean_ndvi < THRESHOLD)          # 1 if stressed
    patch_id = int(f.split(".")[0])
    labels.append((patch_id, label))

# Save to CSV
df = pd.DataFrame(labels, columns=["id", "label"])
df.to_csv("cluster_labels.csv", index=False)

print(f"Saved labels to cluster_labels.csv")
print(df["label"].value_counts())
