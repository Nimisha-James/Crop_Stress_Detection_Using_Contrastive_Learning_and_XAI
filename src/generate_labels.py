import numpy as np
import os
import pandas as pd

SERIES_DIR = "../data/series"
labels = []

for f in sorted(os.listdir(SERIES_DIR)):
    cube = np.load(os.path.join(SERIES_DIR, f))  # [T,64,64]
    curve = np.nanmean(cube, axis=(1,2))         # [T]
    
    peak = np.max(curve)
    final = curve[-1]
    mean = np.nanmean(curve)
    
    # Label as stressed if:
    if peak < 0.25 and final < 0.17 and mean < 0.18:
        label = 1  # stressed
    else:
        label = 0  # healthy


    patch_id = int(f.split('.')[0])
    labels.append((patch_id, label))

df = pd.DataFrame(labels, columns=["id", "label"])
df.to_csv("../cluster_labels.csv", index=False)
