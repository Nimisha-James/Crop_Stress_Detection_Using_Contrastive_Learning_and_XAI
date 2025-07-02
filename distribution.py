# ndvi_distribution.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SERIES_DIR = Path("data/series")            # adjust if your folder is elsewhere
assert SERIES_DIR.is_dir(), f"{SERIES_DIR} not found"

means = []

# ── 1) load every cube ─────────────────────────────────────────────────────────
for npy_file in sorted(SERIES_DIR.glob("*.npy")):
    cube = np.load(npy_file)                # shape: [T,64,64]  (T = months)
    means.append(np.nanmean(cube))          # ignore NaN pixels

means = np.array(means)

if means.size == 0:
    raise RuntimeError("No .npy cubes found. Check SERIES_DIR path.")

# ── 2) quick numeric summary ───────────────────────────────────────────────────
summary = pd.Series(means).describe()
print("\nSummary of mean NDVI across patches:")
print(summary)

# ── 3) histogram plot ──────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.hist(means, bins=50, edgecolor="black")
plt.title("Distribution of mean NDVI values across patches")
plt.xlabel("Mean NDVI")
plt.ylabel("Number of patches")
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.show()
