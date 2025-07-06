# create_patches_multi.py
import os
import rasterio
import numpy as np
import csv
import pandas as pd
from pathlib import Path

RAW_DIR = "../../data/multi_data"
PATCH_DIR = "../../data/patches"
SIZE = 64
STRIDE = 64
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun']

os.makedirs(PATCH_DIR, exist_ok=True)

unclean_csv_path = "../../data/csv_data/patch_unclean.csv"
clean_csv_path = "../../data/csv_data/patch_meta.csv"

# STEP 1: Create and write metadata file
with open(unclean_csv_path, "w", newline="") as meta:
    writer = csv.writer(meta)
    writer.writerow(["id", "month", "row", "col", "path", "note"])

    for m in MONTHS:
        tif_path = f"{RAW_DIR}/indices_{m}_2023.tif"
        out_dir = f"{PATCH_DIR}/{m}"
        os.makedirs(out_dir, exist_ok=True)

        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.float32)  # shape: [5, H, W]
            nodata = src.nodata
            if nodata is not None:
                img = np.where(img == nodata, np.nan, img)
            img[img < -0.5] = np.nan  # filter out extreme values

        _, H, W = img.shape
        pid = 0

        for y in range(0, H - SIZE + 1, STRIDE):
            for x in range(0, W - SIZE + 1, STRIDE):
                patch = img[:, y:y+SIZE, x:x+SIZE]
                if patch.shape != (5, SIZE, SIZE):
                    writer.writerow([pid, m, y, x, "", "shape mismatch"])
                    continue

                if np.isnan(patch).any():
                    patch = np.nan_to_num(patch, nan=0.0)
                    note = "nan replaced with 0"
                else:
                    note = "clean"

                patch_path = f"{out_dir}/{pid:05d}.npy"
                np.save(patch_path, patch)
                writer.writerow([pid, m, y, x, patch_path, note])
                pid += 1

        print(f"{m}: saved {pid} patches")

# STEP 2: Post-process CSV to drop 'note' column
df = pd.read_csv(unclean_csv_path)
if 'note' in df.columns:
    df.drop(columns=['note'], inplace=True)
df.to_csv(clean_csv_path, index=False)
print(f"✅ Metadata saved: {clean_csv_path}")

# STEP 3: Delete patch_unclean.csv
os.remove(unclean_csv_path)
#print(f"🗑️ Deleted unclean metadata file: {unclean_csv_path}")
