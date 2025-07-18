import os
import rasterio
import numpy as np
import csv
import pandas as pd

def extract_patches_from_folder(input_folder, output_folder, patch_size=64, stride=64):
    os.makedirs(output_folder, exist_ok=True)
    
    months = []
    for file in os.listdir(input_folder):
        if file.endswith(".tif") and "multi_" in file:
            month = file.split("_")[1].lower()
            months.append(month)
    if not months:
        raise ValueError("No valid monthly .tif files found in the input folder.")

    unclean_csv_path = os.path.join(output_folder, "patch_unclean.csv")
    clean_csv_path = os.path.join(output_folder, "patch_meta.csv")

    with open(unclean_csv_path, "w", newline="") as meta:
        writer = csv.writer(meta)
        writer.writerow(["id", "month", "row", "col", "path", "note"])

        for m in months:
            tif_path = os.path.join(input_folder, f"multi_{m}_2024.tif")
            out_dir = os.path.join(output_folder, m)
            os.makedirs(out_dir, exist_ok=True)

            with rasterio.open(tif_path) as src:
                img = src.read().astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    img = np.where(img == nodata, np.nan, img)
                img[img < -0.5] = np.nan

            _, H, W = img.shape
            pid = 0

            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    patch = img[:, y:y+patch_size, x:x+patch_size]
                    if patch.shape != (5, patch_size, patch_size):
                        writer.writerow([pid, m, y, x, "", "shape mismatch"])
                        continue

                    if np.isnan(patch).any():
                        patch = np.nan_to_num(patch, nan=0.0)
                        note = "nan replaced with 0"
                    else:
                        note = "clean"

                    patch_path = os.path.join(out_dir, f"{pid:05d}.npy")
                    np.save(patch_path, patch)
                    writer.writerow([pid, m, y, x, patch_path, note])
                    pid += 1

            print(f"{m}: saved {pid} patches")

    df = pd.read_csv(unclean_csv_path)
    if 'note' in df.columns:
        df.drop(columns=['note'], inplace=True)
    df.to_csv(clean_csv_path, index=False)
    os.remove(unclean_csv_path)