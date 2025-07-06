# timeseries_multi.py
import os
import numpy as np
import glob
from collections import defaultdict

def generate_series(patch_dir, series_out_dir, months=None):
    os.makedirs(series_out_dir, exist_ok=True)
    patch_map = defaultdict(list)

    for i, month in enumerate(months):
        folder = os.path.join(patch_dir, month)
        for fpath in glob.glob(os.path.join(folder, "*.npy")):
            pid = os.path.splitext(os.path.basename(fpath))[0]
            patch_map[pid].append((month, fpath))

    count = 0
    for pid, entries in patch_map.items():
        month_order = {month: i for i, month in enumerate(months)}
        entries = sorted(entries, key=lambda x: month_order[x[0]])

        if len(entries) != 6:
            continue

        stack = []
        for month, fpath in entries:
            arr = np.load(fpath)
            if arr.shape != (5, 64, 64):
                break
            stack.append(arr)

        if len(stack) == 6:
            cube = np.stack(stack, axis=0)  # [6, 5, 64, 64]
            out_path = os.path.join(series_out_dir, f"{pid}.npy")
            np.save(out_path, cube)
            count += 1

    print(f"✅ Saved {count} series to {series_out_dir}")

