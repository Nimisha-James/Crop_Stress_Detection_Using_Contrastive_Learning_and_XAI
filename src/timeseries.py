# timeseries_multi.py
import os
import numpy as np
import glob
from collections import defaultdict

PATCH_DIR = "../data/patches"
DEST = "../data/series"
T = 6
STRIDE = 1
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun']

os.makedirs(DEST, exist_ok=True)
series_count = 0
win = 0

for i in range(0, len(MONTHS) - T + 1, STRIDE):
    window = MONTHS[i:i+T]
    patch_dict = defaultdict(list)
    win_dir = f"{DEST}/{win:02d}"
    os.makedirs(win_dir, exist_ok=True)

    for m in window:
        for f in glob.glob(f"{PATCH_DIR}/{m}/*.npy"):
            pid = os.path.splitext(os.path.basename(f))[0]
            patch_dict[pid].append((m, f))

    for pid, files in patch_dict.items():
        if len(files) == T:
            files.sort(key=lambda x: window.index(x[0]))
            # Load and validate each file
            stacks = []
            for m, f in files:
                data = np.load(f)
                if data.shape != (5, 64, 64) or np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    print(f"Warning: Invalid data in {f}, skipping patch {pid}")
                    break
                stacks.append(data)
            else:  # Only proceed if no break occurred
                stack = np.stack(stacks, axis=0)  # [T,5,64,64]
                np.save(f"{win_dir}/{win:02d}_{pid}.npy", stack)
                series_count += 1
    win += 1

print(f"{series_count} time series saved to '{DEST}'")