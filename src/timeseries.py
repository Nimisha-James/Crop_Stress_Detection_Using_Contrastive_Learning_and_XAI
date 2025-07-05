import os, numpy as np, glob
import pandas as pd
from collections import defaultdict

PATCH_DIR = "../data/patches"
DEST = "../data/series"
T = 6
STRIDE = 1
MONTHS     = ['jan','feb','mar','apr','may','jun']

os.makedirs(DEST, exist_ok=True)

series_count = 0
win = 0
for i in range(0, len(MONTHS)-T+1, STRIDE):
    target = MONTHS[i+T-1]
    window = MONTHS[i:i+T]
    patch_dict = defaultdict(list)

    win_dir = DEST + f"/{win:02d}"
    os.makedirs(win_dir, exist_ok=True)

    for m in window:
        for f in glob.glob(f"{PATCH_DIR}/{m}/*.npy"):
            pid = os.path.splitext(os.path.basename(f))[0]
            patch_dict[pid].append((m, f))

    for pid, files in patch_dict.items():
        if len(files) == T:
            files.sort(key=lambda x: window.index(x[0]))
            stack = np.stack([np.load(f[1]) for f in files], axis=0)
            out_path = f"{win_dir}/{win:02d}_{pid}.npy"
            np.save(out_path, stack)
            series_count += 1
    win+=1

print(f"{series_count} sliding-window time series saved to '{DEST}'")
