import os, numpy as np, glob
from collections import defaultdict

MONTHS = ['jan','feb','mar','apr','may','jun']
# MONTHS = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

DATA_DIR = '../data/patches/'
OUT_DIR = '../data/series/'
os.makedirs(OUT_DIR, exist_ok=True)

# har patch ID ke liye list of monthly patches
patch_dict = defaultdict(list)

for m in MONTHS:
    files = glob.glob(f"{DATA_DIR}/{m}/*.npy")
    for f in files:
        patch_id = os.path.basename(f).split('.')[0]  # e.g. "00001"
        patch = np.load(f)
        patch_dict[patch_id].append(patch)

# Ab sirf unhi patches ko save karo jo 12 months ke hain
count = 0
for patch_id, patches in patch_dict.items():
    if len(patches) == len(MONTHS):
        series = np.stack(patches, axis=0)  # [12, 64, 64]
        np.save(f"{OUT_DIR}/{patch_id}.npy", series)
        count += 1

print(f"{count} time series patches saved in '{OUT_DIR}'")
