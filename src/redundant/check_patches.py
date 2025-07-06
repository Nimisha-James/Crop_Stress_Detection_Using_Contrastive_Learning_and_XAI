import os
import numpy as np
from glob import glob

PATCH_DIR = ".../data/patches"
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun']

for month in MONTHS:
    invalid_files = []
    for f in glob(f"{PATCH_DIR}/{month}/*.npy"):
        patch = np.load(f)
        if np.isnan(patch).any() or np.isinf(patch).any():
            invalid_files.append(f)
    print(f"{month}: {len(invalid_files)} invalid patches found")
    for f in invalid_files[:5]:  # Print first 5 for brevity
        print(f"  {f}")