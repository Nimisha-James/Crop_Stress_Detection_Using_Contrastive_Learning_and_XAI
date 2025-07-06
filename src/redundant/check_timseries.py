# check_timeseries.py
import os
import numpy as np
from glob import glob

SERIES_DIR = "../data/series/00"

invalid_files = []
for f in glob(f"{SERIES_DIR}/*.npy"):
    try:
        data = np.load(f)
        print(f"Checking {f}: shape={data.shape}, min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}, has_nan={np.isnan(data).any()}, has_inf={np.isinf(data).any()}")
        if np.isnan(data).any() or np.isinf(data).any():
            invalid_files.append(f)
    except Exception as e:
        print(f"Error loading {f}: {e}")
        invalid_files.append(f)

print(f"Found {len(invalid_files)} invalid time series files:")
for f in invalid_files[:10]:  # Print first 10 for brevity
    print(f"  {f}")