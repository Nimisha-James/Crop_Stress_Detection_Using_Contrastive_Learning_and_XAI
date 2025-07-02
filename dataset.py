import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NDVITimeSeriesDataset(Dataset):
    def __init__(self, series_dir):
        self.files = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cube = np.load(self.files[idx])     # shape: [12, 64, 64]
        cube = torch.tensor(cube, dtype=torch.float32)   # to torch tensor

        # Create two augmented versions
        view1 = self.augment(cube.clone())
        view2 = self.augment(cube.clone())

        return view1, view2

    def augment(self, x):
        # Add small random noise (jitter)
        x += torch.randn_like(x) * 0.02

        # Random time masking: set 1 month to NaN
        if torch.rand(1).item() < 0.5:
            t = torch.randint(0, x.shape[0], (1,))
            x[t] = torch.nan

        # (Optional) Normalization to [0, 1]
        x = (x + 1) / 2
        x = torch.nan_to_num(x, nan=0.0)  # Replace NaNs with 0

        return x