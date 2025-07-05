import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class NDVITimeSeriesDataset(Dataset):
    def __init__(self, series_dir):
        self.files = sorted(glob.glob(f"{series_dir}/**/*.npy", recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cube = np.load(self.files[idx])     # shape: [T, C, 64, 64]
        if cube.ndim == 3: cube = cube[:,None]
        cube = torch.tensor(cube, dtype=torch.float32)   # to torch tensor

        # Create two augmented versions
        view1 = self.augment(cube.clone())
        view2 = self.augment(cube.clone())

        return view1, view2

    def augment(self, x):
        # Add small random noise (jitter)
        x += torch.randn_like(x) * 0.02

        # Random time masking: set 1 month to NaN
        if torch.rand(1) < 0.5:x[torch.randint(0,x.size(0),(1,))] = torch.nan

        # (Optional) Normalization to [0, 1]
        x = (x + 1) / 2
        x = torch.nan_to_num(x, nan=0.0)  # Replace NaNs with 0

        return x