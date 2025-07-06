# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultimodalTimeSeriesDataset(Dataset):
    def __init__(self, patch_root):
        self.root = patch_root
        files = sorted([f for f in os.listdir(self.root) if f.endswith(".npy")])
        if not files:
            raise FileNotFoundError(f"No .npy files found in: {self.root}")
        self.paths = [os.path.join(self.root, f) for f in files]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            x = np.load(path)  # [6, 5, 64, 64]
            #print(f"Loading {path}: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}, has_nan={np.isnan(x).any()}, has_inf={np.isinf(x).any()}")
            if np.isnan(x).any() or np.isinf(x).any():
                raise ValueError("Invalid values in input patch.")
            
            # Apply two augmentations for contrastive learning
            def augment(x):
                noise = np.random.normal(0, 0.005, x.shape).astype(np.float32)
                x_aug = x + noise
                #print(f"After augmentation: min={x_aug.min():.4f}, max={x_aug.max():.4f}, has_nan={np.isnan(x_aug).any()}, has_inf={np.isinf(x_aug).any()}")
                x_aug = np.clip(x_aug, 0.0, 1.0)
                #print(f"After clipping: min={x_aug.min():.4f}, max={x_aug.max():.4f}, has_nan={np.isnan(x_aug).any()}, has_inf={np.isinf(x_aug).any()}")
                return torch.tensor(x_aug, dtype=torch.float32)

            return augment(x), augment(x)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None