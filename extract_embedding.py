import os
import torch
import numpy as np
from model import Encoder3D
from dataset import NDVITimeSeriesDataset
from torch.utils.data import DataLoader

# Load model
model = Encoder3D()
model.load_state_dict(torch.load("encoder_simclr.pt"))
model.eval()
model.cuda()

# Load data
dataset = NDVITimeSeriesDataset("data/series")
loader = DataLoader(dataset, batch_size=64, shuffle=False)

os.makedirs("embeddings", exist_ok=True)

with torch.no_grad():
    for i, (view1, _) in enumerate(loader):
        view1 = view1.cuda()
        embeddings = model(view1).cpu().numpy()  # shape: [B, 128]
        for j, emb in enumerate(embeddings):
            idx = i * loader.batch_size + j
            np.save(f"embeddings/{idx:05d}.npy", emb)
