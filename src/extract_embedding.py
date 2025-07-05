import os
import torch
import numpy as np
from model import Encoder3D
from dataset import NDVITimeSeriesDataset
from torch.utils.data import DataLoader

# Device config (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
C = 1
model = Encoder3D(in_ch=C)
model.load_state_dict(torch.load("../models/encoder_simclr.pt", map_location=device))
model.eval().to(device)

# Load data
dataset = NDVITimeSeriesDataset("../data/series/")
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Create output directory
os.makedirs("../data/embeddings", exist_ok=True)

# Extract embeddings
with torch.no_grad():
    for i, (view1, _) in enumerate(loader):
        view1 = view1.to(device)
        embeddings = model(view1).cpu().numpy()  # shape: [B, 128]
        for j, emb in enumerate(embeddings):
            idx = i * loader.batch_size + j
            src = dataset.files[idx]
            name = os.path.splitext(os.path.basename(src))[0]
            np.save(f"../data/embeddings/{name}.npy", emb)
