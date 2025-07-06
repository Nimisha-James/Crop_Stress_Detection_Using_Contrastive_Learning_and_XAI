import os
import torch
import numpy as np
from model import Encoder3D
from dataset import MultimodalTimeSeriesDataset
from torch.utils.data import DataLoader

# Device config (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
C = 5
model = Encoder3D(in_ch=C)
model.load_state_dict(torch.load("../models/encoder_simclr.pt", map_location=device))
model.eval().to(device)

# Load data
dataset = MultimodalTimeSeriesDataset("../data/series/00")
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Create output directory
os.makedirs("../data/embeddings", exist_ok=True)

# Extract embeddings
with torch.no_grad():
    for i, (view1, _) in enumerate(loader):  # we only need view1 for inference
        view1 = view1.to(device)  # shape: [B, 6, 5, 64, 64]
        embeddings = model(view1).cpu().numpy()  # shape: [B, 128]
        for j, emb in enumerate(embeddings):
            idx = i * loader.batch_size + j
            src = dataset.paths[idx]
            name = os.path.splitext(os.path.basename(src))[0]
            np.save(f"../data/embeddings/{name}.npy", emb)
        
    print("Embeddings saved successfully..!!")


