import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NDVITimeSeriesDataset
from model import Encoder3D
import torch.optim as optim
import torch.nn.functional as F

def nt_xent(a, b, t=0.2):
    """a,b: [B,128]"""
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = torch.mm(a, torch.cat([b, a], dim=0).T) / t
    labels = torch.arange(a.size(0), device=a.device)
    loss_a = F.cross_entropy(logits[:, :a.size(0)], labels)
    loss_b = F.cross_entropy(logits[:, a.size(0):], labels)
    return (loss_a + loss_b) / 2

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ‑‑‑ data
    ds = NDVITimeSeriesDataset("../data/series/")
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    # ‑‑‑ model
    C=1
    net = Encoder3D(in_ch=C).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)

    # ‑‑‑ train
    for epoch in range(50):
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)   # [B,T, C,64,64]
            z1 = net(v1)                            # [B,128]
            z2 = net(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch:02d}: loss {loss.item():.4f}")

    torch.save(net.state_dict(), "../models/encoder_simclr.pt")
