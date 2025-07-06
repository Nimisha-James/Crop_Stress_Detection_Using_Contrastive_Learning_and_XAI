import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalTimeSeriesDataset
from model import Encoder3D
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


def nt_xent(a, b, t=0.2):
    a = F.normalize(a, dim=1, eps=1e-6)
    b = F.normalize(b, dim=1, eps=1e-6)
    representations = torch.cat([a, b], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    batch_size = a.size(0)
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(a.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)

    positives = torch.sum(a * b, dim=-1).repeat(2)
    nominator = torch.exp(positives / t)
    denominator = torch.sum(torch.exp(similarity_matrix / t), dim=-1)

    loss = -torch.log(nominator / (denominator + 1e-8)).mean()
    return loss


def collate_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    v1, v2 = zip(*batch)
    return torch.stack(v1), torch.stack(v2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultimodalTimeSeriesDataset("../../data/series/00")
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    C = 5
    net = Encoder3D(in_ch=C).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)

    epoch_losses = []

    for epoch in range(50):
        last_loss = None

        for v1, v2 in loader:
            if v1 is None or v2 is None:
                continue
            v1, v2 = v1.float().to(device), v2.float().to(device)
            z1 = net(v1)
            z2 = net(v2)
            loss = nt_xent(z1, z2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            last_loss = loss.item()

        if last_loss is not None:
            epoch_losses.append(last_loss)
            print(f"epoch {epoch+1:02d}: loss {last_loss:.4f}")

    torch.save(net.state_dict(), "../../models/encoder_simclr.pt")

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 51), epoch_losses, marker='o', color='green', label='Last Batch Loss')
    plt.title("SimCLR Training Loss Curve (Last Batch Per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(range(1, 51, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../outputs/simclr_loss_curve.png")
    plt.show()
