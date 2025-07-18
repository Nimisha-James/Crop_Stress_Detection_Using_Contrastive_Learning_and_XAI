import sys
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

from training.model   import Encoder3D
from training.dataset import MultimodalTimeSeriesDataset

# gradcam class
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        target_layer.register_forward_hook(self._hook_activation)
        target_layer.register_backward_hook(self._hook_gradient)
        self.act, self.grad = None, None

    def _hook_activation(self, m, i, o): self.act = o.detach()
    def _hook_gradient (self, m, gi, go): self.grad = go[0].detach()

    def __call__(self, view1, view2):
        self.model.zero_grad(set_to_none=True)
        z1, z2 = self.model(view1), self.model(view2)
        similarity = F.cosine_similarity(z1, z2).squeeze()
        similarity.backward()

        A = self.act.squeeze(0)   # [C,T,H,W]
        G = self.grad.squeeze(0)  # [C,T,H,W]
        weights = G.mean(dim=(1,2,3))          # [C]
        cam = torch.sum(weights[:,None,None,None] * A, dim=0)  # [T,H,W]
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.cpu()                       # [T,H,W]
    
def main():
    os.makedirs("../../outputs/gradcam", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Encoder3D(in_ch=5).to(device)
    model.load_state_dict(torch.load("../../models/encoder_simclr.pt",map_location=device))

    target_layer = model.conv[0]
    print(model.conv)
    cam_gen = GradCAM3D(model, target_layer)

    ds = MultimodalTimeSeriesDataset("../../new_data/new_series/00")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    for n, (v1, v2) in enumerate(loader):

        patch_path = ds.paths[n]
        patch_id   = os.path.splitext(os.path.basename(patch_path))[0]

        v1, v2 = v1.to(device), v2.to(device)
        cam    = cam_gen(v1, v2)                            # [T, H, W]


        print("CAM shape:", cam.shape) 
        cam = cam.cpu().numpy()                             # [T, 16, 16]
        T = cam.shape[0]

        cube = v1[0].cpu().numpy()                          # [T, C, H, W]
        base_imgs = cube[:T].mean(axis=1)


        # Plot T subplots
        fig, axes = plt.subplots(1, T, figsize=(6, 3))

        for t in range(T):
            ax   = axes[t]
            base = base_imgs[t]                 # [H,W]  (64×64)
            heat = cam[t]               # [16×16]
            ax.imshow(base, cmap='gray', vmin=0, vmax=1)
            ax.imshow(heat, cmap='jet', alpha=0.55)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Time Filter {t+1}")

        plt.tight_layout(h_pad=0.3, w_pad=0.3)
        fig.supxlabel(f"Patch ID {patch_id}", fontsize=12)
        out_png = f"../../outputs/gradcam/{patch_id}_cam.png"
        plt.savefig(out_png, dpi=140)
        plt.close()
        print(f"✓ Saved CAM row → {out_png}")

if __name__ == "__main__":
    main()