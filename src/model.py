import torch, torch.nn as nn

class Encoder3D(nn.Module):
    """[B, 12, 64, 64] ➜ [B, 256]"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16,  kernel_size=3, stride=2, padding=1),  # (T,H,W) treated as depth‑H‑W
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),                           # output ≈ [B,64,2,8,8]
            nn.AdaptiveAvgPool3d(1)              # ➜ [B,64,1,1,1]
        )
        self.proj = nn.Sequential(
            nn.Flatten(),                        # [B,64]
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128)                  # SimCLR projection dim
        )

    def forward(self, x):                       # x : [B,12,64,64]
        x = x.unsqueeze(1)                      # add channel -> [B,1,12,64,64]
        z = self.conv(x)
        z = self.proj(z)
        return z
