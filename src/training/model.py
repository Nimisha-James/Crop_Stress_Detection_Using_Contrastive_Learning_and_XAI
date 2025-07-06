import torch, torch.nn as nn

class Encoder3D(nn.Module):
    """[B, T, C, 64, 64] ➜ [B, 128]"""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 16,  kernel_size=3, stride=2, padding=1),  # (T,H,W) treated as depth‑H‑W
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

    def forward(self, x):                       # x : [B,T,C,64,64]
        if x.dim() == 5 and x.size(2) != 1:     # [B, T, C, H, W] -> [B,C,T,H,W]  
            x = x.permute(0, 2, 1, 3, 4) 
        z = self.conv(x)
        z = self.proj(z)
        return z
