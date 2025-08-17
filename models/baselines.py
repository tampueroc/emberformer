import torch
import torch.nn as nn
import torch.nn.functional as F

def _wind_broadcast(last_wind, H, W):
    # last_wind: [B,2] -> [B,2,H,W]
    return last_wind[:, :, None, None].expand(-1, -1, H, W)

class CopyLast(nn.Module):
    """Return the last frame as logits (strongly ±)."""
    def __init__(self, logit_scale: float = 8.0):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, fire, static, wind):
        # fire: [B,1,H,W,T]
        last = fire[..., -1]                 # [B,1,H,W] in {0,1}
        # Map {0,1} -> logits with margin
        return (last * 2 - 1) * self.logit_scale

class ConvHead2D(nn.Module):
    """2D conv head over [last_fire ⊕ static ⊕ broadcast(last_wind)]."""
    def __init__(self, static_channels: int, hidden: int = 32):
        super().__init__()
        in_ch = 1 + static_channels + 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, fire, static, wind):
        # fire: [B,1,H,W,T], static: [B,Cs,H,W], wind: [B,2,T]
        last_fire = fire[..., -1]                    # [B,1,H,W]
        last_wind = wind[..., -1]                    # [B,2]
        B, _, H, W = last_fire.shape
        wind_map = _wind_broadcast(last_wind, H, W)  # [B,2,H,W]
        x = torch.cat([last_fire, static, wind_map], dim=1)
        return self.net(x)                            # [B,1,H,W] (logits)

class Tiny3D(nn.Module):
    """Small 3D conv over time; take last depth slice as next-step logits."""
    def __init__(self, in_ch: int = 1, hidden: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, hidden, (3,3,3), padding=(1,1,1)),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, (3,3,3), padding=(1,1,1)),
            nn.GELU(),
            nn.Conv3d(hidden, 1, (1,1,1))
        )

    def forward(self, fire, static, wind):
        # fire: [B,1,H,W,T] -> 3D expects [B,C,D,H,W]
        x = fire.permute(0,1,4,2,3).contiguous()   # [B,1,T,H,W]
        y = self.block(x)                          # [B,1,T,H,W] logits
        return y[:, :, -1]                         # [B,1,H,W]

