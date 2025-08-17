import torch
import torch.nn as nn
import torch.nn.functional as F

def _wind_broadcast(last_wind, H, W):
    return last_wind[:, :, None, None].expand(-1, -1, H, W)

class CopyLast(nn.Module):
    """Return the last frame as logits (strongly ±)."""
    def __init__(self, logit_scale: float = 8.0):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, fire, static, wind):
        last = fire[..., -1]                 # [B,1,H,W] in {0,1}
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
        last_fire = fire[..., -1]                    # [B,1,H,W]
        last_wind = wind[..., -1]                    # [B,2]
        H, W = last_fire.shape[-2:]
        wind_map = _wind_broadcast(last_wind, H, W)  # [B,2,H,W]
        x = torch.cat([last_fire, static, wind_map], dim=1)
        return self.net(x)                            # [B,1,H,W]

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
        x = fire.permute(0,1,4,2,3).contiguous()   # [B,1,T,H,W]
        y = self.block(x)                          # [B,1,T,H,W]
        return y[:, :, -1]                         # [B,1,H,W]


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch = ch_skip + ch_up
        self.conv = _DoubleConv(in_ch, out_ch)
    def forward(self, x_up, x_skip):
        x_up = F.interpolate(x_up, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)

class UNetS(nn.Module):
    """
    Small U-Net for patch grids (Stage C).
    Input:  [B, 1 + Cs + 2, Gy, Gx]   (last_fire, static bands, wind broadcast)
    Output: [B, 1, Gy, Gx]            (logits for next-step fire)
    """
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        c1, c2, c3 = base, base*2, base*4
        self.inc  = _DoubleConv(in_ch, c1)
        self.down1 = _Down(c1, c2)
        self.down2 = _Down(c2, c3)
        self.up1  = _Up(c3 + c2, c2)
        self.up2  = _Up(c2 + c1, c1)
        self.outc = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u2 = self.up1(x3, x2)
        u1 = self.up2(u2, x1)
        return self.outc(u1)
