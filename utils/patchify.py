from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, Literal

PadMode = Literal["edge", "constant"]

def pad_to_multiple2d(
    x: torch.Tensor, multiple: int, *, mode: PadMode = "edge", constant_value: float = 0.0
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Pad HxW (or CxHxW) tensor on bottom/right so H,W are multiples of `multiple`.
    Returns (padded, (pad_left, pad_right, pad_top, pad_bottom)).
    """
    if x.ndim == 2:
        H, W = x.shape
        C = None
    elif x.ndim == 3:
        C, H, W = x.shape
    else:
        raise ValueError("pad_to_multiple2d expects [H,W] or [C,H,W]")
    add_h = (multiple - (H % multiple)) % multiple
    add_w = (multiple - (W % multiple)) % multiple
    pad = (0, add_w, 0, add_h)  # (left,right,top,bottom) for F.pad with 2D uses (W_left,W_right,H_top,H_bottom)

    if add_h == 0 and add_w == 0:
        return x, (0,0,0,0)

    if x.ndim == 2:
        x_in = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        if mode == "edge":
            x_pad = F.pad(x_in, pad, mode="replicate")
        else:
            x_pad = F.pad(x_in, pad, mode="constant", value=constant_value)
        x_pad = x_pad.squeeze(0).squeeze(0)
    else:
        x_in = x.unsqueeze(0)               # [1,C,H,W]
        if mode == "edge":
            x_pad = F.pad(x_in, pad, mode="replicate")
        else:
            x_pad = F.pad(x_in, pad, mode="constant", value=constant_value)
        x_pad = x_pad.squeeze(0)
    return x_pad.contiguous(), (0, pad[1], 0, pad[3])

def patchify_2d(x: torch.Tensor, patch: int) -> torch.Tensor:
    """
    Mean-pool non-overlapping patches.
    x: [H,W] or [C,H,W]
    return: [Ph,Pw] or [C,Ph,Pw] where Ph=H//patch, Pw=W//patch
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)  # [1,H,W]
        c1 = True
    elif x.ndim == 3:
        c1 = False
    else:
        raise ValueError("patchify_2d expects [H,W] or [C,H,W]")
    C, H, W = x.shape
    assert H % patch == 0 and W % patch == 0, "Input must be divisible by patch"
    unfold = F.unfold(x, kernel_size=patch, stride=patch)         # [1,C*ps*ps, Ph*Pw]
    ps2 = patch * patch
    pooled = unfold.view(C, ps2, -1).mean(dim=1)                   # [C, Ph*Pw]
    Ph, Pw = H // patch, W // patch
    out = pooled.view(C, Ph, Pw)
    return out[0] if c1 else out

def unpatchify_2d(tokens: torch.Tensor, patch: int) -> torch.Tensor:
    """
    Reverse of patchify using nearest upsample (broadcast).
    tokens: [Ph,Pw] or [C,Ph,Pw] -> [H,W] or [C,H,W]
    """
    if tokens.ndim == 2:
        t = tokens.unsqueeze(0)  # [1,Ph,Pw]
        c1 = True
    elif tokens.ndim == 3:
        t = tokens
        c1 = False
    else:
        raise ValueError("unpatchify_2d expects [Ph,Pw] or [C,Ph,Pw]")
    t = t.unsqueeze(0)  # [1,C,Ph,Pw]
    up = F.interpolate(t, scale_factor=patch, mode="nearest")  # [1,C,H,W]
    up = up.squeeze(0)
    return up[0] if c1 else up

def valid_token_mask_from_footprint(H: int, W: int, patch: int, *, pad_bottom: int, pad_right: int) -> torch.BoolTensor:
    """
    Build a [Ph,Pw] boolean mask where True means the token is fully inside original (pre-pad) extent.
    """
    Hp, Wp = H + pad_bottom, W + pad_right
    # 1 inside original, 0 in padded region
    m = torch.ones((H, W), dtype=torch.float32)
    m_pad, _ = pad_to_multiple2d(m, patch, mode="constant", constant_value=0.0)  # [Hp,Wp]
    mask = patchify_2d(m_pad, patch)                                            # [Ph,Pw] in [0,1]
    return (mask == 1.0)

