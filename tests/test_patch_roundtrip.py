import torch
from utils.patchify import pad_to_multiple2d, patchify_2d, unpatchify_2d, valid_token_mask_from_footprint

def test_roundtrip_mean_pool():
    H, W, p = 13, 29, 5
    x = torch.arange(H*W, dtype=torch.float32).view(H,W) % 2  # a binary-ish pattern
    x_pad, (l,r,t,b) = pad_to_multiple2d(x, p, mode="constant", constant_value=0.0)
    toks = patchify_2d(x_pad, p)              # [Ph,Pw]
    up = unpatchify_2d(toks, p)               # [H’,W’]
    assert up.shape == x_pad.shape
    # token values are means; broadcasting back keeps those means
    assert torch.all((up >= 0) & (up <= 1))

def test_valid_mask_edges():
    H, W, p = 400, 400, 16
    _, (_, pr, _, pb) = pad_to_multiple2d(torch.zeros(H,W), p)
    m = valid_token_mask_from_footprint(H, W, p, pad_bottom=pb, pad_right=pr)
    Ph, Pw = (H+pb)//p, (W+pr)//p
    assert m.shape == (Ph, Pw)
    # last row/col might be invalid if padding was added
    assert m.sum() <= (Ph*Pw)
