import json, os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image

class TokenFireDataset(torch.utils.data.Dataset):
    """
    Loads per-sequence patch tokens from disk and yields training pairs:
      X = {static_tokens [N,Cs], last_fire_tokens [N], wind_last [2]}
      y = {target_patch_labels [N]}  (mean of target mask within each patch)

    You can later unpatchify predictions for pretty overlays.
    """
    def __init__(self, cache_dir: str, raw_root: str, sequence_length=3, fire_value=231):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.raw_root  = os.path.expanduser(raw_root)  # to read target PNGs for y
        self.sequence_length = sequence_length
        self.fire_value = fire_value

        # index sequences
        self.seq_dirs = sorted(d for d in os.listdir(self.cache_dir) if d.startswith("sequence_"))
        self.samples = []  # each = (seq_dir, t_last)
        for sd in self.seq_dirs:
            meta = json.load(open(os.path.join(self.cache_dir, sd, "meta.json")))
            T = meta["num_frames"]
            # with T frames, you can form (T-1) next-step targets
            for t_last in range(T-1):
                self.samples.append((sd, t_last))

    def __len__(self): return len(self.samples)

    def _target_patch_avg(self, seq_id_int: int, t_next: int, P: int, H:int, W:int, pad_mode: str):
        # load iso mask as [H,W] in {0,1}
        iso_dir = os.path.join(self.raw_root, "isochrones", f"sequence_{seq_id_int:03d}")
        iso_files = sorted(f for f in os.listdir(iso_dir) if f.endswith(".png"))
        img = read_image(os.path.join(iso_dir, iso_files[t_next]))  # [3,h,w]
        iso = (img[1] == self.fire_value).float().unsqueeze(0)      # [1,h,w]

        # we don't know the crop size here; meta contains original H,W
        # If the iso image is already cropped per sequence (it is), H,W should match.
        # pad and avgpool to patch grid
        C, h, w = iso.shape
        pad_h = (P - (h % P)) % P
        pad_w = (P - (w % P)) % P
        iso_pad = F.pad(iso.unsqueeze(0), (0,pad_w,0,pad_h), mode="constant", value=0.0).squeeze(0)
        grid = F.avg_pool2d(iso_pad, kernel_size=P, stride=P)        # [1,Gy,Gx]
        return grid.view(-1)                                         # [N]

    def __getitem__(self, i):
        seq_dir, t_last = self.samples[i]
        seq_path = os.path.join(self.cache_dir, seq_dir)
        meta = json.load(open(os.path.join(seq_path, "meta.json")))

        static = np.load(os.path.join(seq_path, meta["static_tokens"]))  # [N,Cs]
        valid  = np.load(os.path.join(seq_path, meta["valid_tokens"]))   # [N]
        wind   = np.load(os.path.join(seq_path, meta["wind"]))           # [T,2]

        fire_last = np.load(os.path.join(seq_path, meta["fires"][str(t_last)]))  # [N]
        t_next = t_last + 1
        # target per-patch average
        seq_id_int = int(meta["sequence_id"])
        y_patch = self._target_patch_avg(seq_id_int, t_next, meta["patch_size"], meta["H"], meta["W"], meta["pad_mode"])

        X = {
            "static": torch.from_numpy(static).float(),          # [N,Cs]
            "fire_last": torch.from_numpy(fire_last).float(),    # [N]
            "wind_last": torch.from_numpy(wind[t_last]).float(), # [2]
            "valid": torch.from_numpy(valid).bool(),             # [N]
            "meta": meta,
        }
        y = y_patch.float()                                      # [N] in [0,1] mean of target within patch
        return X, y

