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

    Supports both:
      - NEW: single file fire tokens: fire_tokens.npy  [T, N]
      - OLD: per-frame files: fires:{ "0": "fire_tokens_t000.npy", ... }
    """
    def __init__(self, cache_dir: str, raw_root: str, sequence_length=3, fire_value=231):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.raw_root  = os.path.expanduser(raw_root)
        self.sequence_length = sequence_length
        self.fire_value = fire_value

        self.seq_dirs = sorted(d for d in os.listdir(self.cache_dir) if d.startswith("sequence_"))
        self.samples = []  # each = (seq_dir, t_last)
        for sd in self.seq_dirs:
            meta = json.load(open(os.path.join(self.cache_dir, sd, "meta.json")))
            T = int(meta["num_frames"])
            for t_last in range(T - 1):
                self.samples.append((sd, t_last))

    def __len__(self): return len(self.samples)

    def _target_patch_avg(self, seq_id_int: int, t_next: int, P: int):
        iso_dir = os.path.join(self.raw_root, "isochrones", f"sequence_{seq_id_int:03d}")
        iso_files = sorted(f for f in os.listdir(iso_dir) if f.endswith(".png"))
        img = read_image(os.path.join(iso_dir, iso_files[t_next]))  # [3,h,w]
        iso = (img[1] == self.fire_value).float().unsqueeze(0)      # [1,h,w]
        pad_h = (P - (iso.shape[1] % P)) % P
        pad_w = (P - (iso.shape[2] % P)) % P
        iso_pad = F.pad(iso.unsqueeze(0), (0,pad_w,0,pad_h), mode="constant", value=0.0).squeeze(0)
        grid = F.avg_pool2d(iso_pad, kernel_size=P, stride=P)        # [1,Gy,Gx]
        return grid.view(-1)                                         # [N]

    def __getitem__(self, i):
        seq_dir, t_last = self.samples[i]
        seq_path = os.path.join(self.cache_dir, seq_dir)
        meta = json.load(open(os.path.join(seq_path, "meta.json")))

        static = np.load(os.path.join(seq_path, meta["static_tokens"]), mmap_mode="r")  # [N,Cs]
        valid  = np.load(os.path.join(seq_path, meta["valid_tokens"]),  mmap_mode="r")  # [N]
        wind   = np.load(os.path.join(seq_path, meta["wind"]),          mmap_mode="r")  # [T,2]

        # NEW: single-file path
        fire_last = None
        if "fire_tokens" in meta:
            ft = np.load(os.path.join(seq_path, meta["fire_tokens"]), mmap_mode="r")    # [T,N]
            fire_last = ft[t_last]
        else:
            # OLD format fallback
            fire_last = np.load(os.path.join(seq_path, meta["fires"][str(t_last)]))     # [N]

        t_next = t_last + 1
        seq_id_int = int(meta["sequence_id"])
        P = int(meta["patch_size"])
        y_patch = self._target_patch_avg(seq_id_int, t_next, P)

        X = {
            "static": torch.from_numpy(np.asarray(static)).float().clone(),          # [N,Cs]
            "fire_last": torch.from_numpy(np.asarray(fire_last)).float().clone(),    # [N]
            "wind_last": torch.from_numpy(np.asarray(wind[t_last])).float().clone(), # [2]
            "valid": torch.from_numpy(np.asarray(valid)).bool().clone(),             # [N]
            "meta": meta,
        }
        y = y_patch.float()                                                  # [N]
        return X, y

