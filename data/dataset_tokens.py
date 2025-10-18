import json, os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image

class TokenFireDataset(torch.utils.data.Dataset):
    """
    Loads per-sequence patch tokens from disk and yields training pairs:
      X = {static_tokens [N,Cs], fire_hist [T_hist,N], wind_hist [T_hist,2], valid [N], meta}
      y = {target_patch_labels [N]}  (mean of target mask within each patch)

    If use_history=False (backward compat):
      X = {static_tokens [N,Cs], fire_last [N], wind_last [2], valid [N], meta}

    IMPORTANT: Sliding Window Implementation
    -----------------------------------------
    For a sequence with T frames, creates T-1 training samples with EXPANDING history:
      Sample 0: [frame0] → predicts frame1           (T_hist=1)
      Sample 1: [frame0, frame1] → predicts frame2   (T_hist=2)
      Sample 2: [frame0, frame1, frame2] → predicts frame3  (T_hist=3)
      ...
      Sample T-2: [frame0, ..., frameT-2] → predicts frameT-1  (T_hist=T-1)
    
    Each sample uses ALL history from the beginning up to t_last.
    The collate_tokens_temporal function left-pads to T_max in the batch.

    Supports both:
      - NEW: single file fire tokens: fire_tokens.npy  [T, N]
      - OLD: per-frame files: fires:{ "0": "fire_tokens_t000.npy", ... }
    """
    def __init__(self, cache_dir: str, raw_root: str, sequence_length=3, fire_value=231, use_history=True):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.raw_root  = os.path.expanduser(raw_root)
        self.sequence_length = sequence_length
        self.fire_value = fire_value
        self.use_history = use_history

        self.seq_dirs = sorted(d for d in os.listdir(self.cache_dir) if d.startswith("sequence_"))
        self.samples = []  # each = (seq_dir, t_last)
        for sd in self.seq_dirs:
            meta = json.load(open(os.path.join(self.cache_dir, sd, "meta.json")))
            T = int(meta["num_frames"])
            for t_last in range(T - 1):
                self.samples.append((sd, t_last))

    def __len__(self): return len(self.samples)

    def _target_patch_avg(self, seq_id_int: int, t_next: int, P: int):
        """DEPRECATED: Returns patch averages. Use _target_full_res for pixel-level targets."""
        iso_dir = os.path.join(self.raw_root, "isochrones", f"sequence_{seq_id_int:03d}")
        iso_files = sorted(f for f in os.listdir(iso_dir) if f.endswith(".png"))
        img = read_image(os.path.join(iso_dir, iso_files[t_next]))  # [3,h,w]
        iso = (img[1] == self.fire_value).float().unsqueeze(0)      # [1,h,w]
        pad_h = (P - (iso.shape[1] % P)) % P
        pad_w = (P - (iso.shape[2] % P)) % P
        iso_pad = F.pad(iso.unsqueeze(0), (0,pad_w,0,pad_h), mode="constant", value=0.0).squeeze(0)
        grid = F.avg_pool2d(iso_pad, kernel_size=P, stride=P)        # [1,Gy,Gx]
        return grid.view(-1)                                         # [N]
    
    def _target_full_res(self, seq_id_int: int, t_next: int):
        """Load full-resolution target image for pixel-level training."""
        iso_dir = os.path.join(self.raw_root, "isochrones", f"sequence_{seq_id_int:03d}")
        iso_files = sorted(f for f in os.listdir(iso_dir) if f.endswith(".png"))
        img = read_image(os.path.join(iso_dir, iso_files[t_next]))  # [3,H,W]
        iso = (img[1] == self.fire_value).float().unsqueeze(0)      # [1,H,W]
        return iso

    def __getitem__(self, i):
        seq_dir, t_last = self.samples[i]
        seq_path = os.path.join(self.cache_dir, seq_dir)
        meta = json.load(open(os.path.join(seq_path, "meta.json")))

        static = np.load(os.path.join(seq_path, meta["static_tokens"]), mmap_mode="r")  # [N,Cs]
        valid  = np.load(os.path.join(seq_path, meta["valid_tokens"]),  mmap_mode="r")  # [N]
        wind   = np.load(os.path.join(seq_path, meta["wind"]),          mmap_mode="r")  # [T,2]

        t_next = t_last + 1
        seq_id_int = int(meta["sequence_id"])
        P = int(meta["patch_size"])
        
        # Load full-resolution target for pixel-level training
        y_full = self._target_full_res(seq_id_int, t_next)  # [1, H, W]

        static_np = np.asarray(static).copy()
        valid_np  = np.asarray(valid).copy()

        gH = meta.get("Gy") or meta.get("grid_h")
        gW = meta.get("Gx") or meta.get("grid_w")
        grid_hw = torch.tensor([gH, gW], dtype=torch.int32)

        if self.use_history:
            # Return full temporal history: fire_hist [T_hist, N], wind_hist [T_hist, 2]
            if "fire_tokens" in meta:
                ft = np.load(os.path.join(seq_path, meta["fire_tokens"]), mmap_mode="r")    # [T,N]
                fire_hist = ft[:t_last+1]  # [T_hist, N] where T_hist = t_last+1
            else:
                # OLD format fallback: load each frame individually
                fire_hist_list = []
                for t in range(t_last + 1):
                    fire_hist_list.append(np.load(os.path.join(seq_path, meta["fires"][str(t)])))
                fire_hist = np.stack(fire_hist_list, axis=0)  # [T_hist, N]
            
            wind_hist = wind[:t_last+1]  # [T_hist, 2]
            
            X = {
                "static":    torch.from_numpy(static_np).float(),        # [N,Cs]
                "fire_hist": torch.from_numpy(fire_hist.copy()).float(), # [T_hist,N]
                "wind_hist": torch.from_numpy(wind_hist.copy()).float(), # [T_hist,2]
                "valid":     torch.from_numpy(valid_np).bool(),          # [N]
                "meta":      grid_hw,
                "seq_id":    seq_id_int,                                 # For loading raw images
                "t_last":    t_last,                                     # Frame index
                "t_next":    t_next,                                     # Target frame index
                "patch_size": P,                                         # For proper unpatchification
            }
        else:
            # Backward compatibility: return only last frame
            if "fire_tokens" in meta:
                ft = np.load(os.path.join(seq_path, meta["fire_tokens"]), mmap_mode="r")    # [T,N]
                fire_last = ft[t_last]
            else:
                # OLD format fallback
                fire_last = np.load(os.path.join(seq_path, meta["fires"][str(t_last)]))     # [N]
            
            fire_np   = np.asarray(fire_last).copy()
            wind_np   = np.asarray(wind[t_last]).copy()
            
            X = {
                "static":    torch.from_numpy(static_np).float(),   # [N,Cs]
                "fire_last": torch.from_numpy(fire_np).float(),     # [N]
                "wind_last": torch.from_numpy(wind_np).float(),     # [2]
                "valid":     torch.from_numpy(valid_np).bool(),     # [N]
                "meta":      grid_hw,
            }

        y = y_full  # [1, H, W] full resolution
        return X, y

