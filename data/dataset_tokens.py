from __future__ import annotations
import os, json, torch
from typing import Dict, List, Tuple

class TokenFireDataset(torch.utils.data.Dataset):
    """
    Loads patch-token cache produced by scripts/build_patch_cache.py

    Each sample (sliding window of length T):
      fire_tokens: [1, Ph, Pw, T-1]   (mean occupancy per patch in [0,1])
      static_tok : [Cs, Ph, Pw]
      wind       : [T-1, 2]           (normalized)
      target_tok : [1, Ph, Pw]        (iso tokens for next timestep)
    """
    def __init__(self, cache_root: str, sequence_length: int = 3):
        super().__init__()
        self.cache_root = os.path.expanduser(cache_root)
        self.sequence_length = int(sequence_length)

        # Scene meta
        with open(os.path.join(self.cache_root, "meta.json"), "r") as f:
            self.scene_meta = json.load(f)

        # Enumerate sequences
        self.seq_root = os.path.join(self.cache_root, "sequences")
        self.seq_dirs = sorted([d for d in os.listdir(self.seq_root) if d.startswith("sequence_")])

        # Build index of windows
        self.index: List[Dict] = []
        for sd in self.seq_dirs:
            sdir = os.path.join(self.seq_root, sd)
            with open(os.path.join(sdir, "meta.json"), "r") as f:
                smeta = json.load(f)
            T = int(smeta["frames"])
            if T < 2:
                continue
            # Store path refs and sliding windows
            for start in range(T - self.sequence_length + 1):
                self.index.append({
                    "seq_dir": sdir,
                    "fire_path": os.path.join(sdir, "fire_tokens.pt"),
                    "iso_path":  os.path.join(sdir, "iso_tokens.pt"),
                    "static_path": os.path.join(sdir, "static_tokens.pt"),
                    "wind_path": os.path.join(sdir, "wind.pt"),
                    "valid_spatial_path": os.path.join(sdir, "valid_spatial.pt"),
                    "sub_ids": list(range(start, start + self.sequence_length)),
                })

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        it = self.index[idx]
        fire = torch.load(it["fire_path"])    # [Tf, Ph, Pw]
        iso  = torch.load(it["iso_path"])     # [Tf, Ph, Pw]
        stat = torch.load(it["static_path"])  # [Cs, Ph, Pw]
        wind = torch.load(it["wind_path"])    # [Tf, 2]
        # vmask not needed for forward pass yet; you may use it downstream
        # vmask = torch.load(it["valid_spatial_path"])  # [Ph, Pw] (bool)

        ids = it["sub_ids"]
        past = ids[:-1]
        target_id = ids[-1]

        # Assemble shapes to match collate contract but at patch grid
        # fire_tokens -> [1, Ph, Pw, T-1]
        fire_hist = fire[past]                  # [T-1, Ph, Pw]
        fire_hist = fire_hist.permute(1,2,0).unsqueeze(0).contiguous()  # [1,Ph,Pw,T-1]

        static_tok = stat.contiguous()          # [Cs, Ph, Pw]
        wind_hist  = wind[past].contiguous()    # [T-1, 2]

        target_tok = iso[target_id].unsqueeze(0).contiguous()  # [1,Ph,Pw]

        return fire_hist, static_tok, wind_hist, target_tok

