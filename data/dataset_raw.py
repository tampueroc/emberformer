import os
import torch
import json
from torchvision.io import read_image

from .transforms import LandscapeNormalize, WeatherNormalize


class RawFireDataset:
    """
    Custom PyTorch Dataset for fire state sequences, static landscapes, and wind inputs.

    Returns per sample (no padding applied):
      fire_sequence:  [1, H, W, T]  (binary)
      static_data:    [C_static, H, W]  (normalized; binary bands untouched)
      wind_inputs:    [T, 2]  (WS, WD normalized globally)
      isochrone_mask: [1, H, W]  (binary)
    """
    def __init__(self, data_dir, sequence_length=3, transform=None):
        super().__init__()
        self.data_dir = os.path.expanduser(f"~/{data_dir}")
        self.sequence_length = sequence_length
        self.transform = transform

        # Normalizers
        self.landscape_normalizer = LandscapeNormalize()
        self.weather_normalizer = WeatherNormalize()

        # Internal containers
        self.indices = {}
        self.samples = []

        # 1) Load & normalize landscape data (channels-first xarray)
        self.landscape_data = self.landscape_normalizer(
            os.path.join(self.data_dir, "landscape", "Input_Geotiff.tif")
        )
        self.landscape_max = self.landscape_normalizer.landscape_max
        self.landscape_min = self.landscape_normalizer.landscape_min

        # 2) Load spatial indices
        self._load_spatial_index()

        # 3) Load & normalize weather (compute global min/max)
        weather_folder = os.path.join(self.data_dir, "landscape", "Weathers")
        weather_history_path = os.path.join(self.data_dir, "landscape", "WeatherHistory.csv")
        self.weathers, self.weather_history = self.weather_normalizer.fit_transform(
            weather_folder, weather_history_path
        )

        # 4) Prepare sub-sequences
        self._prepare_samples()

    def _load_spatial_index(self):
        path = os.path.join(self.data_dir, "landscape", "indices.json")
        with open(path, "r") as f:
            self.indices = json.load(f)

    def _prepare_samples(self):
        fire_root = os.path.join(self.data_dir, "fire_frames")
        iso_root = os.path.join(self.data_dir, "isochrones")
        seq_dirs = sorted(os.listdir(fire_root))

        for seq_dir in seq_dirs:
            seq_id = seq_dir.replace("sequence_", "")
            fseq_path = os.path.join(fire_root, seq_dir)
            iseq_path = os.path.join(iso_root, seq_dir)

            fire_files = sorted(f for f in os.listdir(fseq_path) if f.endswith(".png"))
            iso_files = sorted(f for f in os.listdir(iseq_path) if f.endswith(".png"))

            num_frames = len(fire_files)
            assert num_frames == len(iso_files), f"{seq_id} mismatch in frames"

            # Crop the big landscape to just the chunk for this sequence
            seq_id = str(int(seq_id))
            if seq_id not in self.indices:
                raise ValueError(f"No indices found for seq {seq_id}")
            y, y_, x, x_ = self.indices[seq_id]

            # xarray -> numpy -> torch (C, H, W)
            cropped = self.landscape_data[:, y:y_, x:x_].values
            cropped_tensor = torch.from_numpy(cropped).float()

            # Build sub-sequences up to `sequence_length`
            T = min(num_frames, self.sequence_length)
            if T < 2:
                continue

            for start in range(num_frames - T + 1):
                sub_ids = list(range(start, start + T))
                self.samples.append({
                    "sequence_id": seq_id,
                    "fire_path": fseq_path,
                    "iso_path": iseq_path,
                    "fire_files": fire_files,
                    "iso_files": iso_files,
                    "fire_frame_indices": sub_ids[:-1],  # past frames
                    "iso_target_index": sub_ids[-1],     # next frame (target)
                    "landscape": cropped_tensor
                })

    def __getitem__(self, index):
        item = self.samples[index]
        seq_id = item["sequence_id"]

        # 1) Fire history -> [1, H, W, T]
        frames = []
        for frame_idx in item["fire_frame_indices"]:
            fpath = os.path.join(item["fire_path"], item["fire_files"][frame_idx])
            img = read_image(fpath)  # [3, H, W], uint8
            mask = (img[1] == 231).float().unsqueeze(0)  # [1, H, W]
            frames.append(mask)
        # Stack along new time axis (last)
        fire_sequence = torch.stack(frames, dim=-1)  # [1, H, W, T]

        # 2) Static landscape (already cropped & normalized) -> [C_static, H, W]
        static_data = item["landscape"]  # [C_static, H, W]

        # 3) Wind inputs per timestep -> [T, 2]
        weather_file_name = self.weather_history.iloc[int(seq_id) - 1].values[0].split("Weathers/")[1]
        weather_df = self.weathers[weather_file_name]
        wind_rows = []
        for frame_idx in item["fire_frame_indices"]:
            row = weather_df.iloc[frame_idx]
            ws, wd = row["WS"], row["WD"]
            wind_rows.append(self.weather_normalizer(ws, wd))  # [2]
        wind_inputs = torch.stack(wind_rows, dim=0)  # [T, 2]

        # 4) Isochrone target -> [1, H, W]
        itarget = item["iso_target_index"]
        ipath = os.path.join(item["iso_path"], item["iso_files"][itarget])
        iso_img = read_image(ipath)  # [3, H, W]
        isochrone_mask = (iso_img[1] == 231).float().unsqueeze(0)  # [1, H, W]

        # Optional transforms (apply consistently per tensor)
        if self.transform is not None:
            # Apply per-timestep transform correctly along T (last axis)
            transformed = []
            T = fire_sequence.shape[-1]
            for t in range(T):
                f_t = fire_sequence[..., t]          # [1, H, W]
                f_t = self.transform(f_t)            # [1, H, W]
                transformed.append(f_t)
            fire_sequence = torch.stack(transformed, dim=-1)  # [1, H, W, T]

            static_data = self.transform(static_data)         # [C_static, H, W]
            isochrone_mask = self.transform(isochrone_mask)   # [1, H, W]

        return fire_sequence, static_data, wind_inputs, isochrone_mask

    def __len__(self):
        return len(self.samples)

