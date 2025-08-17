# scripts/build_patch_cache.py
from __future__ import annotations
import os, json, time, argparse, yaml, pathlib
import torch
from torchvision.io import read_image
import rioxarray
import pandas as pd

from utils import init_wandb, save_artifact
from utils.patchify import pad_to_multiple2d, patchify_2d, valid_token_mask_from_footprint
from data.transforms import LandscapeNormalize, WeatherNormalize

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _scene_meta(data_dir: str):
    with open(os.path.join(data_dir, "landscape", "indices.json"), "r") as f:
        indices = json.load(f)
    return indices

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _nowstamp():
    return time.strftime("%Y%m%d-%H%M%S")

def build_cache(cfg):
    dcfg = cfg["data"]
    pcfg = cfg.get("patchify_on_disk", {})
    psize = int(pcfg.get("patch_size", 16))
    stride = int(pcfg.get("stride", psize))
    assert stride == psize, "For Phase 1 use non-overlapping patches (stride==patch_size)"

    data_dir = os.path.expanduser(dcfg["data_dir"])
    out_root = os.path.join(data_dir, pcfg.get("save_dir", "patch_cache"))
    _ensure_dir(out_root)
    seq_out = os.path.join(out_root, "sequences")
    _ensure_dir(seq_out)

    # ---- Normalizers & global stats ----
    lnorm = LandscapeNormalize()
    wnorm = WeatherNormalize()

    # Landscape full read (channels-first xarray)
    l_path = os.path.join(data_dir, "landscape", "Input_Geotiff.tif")
    xarr = lnorm(l_path)               # normalized, channels-first DataArray
    C, H, W = xarr.shape[0], xarr.shape[1], xarr.shape[2]
    static_min = lnorm.landscape_min.tolist()
    static_max = lnorm.landscape_max.tolist()

    # Pad full static once; compute static tokens once
    static_full = torch.from_numpy(xarr.values).float()  # [C,H,W]
    static_pad, (pl, pr, pt, pb) = pad_to_multiple2d(static_full, psize, mode="edge")
    PH, PW = static_pad.shape[1] // psize, static_pad.shape[2] // psize
    static_tokens_full = torch.stack([patchify_2d(static_pad[c], psize) for c in range(C)], dim=0)  # [C,PH,PW]
    torch.save(static_tokens_full, os.path.join(out_root, "static_tokens_full.pt"))

    # Weather: fit global min/max over all CSVs
    weather_folder = os.path.join(data_dir, "landscape", "Weathers")
    history_csv = os.path.join(data_dir, "landscape", "WeatherHistory.csv")
    weathers, weather_history = wnorm.fit_transform(weather_folder, history_csv)
    wstats = {
        "min_ws": float(wnorm.min_ws), "max_ws": float(wnorm.max_ws),
        "min_wd": float(wnorm.min_wd), "max_wd": float(wnorm.max_wd),
    }

    # Persist stats
    with open(os.path.join(out_root, "static_minmax.json"), "w") as f:
        json.dump({"min": static_min, "max": static_max}, f)
    with open(os.path.join(out_root, "weather_minmax.json"), "w") as f:
        json.dump(wstats, f)

    # Scene meta
    scene_meta = {
        "build_time": _nowstamp(),
        "patch_size": psize,
        "stride": stride,
        "pad": {"left": pl, "right": pr, "top": pt, "bottom": pb},
        "orig_hw": [H, W],
        "padded_hw": [static_pad.shape[1], static_pad.shape[2]],
        "token_hw": [PH, PW],
        "static_channels": C,
        "encoding": cfg.get("encoding", {}),
    }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(scene_meta, f, indent=2)

    # ---- Iterate sequences ----
    indices = _scene_meta(data_dir)
    fire_root = os.path.join(data_dir, "fire_frames")
    iso_root  = os.path.join(data_dir, "isochrones")
    seq_dirs = sorted([d for d in os.listdir(fire_root) if d.startswith("sequence_")])

    built = 0
    for seq_dir in seq_dirs:
        seq_id = seq_dir.replace("sequence_", "")
        seq_idx = str(int(seq_id))
        if seq_idx not in indices:
            print(f"[warn] indices missing for seq {seq_idx}, skipping")
            continue
        y0, y1, x0, x1 = indices[seq_idx]    # pixel coords (orig, unpadded)
        # Token coords in the padded grid
        ty0, ty1 = y0 // psize, (y1 + psize - 1) // psize
        tx0, tx1 = x0 // psize, (x1 + psize - 1) // psize
        Ph_seq, Pw_seq = ty1 - ty0, tx1 - tx0

        # Precompute spatial valid tokens for this crop
        vmask = valid_token_mask_from_footprint(H, W, psize, pad_bottom=pb, pad_right=pr)  # [PH,PW]
        vmask_seq = vmask[ty0:ty1, tx0:tx1].contiguous()                                   # [Ph_seq,Pw_seq]
        # Slice static tokens from full
        static_seq = static_tokens_full[:, ty0:ty1, tx0:tx1].contiguous()                  # [C,Ph_seq,Pw_seq]

        fire_files = sorted([f for f in os.listdir(os.path.join(fire_root, seq_dir)) if f.endswith(".png")])
        iso_files  = sorted([f for f in os.listdir(os.path.join(iso_root,  seq_dir)) if f.endswith(".png")])
        assert len(fire_files) == len(iso_files), f"{seq_dir}: mismatch fire/iso frames"
        T_total = len(fire_files)

        # Build per-frame fire/iso tokens + wind
        fire_tok = torch.empty((T_total, Ph_seq, Pw_seq), dtype=torch.float32)
        iso_tok  = torch.empty((T_total, Ph_seq, Pw_seq), dtype=torch.float32)
        wind     = torch.empty((T_total, 2), dtype=torch.float32)

        weather_file_name = weather_history.iloc[int(seq_id) - 1].values[0].split("Weathers/")[1]
        wdf = weathers[weather_file_name]

        for t in range(T_total):
            f_img = read_image(os.path.join(fire_root, seq_dir, fire_files[t]))   # [3,Hs,Ws]
            i_img = read_image(os.path.join(iso_root,  seq_dir, iso_files[t]))    # [3,Hs,Ws]
            f_mask = (f_img[1] == 231).float()                                    # [Hs,Ws]
            i_mask = (i_img[1] == 231).float()

            # paste into full-size canvas to reuse the same padding logic
            canvas_f = torch.zeros((H, W), dtype=torch.float32)
            canvas_i = torch.zeros((H, W), dtype=torch.float32)
            canvas_f[y0:y1, x0:x1] = f_mask
            canvas_i[y0:y1, x0:x1] = i_mask

            canvas_f_pad, _ = pad_to_multiple2d(canvas_f, psize, mode="constant", constant_value=0.0)
            canvas_i_pad, _ = pad_to_multiple2d(canvas_i, psize, mode="constant", constant_value=0.0)
            f_tok_full = patchify_2d(canvas_f_pad, psize)  # [PH,PW]
            i_tok_full = patchify_2d(canvas_i_pad, psize)

            fire_tok[t] = f_tok_full[ty0:ty1, tx0:tx1]
            iso_tok[t]  = i_tok_full[ty0:ty1, tx0:tx1]

            ws, wd = float(wdf.iloc[t]["WS"]), float(wdf.iloc[t]["WD"])
            wind[t] = torch.tensor([
                (ws - wnorm.min_ws) / max(1e-12, (wnorm.max_ws - wnorm.min_ws)),
                (wd - wnorm.min_wd) / max(1e-12, (wnorm.max_wd - wnorm.min_wd)),
            ])

        # Save seq
        seq_out_dir = os.path.join(seq_out, seq_dir)
        _ensure_dir(seq_out_dir)
        torch.save(static_seq, os.path.join(seq_out_dir, "static_tokens.pt"))
        torch.save(fire_tok,   os.path.join(seq_out_dir, "fire_tokens.pt"))
        torch.save(iso_tok,    os.path.join(seq_out_dir, "iso_tokens.pt"))
        torch.save(vmask_seq,  os.path.join(seq_out_dir, "valid_spatial.pt"))
        torch.save(wind,       os.path.join(seq_out_dir, "wind.pt"))
        with open(os.path.join(seq_out_dir, "meta.json"), "w") as f:
            json.dump({
                "sequence_id": seq_idx,
                "orig_bbox_xyxy": [x0, y0, x1, y1],
                "token_bbox_tytytxtx": [ty0, tx0, ty1, tx1],
                "frames": T_total,
                "token_hw": [Ph_seq, Pw_seq],
            }, f, indent=2)

        built += 1

    return out_root, built

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run = init_wandb(cfg, context={"script": pathlib.Path(__file__).stem})

    out_root, nseq = build_cache(cfg)
    print(f"[patchify] built cache at {out_root} | sequences: {nseq}")

    if run:
        # Log one artifact representing this build
        art_name = f"patch_cache-{_nowstamp()}"
        meta = {"sequences": nseq}
        from wandb import Artifact  # type: ignore
        # zip directory lazily via add_dir
        art = Artifact(name=art_name, type="dataset", metadata=meta)
        art.add_dir(out_root)
        run.log_artifact(art)
        run.finish()

if __name__ == "__main__":
    main()

