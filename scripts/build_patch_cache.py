import argparse, json, os, pathlib, time, math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from tqdm import tqdm  # progress bar (guarded by config)

from data.dataset_raw import RawFireDataset
import yaml

# ----------------
# Config + helpers
# ----------------
def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _expand(p): return os.path.expanduser(p)
def _now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _write_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")

# -----------------------
# Pooling / tokenization
# -----------------------
@torch.no_grad()
def _avgpool_grid(x_chw: torch.Tensor, P: int, pad_mode: str, const_val: float = 0.0):
    """
    x_chw: [C,H,W]; returns (grid[C, Gy, Gx], meta)
    """
    C, H, W = x_chw.shape
    if pad_mode == "strict":
        if (H % P) != 0 or (W % P) != 0:
            raise RuntimeError(f"strict mode: H={H}, W={W} must be divisible by P={P}")
        grid = F.avg_pool2d(x_chw, kernel_size=P, stride=P)  # [C,Gy,Gx]
        Gy, Gx = grid.shape[-2:]
        meta = dict(H=H, W=W, H_pad=H, W_pad=W, Gy=Gy, Gx=Gx, P=P, pad_bottom=0, pad_right=0)
        return grid, meta

    # legacy pad modes (kept if you flip the config)
    pad_h = (P - (H % P)) % P
    pad_w = (P - (W % P)) % P
    pad = (0, pad_w, 0, pad_h)
    if pad_mode == "edge":
        x_pad = F.pad(x_chw.unsqueeze(0), pad, mode="replicate").squeeze(0)
    elif pad_mode == "constant":
        x_pad = F.pad(x_chw.unsqueeze(0), pad, mode="constant", value=const_val).squeeze(0)
    elif pad_mode == "pad_to_multiple":
        x_pad = F.pad(x_chw.unsqueeze(0), pad, mode="constant", value=0.0).squeeze(0)
    else:
        raise ValueError(f"unknown pad_mode {pad_mode}")

    grid = F.avg_pool2d(x_pad, kernel_size=P, stride=P)
    Gy, Gx = grid.shape[-2:]
    meta = dict(H=H, W=W, H_pad=H+pad_h, W_pad=W+pad_w, Gy=Gy, Gx=Gx, P=P,
                pad_bottom=pad_h, pad_right=pad_w)
    return grid, meta

def _target_patch_avg_png(iso_png_path: str, fire_value: int, P: int):
    img = read_image(iso_png_path)  # [3,h,w], uint8
    iso = (img[1] == fire_value).float().unsqueeze(0)  # [1,h,w]
    _, h, w = iso.shape
    pad_h = (P - (h % P)) % P
    pad_w = (P - (w % P)) % P
    iso_pad = F.pad(iso.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=0.0).squeeze(0)
    grid = F.avg_pool2d(iso_pad, kernel_size=P, stride=P)  # [1,Gy,Gx]
    return grid.view(-1)  # [N]

# ---------------
# Build one seq
# ---------------
def build_one_sequence(ds: RawFireDataset, seq_id: str, out_dir: str, pcfg: dict, fire_value: int):
    P = int(pcfg["patch_size"])
    pad_mode = pcfg.get("pad_mode", "strict")
    const_val = float(pcfg.get("constant_pad_value", 0.0))

    # crop static once from the big GeoTIFF using indices
    y0, y1, x0, x1 = ds.indices[seq_id]
    static_chw_np = ds.landscape_data[:, y0:y1, x0:x1].values  # (Cs, H, W)
    static_chw = torch.from_numpy(static_chw_np).float()
    Cs, H, W = static_chw.shape

    # strict: H,W must be divisible by P (e.g., 400 % 16 == 0)
    static_grid, meta = _avgpool_grid(static_chw, P, pad_mode=pad_mode, const_val=const_val)
    Gy, Gx = meta["Gy"], meta["Gx"]
    N = Gy * Gx
    static_tokens = static_grid.view(Cs, -1).T.contiguous().numpy().astype(np.float32)  # [N, Cs]

    # strict: all tokens valid
    valid_tokens = np.ones((N,), dtype=np.bool_)

    fire_root = os.path.join(ds.data_dir, "fire_frames", f"sequence_{int(seq_id):03d}")
    iso_root  = os.path.join(ds.data_dir, "isochrones",  f"sequence_{int(seq_id):03d}")
    fire_files = sorted(f for f in os.listdir(fire_root) if f.endswith(".png"))
    iso_files  = sorted(f for f in os.listdir(iso_root)  if f.endswith(".png"))
    assert len(fire_files) == len(iso_files), f"mismatch frames on seq {seq_id}"
    T_all = len(fire_files)

    # assert PNG size matches static H,W in strict mode
    sample_img = read_image(os.path.join(fire_root, fire_files[0]))  # [3,h,w]
    h_png, w_png = sample_img.shape[1], sample_img.shape[2]
    if pad_mode == "strict":
        if (h_png != H) or (w_png != W):
            raise RuntimeError(f"seq {seq_id}: PNG {h_png}x{w_png} != static {H}x{W} in strict mode")
        if (h_png % P) != 0 or (w_png % P) != 0:
            raise RuntimeError(f"seq {seq_id}: PNG {h_png}x{w_png} not divisible by P={P}")

    # wind table
    weather_file = ds.weather_history.iloc[int(seq_id) - 1].values[0].split("Weathers/")[1]
    wdf = ds.weathers[weather_file]

    _ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "static_tokens.npy"), static_tokens)
    np.save(os.path.join(out_dir, "valid_tokens.npy"),  valid_tokens)

    wind_list = []
    fire_index_to_file = {}

    for t, fname in enumerate(fire_files):
        img = read_image(os.path.join(fire_root, fname))  # [3,H,W]
        fire = (img[1] == fire_value).float().unsqueeze(0)  # [1,H,W]  (no extra crop)
        fire_grid, _ = _avgpool_grid(fire, P, pad_mode=pad_mode, const_val=0.0)
        fire_tokens = fire_grid.view(-1).contiguous().numpy().astype(np.float32)  # [N]
        np.save(os.path.join(out_dir, f"fire_tokens_t{t:03d}.npy"), fire_tokens)
        fire_index_to_file[t] = f"fire_tokens_t{t:03d}.npy"

        ws = float(wdf.iloc[t]["WS"]); wd = float(wdf.iloc[t]["WD"])
        wind_list.append(ds.weather_normalizer(ws, wd).numpy())

    wind = np.stack(wind_list, axis=0)  # [T,2]
    np.save(os.path.join(out_dir, "wind.npy"), wind)

    meta.update({
        "sequence_id": seq_id,
        "patch_size": P,
        "grid_h": Gy, "grid_w": Gx, "num_patches": N,
        "num_frames": T_all,
        "pad_mode": pad_mode,
        "static_tokens": "static_tokens.npy",
        "valid_tokens": "valid_tokens.npy",
        "wind": "wind.npy",
        "fires": fire_index_to_file,
        "fire_value": int(fire_value),
    })
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    preview = torch.from_numpy(np.load(os.path.join(out_dir, fire_index_to_file[T_all-1]))).view(Gy, Gx)
    return {"Gy": Gy, "Gx": Gx, "N": N, "T": T_all, "Cs": Cs, "H": H, "W": W, "preview": preview}


def persist_norm_stats(cache_root: str, ds: RawFireDataset):
    stats_dir = _ensure_dir(os.path.join(cache_root, "_stats"))
    s = {"min": [float(x) for x in ds.landscape_min],
         "max": [float(x) for x in ds.landscape_max]}
    with open(os.path.join(stats_dir, "static_minmax.json"), "w") as f:
        json.dump(s, f, indent=2)
    w = {"ws_min": float(ds.weather_normalizer.min_ws),
         "ws_max": float(ds.weather_normalizer.max_ws),
         "wd_min": float(ds.weather_normalizer.min_wd),
         "wd_max": float(ds.weather_normalizer.max_wd)}
    with open(os.path.join(stats_dir, "weather_minmax.json"), "w") as f:
        json.dump(w, f, indent=2)

# -------------------
# W&B helper (light)
# -------------------
def _maybe_wandb_start(cfg):
    wb = cfg.get("wandb", {})
    pc = cfg.get("patchify_on_disk", {})
    if not (wb.get("enabled", True) and pc.get("progress_log_to_wandb", True)):
        return None, None
    try:
        import wandb
    except Exception:
        return None, None
    run = wandb.init(
        project=wb.get("project", "emberformer"),
        entity=wb.get("entity") or None,
        name=f"build-cache-{time.strftime('%m%d-%H%M')}",
        job_type="build_cache",
        config=cfg,
        mode=wb.get("mode","online"),
    )
    # define counters
    try:
        wandb.define_metric("build/seq_idx")
        wandb.define_metric("build/*", step_metric="build/seq_idx")
    except Exception:
        pass
    return run, wandb

def _maybe_log_artifact(run, cfg, cache_root):
    wb = cfg.get("wandb", {})
    pc = cfg.get("patchify_on_disk", {})
    if run is None or not pc.get("log_artifact", True):
        return
    try:
        import wandb
    except Exception:
        return
    art = wandb.Artifact(
        name=f"patch-cache-{time.strftime('%m%d-%H%M')}",
        type="patch_cache",
        metadata={
            "patch_size": pc["patch_size"],
            "stride": pc.get("stride", 16),
            "pad_mode": pc.get("pad_mode", "strict"),
        },
    )
    art.add_dir(cache_root)
    run.log_artifact(art)


# -----------
# Main entry
# -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    ap.add_argument("--subset", type=int, default=None, help="build only first N sequences")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    dcfg = cfg["data"]; pcfg = cfg["patchify_on_disk"]
    cache_root = _ensure_dir(_expand(pcfg["cache_dir"]))
    logs_dir = _ensure_dir(os.path.join(cache_root, "_logs"))
    progress_path = os.path.join(logs_dir, f"build_progress_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")

    # load raw once (normalizers + indices)
    ds = RawFireDataset(dcfg["data_dir"], sequence_length=dcfg.get("sequence_length", 3))

    # decide which sequences
    # use the unique sequence ids already enumerated by RawFireDataset
    seq_ids = sorted(set(s["sequence_id"] for s in ds.samples), key=lambda x: int(x))
    if args.subset:
        seq_ids = seq_ids[:args.subset]

    # start W&B (optional)
    run, wandb = _maybe_wandb_start(cfg)
    if run:
        run.summary["cache_dir"] = cache_root
        run.summary["num_sequences_planned"] = len(seq_ids)

    # progress bar
    use_bar = bool(pcfg.get("progress_use_tqdm", True))
    iterator = tqdm(seq_ids, desc="patchify", unit="seq") if use_bar else seq_ids

    t0 = time.time()
    totals = defaultdict(float)
    log_every = int(pcfg.get("progress_log_every", 5))
    preview_budget = int(pcfg.get("preview_n", 1))

    print(f"[build] { _now() }  planning {len(seq_ids)} sequences → {cache_root}")

    for idx, seq in enumerate(iterator):
        out_dir = os.path.join(cache_root, f"sequence_{int(seq):03d}")
        meta_path = os.path.join(out_dir, "meta.json")
        start = time.time()

        skipped = False
        error_msg = None
        stats = None

        try:
            if os.path.exists(meta_path):
                skipped = True
                # still count frames & patches by reading meta (cheap)
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                stats = {"T": meta.get("num_frames", 0), "N": meta.get("num_patches", 0)}
            else:
                fire_val = int(cfg.get("encoding", {}).get("fire_value", 231))
                stats = build_one_sequence(ds, seq, out_dir, pcfg, fire_val)

            dt = time.time() - start
            seq_T = int(stats["T"])
            seq_N = int(stats.get("N", stats["Gy"] * stats["Gx"]))
            totals["sequences"] += 1
            totals["frames"] += seq_T
            totals["patches"] += seq_N
            totals["seconds"] += dt

            # JSONL progress row (always)
            _write_jsonl(progress_path, {
                "ts": _now(),
                "seq": int(seq),
                "skipped": skipped,
                "frames": seq_T,
                "patches": seq_N,
                "seconds": round(dt, 3),
                "status": "ok",
            })

            # W&B heartbeat (optional)
            if run and ((idx + 1) % log_every == 0 or idx == 0):
                elapsed = time.time() - t0
                seq_per_min = (idx + 1) / max(1e-9, elapsed / 60.0)
                eta_min = (len(seq_ids) - (idx + 1)) / max(1e-9, seq_per_min)
                wandb.log({
                    "build/seq_idx": idx + 1,
                    "build/sequences_done": idx + 1,
                    "build/frames_done": int(totals["frames"]),
                    "build/patches_done": int(totals["patches"]),
                    "build/elapsed_min": elapsed / 60.0,
                    "build/seq_per_min": seq_per_min,
                    "build/eta_min": eta_min,
                })

            # optional tiny preview (first few sequences)
            if run and preview_budget > 0 and not skipped and "preview" in stats:
                prev = stats["preview"].clamp(0, 1)            # [Gy,Gx], mean fire per patch
                # upscale to ~160px for a quick look
                P = 160 // max(prev.shape)
                P = max(P, 1)
                up = torch.kron(prev, torch.ones((P, P))).numpy()
                wandb.log({"build/preview": wandb.Image((up * 255).astype(np.uint8),
                                                        caption=f"seq {int(seq)} (last fire patch grid)")})
                preview_budget -= 1

            # progress bar description
            if use_bar and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"T={seq_T} N={seq_N} dt={dt:.2f}s")

        except Exception as e:
            error_msg = repr(e)
            _write_jsonl(progress_path, {
                "ts": _now(),
                "seq": int(seq),
                "skipped": False,
                "frames": None,
                "patches": None,
                "seconds": round(time.time() - start, 3),
                "status": "error",
                "error": error_msg,
            })
            print(f"[build][ERROR] seq {seq}: {error_msg}")

    # persist global stats + norm stats
    persist_norm_stats(cache_root, ds)
    totals_out = {
        "finished_at": _now(),
        "sequences": int(totals["sequences"]),
        "frames": int(totals["frames"]),
        "patches": int(totals["patches"]),
        "seconds": round(totals["seconds"], 3),
        "seq_per_min": (totals["sequences"] / max(1e-9, totals["seconds"] / 60.0)),
    }
    with open(os.path.join(logs_dir, "build_totals_last.json"), "w") as f:
        json.dump(totals_out, f, indent=2)
    print(f"[build] done. totals: {totals_out}")

    # log artifact snapshot (optional)
    if run:
        _maybe_log_artifact(run, cfg, cache_root)
        run.finish()

if __name__ == "__main__":
    main()

