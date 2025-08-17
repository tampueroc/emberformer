# scripts/phase0_check.py
import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# Import from top-level 'data' package (matches your current tree)
from data import RawFireDataset, collate_fn

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def maybe_init_wandb(cfg):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", True) or wb.get("mode", "online") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[phase0] wandb not installed; continuing without it.")
        return None

    mode = wb.get("mode", "online")
    if mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project=wb.get("project", "emberformer"),
        entity=wb.get("entity") or None,
        name=wb.get("run_name", "phase0-sanity"),
        tags=wb.get("tags", []),
        notes=wb.get("notes", ""),
        config=cfg,
    )
    return run

def to_numpy_img(t):
    """t: [1,H,W] or [H,W]; returns HxW uint8 for wandb.Image"""
    t = t.detach().cpu()
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    t = (t.float() * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    g = cfg.get("global", {})
    data_cfg = cfg.get("data", {})
    checks = cfg.get("checks", {})
    enc = cfg.get("encoding", {})
    weather_cfg = cfg.get("weather", {})

    torch.manual_seed(g.get("seed", 42))

    # Dataset
    ds = RawFireDataset(
        data_dir=data_cfg["data_dir"],              # NOTE: dataset now uses expanduser(data_dir)
        sequence_length=int(data_cfg.get("sequence_length", 3)),
        transform=None,
    )
    print(f"[phase0] Dataset size: {len(ds)} samples")

    # Quick per-sample scan
    n_scan = int(checks.get("sample_count", 16))
    n_scan = min(n_scan, len(ds))
    shapes = set()
    seq_lens = set()

    # Basic integrity checks
    if checks.get("assert_frames_match", True):
        # If RawFireDataset raised no assertion while building, this is already OK.
        print("[phase0] Frame count check passed during dataset build.")

    # Peek some samples
    for i in range(n_scan):
        fire, static, wind, target = ds[i]
        H, W = fire.shape[1], fire.shape[2]     # fire: [1,H,W,T]
        T = fire.shape[-1]
        shapes.add((H, W))
        seq_lens.add(T)

    print(f"[phase0] Observed spatial sizes (H,W) in first {n_scan}: {sorted(list(shapes))}")
    print(f"[phase0] Observed sequence lengths T in first {n_scan}: {sorted(list(seq_lens))}")

    if checks.get("fail_on_mixed_hw", True) and len(shapes) > 1:
        raise RuntimeError(f"[phase0] Mixed spatial sizes detected: {shapes}")

    # Dataloader + one batch
    dl = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 4)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        collate_fn=collate_fn,
    )

    batch = next(iter(dl))
    fire_b, static_b, wind_b, target_b, valid = batch
    # Shapes
    # fire_b: [B,1,H,W,T_max]
    # static_b: [B,Cs,H,W]
    # wind_b: [B,2,T_max]
    # target_b: [B,1,H,W]
    # valid: [B,T_max]
    print("[phase0] Batch shapes:")
    print("  fire   :", tuple(fire_b.shape))
    print("  static :", tuple(static_b.shape))
    print("  wind   :", tuple(wind_b.shape))
    print("  target :", tuple(target_b.shape))
    print("  valid  :", tuple(valid.shape))

    # W&B logging (optional)
    run = maybe_init_wandb(cfg)
    if run is not None:
        import wandb

        # Log min/max per landscape band if available
        if cfg.get("wandb", {}).get("log_norm_stats", True):
            ln = ds.landscape_normalizer
            if getattr(ln, "landscape_min", None) is not None:
                run.log({
                    "landscape/min_per_band": wandb.Histogram(ln.landscape_min),
                    "landscape/max_per_band": wandb.Histogram(ln.landscape_max),
                })

        # Batch preview (first N)
        if cfg.get("wandb", {}).get("log_batch_preview", True):
            n_img = min(cfg["wandb"].get("max_preview_images", 8), fire_b.size(0))
            images = []
            for i in range(n_img):
                # last time slice of fire & target for quick visual
                fire_last = fire_b[i, 0, :, :, -1]     # [H,W]
                targ = target_b[i, 0]                  # [H,W]
                images.append(wandb.Image(to_numpy_img(fire_last), caption=f"fire_last/b{i}"))
                images.append(wandb.Image(to_numpy_img(targ), caption=f"target/b{i}"))
            run.log({"preview": images})

        # Basic scalars
        run.log({
            "batch/T_max": fire_b.shape[-1],
            "batch/H": fire_b.shape[2],
            "batch/W": fire_b.shape[3],
        })

        run.finish()

    print("[phase0] ✅ Data layer sanity check completed.")

if __name__ == "__main__":
    main()

