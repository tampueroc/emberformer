import os
import argparse
import random
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

# local imports
from emberformer.data import RawFireDataset, collate_fn

try:
    import wandb  # optional
except Exception:
    wandb = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_wandb_preview(cfg, batch, landscape_minmax=None):
    if not cfg["wandb"]["enabled"] or cfg["wandb"]["mode"] == "disabled":
        return
    if wandb is None:
        print("[WARN] wandb not installed; skipping W&B logging.")
        return

    mode = cfg["wandb"]["mode"]
    kwargs = {"project": cfg["wandb"]["project"], "name": cfg["wandb"]["run_name"],
              "tags": cfg["wandb"]["tags"], "notes": cfg["wandb"]["notes"]}
    if cfg["wandb"]["entity"]:
        kwargs["entity"] = cfg["wandb"]["entity"]
    if mode in ("online", "offline"):
        os.environ["WANDB_MODE"] = mode
    wandb.init(**kwargs)

    fire, static, wind, target, valid = batch
    B, _, H, W, T = fire.shape
    max_imgs = min(cfg["wandb"]["max_preview_images"], B)

    # Log basic shapes
    wandb.log({
        "batch/B": B, "batch/H": H, "batch/W": W, "batch/T": T,
        "static/Cs": static.shape[1]
    })

    # Make a simple preview: last fire frame vs target
    previews = []
    for i in range(max_imgs):
        last_valid_t = int(valid[i].nonzero(as_tuple=False).max().item())
        fire_last = fire[i, 0, :, :, last_valid_t].cpu().float().numpy()
        tgt = target[i, 0].cpu().float().numpy()
        previews.append(wandb.Image(
            np.stack([fire_last, tgt], axis=0),
            caption=f"sample={i} | rows: [fire_last, target]"
        ))
    wandb.log({"preview/fire_last_and_target": previews})

    # Static band min/max
    if cfg["wandb"]["log_norm_stats"] and landscape_minmax is not None:
        lmin, lmax = landscape_minmax
        table = wandb.Table(columns=["band", "min", "max"])
        for i, (mn, mx) in enumerate(zip(lmin, lmax)):
            table.add_data(i, float(mn), float(mx))
        wandb.log({"landscape/minmax": table})


def run_checks(cfg, ds: RawFireDataset):
    print("\n[Phase-0] Running quick dataset checks…")

    # 1) Weather columns present
    if cfg["checks"]["assert_ws_wd_present"]:
        sample_key = next(iter(ds.weathers))  # grab any one file
        df = ds.weathers[sample_key]
        assert cfg["weather"]["ws_col"] in df.columns and cfg["weather"]["wd_col"] in df.columns, \
            f"Weather CSVs must contain columns '{cfg['weather']['ws_col']}' and '{cfg['weather']['wd_col']}'"
        print(" ✓ Weather columns present.")

    # 2) Frame count parity (already asserted in dataset, but sample a few)
    if cfg["checks"]["assert_frames_match"]:
        root = os.path.join(os.path.expanduser(cfg["data"]["data_dir"]))
        fire_root = os.path.join(root, "fire_frames")
        iso_root = os.path.join(root, "isochrones")
        seqs = sorted(os.listdir(fire_root))[: cfg["checks"]["sample_count"]]
        for sd in seqs:
            nf = len([f for f in os.listdir(os.path.join(fire_root, sd)) if f.endswith(".png")])
            ni = len([f for f in os.listdir(os.path.join(iso_root, sd)) if f.endswith(".png")])
            assert nf == ni, f"{sd}: fire({nf}) != iso({ni})"
        print(f" ✓ Fire/iso frames match for {len(seqs)} sampled sequences.")

    # 3) H/W consistency across samples
    if cfg["checks"]["fail_on_mixed_hw"]:
        H0 = W0 = None
        for i in range(min(cfg["checks"]["sample_count"], len(ds))):
            fire, static, _, tgt = ds[i]
            _, H, W, _ = fire.shape
            if H0 is None:
                H0, W0 = H, W
            assert (H, W) == (H0, W0), f"Mixed spatial sizes found: got {(H, W)} vs {(H0, W0)}"
        print(f" ✓ Spatial size consistent at {H0}x{W0} (checked {min(cfg['checks']['sample_count'], len(ds))} samples).")

    # 4) Landscape normalization stats
    lmin = ds.landscape_min
    lmax = ds.landscape_max
    print(f" ✓ Landscape bands: {len(lmin)} | min[0]={float(lmin[0]) if len(lmin)>0 else '—'} | max[0]={float(lmax[0]) if len(lmax)>0 else '—'}")
    return (lmin, lmax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="emberformer/configs/emberformer.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["global"]["seed"])
    torch.backends.cudnn.benchmark = bool(cfg["global"]["cudnn_benchmark"])

    # Dataset + Loader
    ds = RawFireDataset(
        data_dir=cfg["data"]["data_dir"],
        sequence_length=int(cfg["data"]["sequence_length"]),
        transform=None,
    )

    # quick checks
    landscape_minmax = run_checks(cfg, ds)

    loader = DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=bool(cfg["data"]["shuffle"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=bool(cfg["data"]["drop_last"]),
        collate_fn=collate_fn,
    )

    # Pull one batch
    batch = next(iter(loader))
    fire, static, wind, target, valid = batch

    # Print shapes
    print("\n[Batch Shapes]")
    print(f" fire      : {tuple(fire.shape)}   (B, 1, H, W, T_max)")
    print(f" static    : {tuple(static.shape)} (B, Cs, H, W)")
    print(f" wind      : {tuple(wind.shape)}   (B, 2, T_max)")
    print(f" target    : {tuple(target.shape)} (B, 1, H, W)")
    print(f" valid     : {tuple(valid.shape)}  (B, T_max)")

    # Basic assertions
    B, _, H, W, T = fire.shape
    assert wind.shape[-1] == T, "Wind T_max must match fire T_max after collate."
    assert static.shape[2] == H and static.shape[3] == W, "Static H/W must match fire H/W."

    # W&B preview (optional)
    if cfg["wandb"]["enabled"] and cfg["wandb"]["mode"] != "disabled":
        log_wandb_preview(cfg, batch, landscape_minmax)

    print("\n[OK] Phase-0 dataloader sanity check complete.")


if __name__ == "__main__":
    main()

