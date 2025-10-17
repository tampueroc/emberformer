"""
Training script for EmberFormer (temporal transformer + spatial decoder)

Mirrors train_stage_c_tokens.py but uses:
- TokenFireDataset with use_history=True
- collate_tokens_temporal for variable-length sequences
- EmberFormer model (temporal transformer + SegFormer/UNet decoder)
- Differential learning rates for temporal vs spatial components
"""
import os, time, argparse, yaml, pathlib, json, random
from collections import defaultdict
import torch
import torch.nn as nn
from torch import amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from data import TokenFireDataset, collate_tokens_temporal
from models import EmberFormer
from utils import init_wandb, define_common_metrics, save_artifact, log_grid_preview

# ----------------
# Config helpers
# ----------------
def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _pick_device(gpu_arg: int | None):
    if torch.cuda.is_available():
        if gpu_arg is None:
            return torch.device("cuda")
        n = torch.cuda.device_count()
        if gpu_arg < 0 or gpu_arg >= n:
            gpu_arg = 0
        torch.cuda.set_device(gpu_arg)
        return torch.device(f"cuda:{gpu_arg}")
    return torch.device("cpu")

def _coerce_float(val, name):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            raise ValueError(f"Config field '{name}' must be a number or string float, got: {val!r}")
    if val is None:
        return None
    raise ValueError(f"Config field '{name}' has unsupported type: {type(val)}")

# ----------------
# Batch shaping
# ----------------
def _extract_grid_hw(metas):
    """Extract grid dimensions from meta (list of tensors or single tensor)"""
    if isinstance(metas, list):
        # List of [2] tensors
        first = metas[0]
        gH, gW = int(first[0].item()), int(first[1].item())
    elif torch.is_tensor(metas):
        if metas.ndim == 2:
            gH, gW = int(metas[0, 0].item()), int(metas[0, 1].item())
        elif metas.ndim == 1 and metas.numel() == 2:
            gH, gW = int(metas[0].item()), int(metas[1].item())
        else:
            raise TypeError(f"Unexpected metas tensor shape: {tuple(metas.shape)}")
    else:
        raise TypeError(f"Unexpected meta type: {type(metas)}")
    return gH, gW

# ----------------
# Loss / metrics
# ----------------
def _masked_bce(logits, y, mask, pos_weight=None):
    """
    logits, y: same spatial size (grid or pixel); mask: boolean or {0,1}
    """
    loss_map = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    loss_map = loss_map * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss_map.sum() / denom

def _auto_pos_weight(y_map, mask):
    """
    y_map in [0,1], mask boolean; returns scalar tensor ~ (neg_mass / pos_mass).
    """
    y = (y_map * mask).float()
    pos_mass = y.sum().clamp_min(1.0)
    neg_mass = (mask.float().sum() - pos_mass).clamp_min(1.0)
    return (neg_mass / pos_mass)

# ----------------
# Deterministic split helpers
# ----------------
def _split_file_path(cache_root: str, seed: int) -> str:
    split_dir = os.path.join(cache_root, "_splits")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"emberformer-seq-split-seed{seed}.json")

def _make_or_load_sequence_split(ds, cache_root: str, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    path = _split_file_path(cache_root, seed)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f), path

    seqs = sorted(ds.seq_dirs)
    rng = random.Random(seed)
    rng.shuffle(seqs)

    n = len(seqs)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = max(0, n - n_train - n_val)

    split = {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "counts":   {"train": n_train, "val": n_val, "test": n_test, "total": n},
        "train": seqs[:n_train],
        "val":   seqs[n_train:n_train + n_val],
        "test":  seqs[n_train + n_val:],
    }
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    return split, path

def _subset_indices_by_sequence(ds, seq_list):
    seq_to_idxs = defaultdict(list)
    for idx, (sd, _t) in enumerate(ds.samples):
        seq_to_idxs[sd].append(idx)
    out = []
    for sd in seq_list:
        out.extend(seq_to_idxs.get(sd, []))
    return sorted(out)

# ----------------
# Main
# ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr_temporal", type=float, default=None)
    ap.add_argument("--lr_spatial", type=float, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _pick_device(args.gpu)
    torch.manual_seed(cfg.get("global", {}).get("seed", 42))
    torch.backends.cudnn.benchmark = cfg.get("global", {}).get("cudnn_benchmark", True)

    # Data cfg
    d = cfg["data"]
    pcfg = cfg.get("patchify_on_disk", {})
    cache_root = os.path.expanduser(args.cache_dir or pcfg["cache_dir"])
    raw_root   = os.path.expanduser(d["data_dir"])
    P = int(pcfg.get("patch_size", 16))

    # Dataset with temporal history
    ds = TokenFireDataset(
        cache_root, raw_root=raw_root,
        sequence_length=d.get("sequence_length", 3),
        fire_value=cfg.get("encoding", {}).get("fire_value", 231),
        use_history=True  # NEW: enable temporal history
    )

    # Peek one sample for channel count
    X0, _ = ds[0]
    Cs = X0["static"].shape[1]

    # W&B
    run = init_wandb(cfg, context={"script": pathlib.Path(__file__).stem})
    if run:
        define_common_metrics()

    # Metric/imbalance config
    mcfg = cfg.get("model", {}).get("metrics", {})
    pred_thresh   = float(mcfg.get("pred_thresh", 0.5))
    target_thresh = float(mcfg.get("target_thresh", 0.05))
    posw_cfg      = mcfg.get("pos_weight", "auto")

    # Deterministic split
    split_cfg = cfg.get("split", {})
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac   = float(split_cfg.get("val",   0.1))
    test_frac  = float(split_cfg.get("test",  0.1))
    seed_split = int(split_cfg.get("seed", cfg.get("global", {}).get("seed", 42)))

    split, split_path = _make_or_load_sequence_split(ds, cache_root, train_frac, val_frac, test_frac, seed_split)
    train_idxs = _subset_indices_by_sequence(ds, split["train"])
    val_idxs   = _subset_indices_by_sequence(ds, split["val"])
    test_idxs  = _subset_indices_by_sequence(ds, split["test"])

    train_set = torch.utils.data.Subset(ds, train_idxs)
    val_set   = torch.utils.data.Subset(ds, val_idxs)

    bs = args.batch_size or d.get("batch_size", 8)
    dl_train = DataLoader(
        train_set, batch_size=bs, shuffle=True,
        num_workers=d.get("num_workers", 4),
        pin_memory=d.get("pin_memory", True),
        drop_last=d.get("drop_last", False),
        collate_fn=collate_tokens_temporal  # NEW: temporal collate
    )
    dl_val = DataLoader(
        val_set, batch_size=bs, shuffle=False,
        num_workers=d.get("num_workers", 4),
        pin_memory=d.get("pin_memory", True),
        drop_last=False,
        collate_fn=collate_tokens_temporal  # NEW: temporal collate
    )

    if run:
        gh0, gw0 = X0["meta"]
        run.summary.update({
            "dataset_size": len(ds),
            "grid_h": gh0,
            "grid_w": gw0,
            "static_channels": Cs,
            "model_name": "EmberFormer",
            "split_file": split_path,
            "split/train_seqs": len(split["train"]),
            "split/val_seqs":   len(split["val"]),
            "split/test_seqs":  len(split["test"]),
            "train_samples":    len(train_set),
            "val_samples":      len(val_set),
            "test_samples":     len(test_idxs),
        })

    # Model config
    mcfg = cfg.get("model", {})
    tcfg = mcfg.get("temporal", {})
    scfg = mcfg.get("spatial", {})
    
    model = EmberFormer(
        d_model=int(tcfg.get("d_model", 64)),
        static_channels=Cs,
        nhead=int(tcfg.get("nhead", 4)),
        num_layers=int(tcfg.get("num_layers", 3)),
        dim_feedforward=int(tcfg.get("dim_feedforward", 256)),
        dropout=float(tcfg.get("dropout", 0.1)),
        spatial_decoder=scfg.get("decoder_type", "segformer"),
        spatial_base_channels=int(scfg.get("base_channels", 32)),
        segformer_pretrained=scfg.get("pretrained", True),
        segformer_model=scfg.get("model_name", "nvidia/segformer-b0-finetuned-ade-512-512"),
        freeze_decoder=scfg.get("freeze_initially", False),
        patch_size=P,
        use_wind=tcfg.get("use_wind", True),
        use_static=tcfg.get("use_static", True),
        max_seq_len=int(tcfg.get("max_seq_len", 32)),
    ).to(device)

    # Optimizer with differential learning rates
    train_cfg = cfg["train"]
    lr_temporal = _coerce_float(args.lr_temporal, "cli.lr_temporal") or _coerce_float(train_cfg.get("lr_temporal", 1e-3), "train.lr_temporal")
    lr_spatial = _coerce_float(args.lr_spatial, "cli.lr_spatial") or _coerce_float(train_cfg.get("lr_spatial", 1e-4), "train.lr_spatial")
    wd = _coerce_float(train_cfg.get("weight_decay", 1e-4), "train.weight_decay")
    
    # Group parameters: temporal (embedder + transformer) vs spatial (decoder)
    temporal_params = list(model.token_embedder.parameters()) + list(model.temporal_transformer.parameters())
    spatial_params = list(model.spatial_decoder.parameters())
    
    opt = torch.optim.Adam([
        {'params': temporal_params, 'lr': lr_temporal},
        {'params': spatial_params, 'lr': lr_spatial}
    ], weight_decay=wd)
    
    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # torchmetrics (PIXEL space)
    def _make_metrics():
        return dict(
            acc  = torchmetrics.classification.BinaryAccuracy().to(device),
            prec = torchmetrics.classification.BinaryPrecision().to(device),
            rec  = torchmetrics.classification.BinaryRecall().to(device),
            f1   = torchmetrics.classification.BinaryF1Score().to(device),
            iou  = torchmetrics.classification.BinaryJaccardIndex().to(device),
        )
    Mtr = _make_metrics()
    Mva = _make_metrics()

    step = 0
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 20)
    log_imgs_every = int(train_cfg.get("log_images_every", 200))
    preview_size = int(cfg["wandb"].get("preview_size", 160))
    freeze_epochs = int(scfg.get("freeze_epochs", 0))

    for ep in range(epochs):
        # Unfreeze decoder after N epochs
        if ep == freeze_epochs and freeze_epochs > 0:
            print(f"[Epoch {ep}] Unfreezing spatial decoder")
            model.unfreeze_decoder()

        # ----- TRAIN -----
        model.train()
        for batch in dl_train:
            X_batch, y_batch = batch
            
            # Move to device
            fire_hist = X_batch["fire_hist"].to(device, non_blocking=True)
            wind_hist = X_batch["wind_hist"].to(device, non_blocking=True)
            static = X_batch["static"].to(device, non_blocking=True)
            valid = X_batch["valid"].to(device, non_blocking=True)
            valid_t = X_batch["valid_t"].to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            # Extract grid shape
            gH, gW = _extract_grid_hw(X_batch["meta"])
            
            opt.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=(device.type == "cuda")):
                # Forward: get patch-level logits [B, 1, Gy, Gx]
                logits_grid = model(
                    fire_hist, wind_hist, static, valid_t, grid_shape=(gH, gW)
                )
                
                # Unpatchify to pixel space
                logits_pix = F.interpolate(logits_grid, scale_factor=P, mode="nearest")
                y_map = y_batch.view(-1, 1, gH, gW)
                y_pix = F.interpolate(y_map, scale_factor=P, mode="nearest")
                mask = valid.view(-1, 1, gH, gW)
                mask_pix = F.interpolate(mask.float(), scale_factor=P, mode="nearest").bool()
                
                # Pos weight
                if posw_cfg == "auto":
                    pos_weight = _auto_pos_weight(y_pix, mask_pix).detach().to(device)
                elif isinstance(posw_cfg, (int, float)):
                    pos_weight = torch.tensor(float(posw_cfg), device=device)
                else:
                    pos_weight = None
                
                loss = _masked_bce(logits_pix, y_pix, mask_pix, pos_weight)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Pixel-space metrics
            with torch.no_grad():
                probs_pix = torch.sigmoid(logits_pix.float())
                preds_bin_pix = (probs_pix > pred_thresh).int()
                target_bin_pix = (y_pix > target_thresh).int()
                m = mask_pix
                p = preds_bin_pix[m].flatten()
                yb = target_bin_pix[m].flatten()
                Mtr["acc"].update(p, yb)
                Mtr["prec"].update(p, yb)
                Mtr["rec"].update(p, yb)
                Mtr["f1"].update(p, yb)
                Mtr["iou"].update(p, yb)

            if run:
                run.log({"epoch": ep, "step": step, "train/loss": loss.item()}, step=step)
            step += 1

        train_metrics = {f"train/{k}": v.compute().item() for k, v in Mtr.items()}
        for v in Mtr.values(): v.reset()

        # ----- VALIDATION -----
        model.eval()
        val_loss_total = 0.0
        nb = 0
        logged_preview = False
        with torch.no_grad():
            for batch in dl_val:
                X_batch, y_batch = batch
                
                fire_hist = X_batch["fire_hist"].to(device, non_blocking=True)
                wind_hist = X_batch["wind_hist"].to(device, non_blocking=True)
                static = X_batch["static"].to(device, non_blocking=True)
                valid = X_batch["valid"].to(device, non_blocking=True)
                valid_t = X_batch["valid_t"].to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                gH, gW = _extract_grid_hw(X_batch["meta"])
                
                with amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits_grid = model(
                        fire_hist, wind_hist, static, valid_t, grid_shape=(gH, gW)
                    )
                    logits_pix = F.interpolate(logits_grid, scale_factor=P, mode="nearest")
                    y_map = y_batch.view(-1, 1, gH, gW)
                    y_pix = F.interpolate(y_map, scale_factor=P, mode="nearest")
                    mask = valid.view(-1, 1, gH, gW)
                    mask_pix = F.interpolate(mask.float(), scale_factor=P, mode="nearest").bool()
                    vloss = _masked_bce(logits_pix, y_pix, mask_pix, None)
                
                val_loss_total += float(vloss)
                nb += 1

                probs_pix = torch.sigmoid(logits_pix.float())
                preds_bin_pix = (probs_pix > pred_thresh).int()
                target_bin_pix = (y_pix > target_thresh).int()
                m = mask_pix
                p = preds_bin_pix[m].flatten()
                yb = target_bin_pix[m].flatten()
                Mva["acc"].update(p, yb)
                Mva["prec"].update(p, yb)
                Mva["rec"].update(p, yb)
                Mva["f1"].update(p, yb)
                Mva["iou"].update(p, yb)

                # Log preview
                if run and (not logged_preview) and (ep % log_imgs_every == 0 or ep == epochs - 1):
                    log_grid_preview(run, probs_pix[:2], y_pix[:2], key="val/preview", size=preview_size)
                    logged_preview = True

        val_loss = val_loss_total / max(nb, 1)
        val_metrics = {f"val/{k}": v.compute().item() for k, v in Mva.items()}
        for v in Mva.values(): v.reset()

        # Log epoch summary
        all_metrics = {"epoch": ep, "val/loss": val_loss, **train_metrics, **val_metrics}
        print(f"[Epoch {ep:2d}] loss={val_loss:.4f} | "
              f"train f1={train_metrics['train/f1']:.3f} | "
              f"val f1={val_metrics['val/f1']:.3f} | "
              f"val iou={val_metrics['val/iou']:.3f}")
        
        if run:
            run.log(all_metrics, step=step)

    print("Training complete!")
    if run:
        # Save final model
        save_path = f"emberformer_ep{epochs}.pt"
        torch.save(model.state_dict(), save_path)
        save_artifact(run, save_path, name=f"emberformer-{run.id}", type="model")
        run.finish()

if __name__ == "__main__":
    main()
