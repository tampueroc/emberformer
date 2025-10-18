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
from tqdm import tqdm

from data import TokenFireDataset, collate_tokens_temporal
from models import EmberFormer
from utils import init_wandb, define_common_metrics, save_artifact, log_grid_preview

# ----------------
# Early Stopping
# ----------------
class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (F1, IoU), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch, value):
        """
        Returns True if should stop training
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def state_dict(self):
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
        }

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

def _masked_soft_dice_loss(logits, y, mask, smooth=1.0, ignore_empty=True):
    """
    Soft Dice loss over masked pixels. Optimizes for F1/IoU directly.
    
    Scaled to match BCE magnitude (~0.1-0.3 range) for balanced weighting.
    
    Args:
        logits: [B, 1, H, W] raw logits
        y: [B, 1, H, W] targets in [0, 1]
        mask: [B, 1, H, W] boolean mask
        smooth: smoothing term to avoid division by zero
        ignore_empty: if True, samples with no positives don't contribute
    
    Returns:
        scalar Dice loss (scaled to ~BCE magnitude)
    """
    p = torch.sigmoid(logits).float()
    y = y.float()
    m = mask.float()
    
    # Apply mask
    p = p * m
    y = y * m
    
    # Compute per-sample Dice
    dims = (1, 2, 3)  # Sum over C, H, W
    intersection = (p * y).sum(dims)
    p_sum = p.sum(dims)
    y_sum = y.sum(dims)
    
    dice = (2.0 * intersection + smooth) / (p_sum + y_sum + smooth)
    
    if ignore_empty:
        # Only average over samples that have positive pixels
        valid = (y_sum > 0)
        if valid.any():
            dice_loss = (1.0 - dice[valid]).mean()
        else:
            # No positive pixels in batch → return zero loss
            return torch.zeros([], device=logits.device, dtype=logits.dtype)
    else:
        dice_loss = (1.0 - dice).mean()
    
    # Scale Dice to match BCE magnitude
    # Dice range: [0, 1], BCE range: ~[0.05, 0.3]
    # Apply square root to compress high values: sqrt(0.8) = 0.89 → 0.3 is more comparable to BCE
    return torch.sqrt(dice_loss + 1e-7)

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

    # Metric/loss config
    mcfg = cfg.get("model", {}).get("metrics", {})
    lcfg = cfg.get("model", {}).get("loss", {})
    pred_thresh   = float(mcfg.get("pred_thresh", 0.5))
    target_thresh = float(mcfg.get("target_thresh", 0.05))
    posw_cfg      = mcfg.get("pos_weight", "auto")
    
    # Loss configuration
    use_dice = lcfg.get("use_dice", True)
    bce_weight = float(lcfg.get("bce_weight", 0.5))
    dice_weight = float(lcfg.get("dice_weight", 0.5))

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
    log_imgs_every = int(train_cfg.get("log_images_every", 1))  # Log every epoch by default
    log_metrics_every = int(train_cfg.get("log_metrics_every", 50))  # Log metrics every N steps
    preview_size = int(cfg["wandb"].get("preview_size", 400))
    max_preview_images = int(cfg["wandb"].get("max_preview_images", 2))
    freeze_epochs = int(scfg.get("freeze_epochs", 0))
    
    # Early stopping
    early_stop_cfg = train_cfg.get("early_stopping", {})
    if early_stop_cfg.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=int(early_stop_cfg.get("patience", 10)),
            min_delta=float(early_stop_cfg.get("min_delta", 0.001)),
            mode=early_stop_cfg.get("mode", "max")  # 'max' for F1/IoU, 'min' for loss
        )
        monitor_metric = early_stop_cfg.get("monitor", "val/f1")
    else:
        early_stopper = None
    
    # Checkpoint directory
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = None
    save_checkpoints = train_cfg.get("save_checkpoints", True)

    print("\n" + "="*60)
    print(f"Starting Training: {epochs} epochs")
    print(f"  Model: EmberFormer ({model.decoder_type} decoder)")
    print(f"  Train samples: {len(train_set)}")
    print(f"  Val samples: {len(val_set)}")
    print(f"  Batch size: {bs}")
    print(f"  LR temporal: {lr_temporal:.2e}, LR spatial: {lr_spatial:.2e}")
    if use_dice:
        print(f"  Loss: BCE ({bce_weight}) + Dice ({dice_weight})")
    else:
        print(f"  Loss: BCE only (pos_weight={posw_cfg})")
    if early_stopper:
        print(f"  Early stopping: enabled (patience={early_stopper.patience}, monitor={monitor_metric})")
    if run:
        print(f"  W&B run: {run.name} ({run.id})")
        print(f"  W&B url: {run.url}")
    print("="*60 + "\n")

    for ep in range(epochs):
        # Unfreeze decoder after N epochs
        if ep == freeze_epochs and freeze_epochs > 0:
            print(f"\n[Epoch {ep}] Unfreezing spatial decoder\n")
            model.unfreeze_decoder()

        # ----- TRAIN -----
        model.train()
        train_loss_accum = 0.0
        batch_t_lengths = []  # Track sequence lengths
        pbar = tqdm(dl_train, desc=f"Epoch {ep+1}/{epochs} [Train]", leave=False)
        for batch in pbar:
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
                
                # Combined loss: BCE + Dice (configured from yaml)
                if use_dice:
                    # Use pos_weight with Dice when precision is too low
                    if posw_cfg == "auto":
                        pos_weight = _auto_pos_weight(y_pix, mask_pix).detach().to(device)
                    elif isinstance(posw_cfg, (int, float)):
                        pos_weight = torch.tensor(float(posw_cfg), device=device)
                    else:
                        pos_weight = None
                    
                    bce = _masked_bce(logits_pix, y_pix, mask_pix, pos_weight=pos_weight)
                    dice = _masked_soft_dice_loss(logits_pix, y_pix, mask_pix)
                    loss = bce_weight * bce + dice_weight * dice
                else:
                    # Fallback to BCE only with pos_weight
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

            # Update progress bar
            train_loss_accum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Track sequence length statistics
            batch_t_len = valid_t.sum(dim=1).float().mean().item()
            batch_t_lengths.append(batch_t_len)

            # Log to W&B
            if run:
                log_dict = {
                    "epoch": ep, 
                    "train/loss_step": loss.item(),
                    "train/batch_T_avg": batch_t_len,
                }
                
                # Log individual loss components
                if use_dice:
                    log_dict["train/bce_step"] = bce.item()
                    log_dict["train/dice_step"] = dice.item()
                
                # Log step-level metrics every N steps (not every step to reduce overhead)
                if step % log_metrics_every == 0:
                    # Compute current metrics (don't reset)
                    step_metrics = {
                        f"train/{k}_step": v.compute().item() 
                        for k, v in Mtr.items()
                    }
                    log_dict.update(step_metrics)
                
                run.log(log_dict, step=step)
            step += 1

        train_metrics = {f"train/{k}": v.compute().item() for k, v in Mtr.items()}
        for v in Mtr.values(): v.reset()
        avg_train_loss = train_loss_accum / len(dl_train)

        # ----- VALIDATION -----
        model.eval()
        val_loss_total = 0.0
        nb = 0
        logged_preview = False
        pbar_val = tqdm(dl_val, desc=f"Epoch {ep+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in pbar_val:
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
                    
                    # Use same loss as training
                    if use_dice:
                        bce_val = _masked_bce(logits_pix, y_pix, mask_pix, None)
                        dice_val = _masked_soft_dice_loss(logits_pix, y_pix, mask_pix)
                        vloss = bce_weight * bce_val + dice_weight * dice_val
                    else:
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

                # Update val progress bar
                pbar_val.set_postfix({"loss": f"{vloss.item():.4f}"})

                # Log validation preview (first batch only, every epoch)
                if run and (not logged_preview):
                    log_grid_preview(
                        run, probs_pix[:max_preview_images], y_pix[:max_preview_images], 
                        key="val/preview", max_images=max_preview_images, size=preview_size
                    )
                    logged_preview = True

        val_loss = val_loss_total / max(nb, 1)
        val_metrics = {f"val/{k}": v.compute().item() for k, v in Mva.items()}
        for v in Mva.values(): v.reset()

        # Log epoch summary to console
        loss_str = f"BCE+Dice" if use_dice else "BCE"
        print(f"[Epoch {ep+1:2d}/{epochs}] {loss_str} | "
              f"train: loss={avg_train_loss:.4f} f1={train_metrics['train/f1']:.3f} prec={train_metrics['train/prec']:.3f} | "
              f"val: loss={val_loss:.4f} f1={val_metrics['val/f1']:.3f} prec={val_metrics['val/prec']:.3f} "
              f"iou={val_metrics['val/iou']:.3f}")
        
        # Compute sequence length statistics for this epoch
        avg_t_len = sum(batch_t_lengths) / len(batch_t_lengths) if batch_t_lengths else 0
        
        # Log all metrics to W&B
        all_metrics = {
            "epoch": ep, 
            "train/loss_epoch": avg_train_loss,
            "train/T_avg_epoch": avg_t_len,
            "val/loss_epoch": val_loss, 
        }
        
        # Add epoch-level train metrics (with _epoch suffix for clarity)
        for k, v in train_metrics.items():
            all_metrics[f"{k}_epoch"] = v
        
        # Add epoch-level val metrics (with _epoch suffix for clarity)
        for k, v in val_metrics.items():
            all_metrics[f"{k}_epoch"] = v
        
        if run:
            run.log(all_metrics, step=step)
            
            # Log current learning rates
            run.log({
                "train/lr_temporal": opt.param_groups[0]['lr'],
                "train/lr_spatial": opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else opt.param_groups[0]['lr'],
            }, step=step)
        
        # Save best model checkpoint
        if save_checkpoints and early_stopper:
            # Extract metric value to monitor
            if monitor_metric == "val/loss":
                metric_value = val_loss
            elif monitor_metric == "val/f1":
                metric_value = val_metrics['val/f1']
            elif monitor_metric == "val/iou":
                metric_value = val_metrics['val/iou']
            else:
                metric_value = val_metrics['val/f1']
            
            # Check if this is the best epoch
            is_best = (early_stopper.best_value is None or 
                      (early_stopper.mode == 'max' and metric_value > early_stopper.best_value) or
                      (early_stopper.mode == 'min' and metric_value < early_stopper.best_value))
            
            if is_best:
                # Save best model
                best_model_path = os.path.join(ckpt_dir, f"emberformer_best.pt")
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_f1': val_metrics['val/f1'],
                    'val_iou': val_metrics['val/iou'],
                    'val_prec': val_metrics['val/prec'],
                    'val_rec': val_metrics['val/rec'],
                    'val_acc': val_metrics['val/acc'],
                    'val_loss': val_loss,
                    'config': cfg,
                }, best_model_path)
                print(f"   💾 Saved best model (epoch {ep+1}, {monitor_metric}={metric_value:.4f})")
        
        # Early stopping check
        if early_stopper:
            # Extract metric value (same as above)
            if monitor_metric == "val/loss":
                metric_value = val_loss
            elif monitor_metric == "val/f1":
                metric_value = val_metrics['val/f1']
            elif monitor_metric == "val/iou":
                metric_value = val_metrics['val/iou']
            else:
                metric_value = val_metrics['val/f1']
            
            should_stop = early_stopper(ep, metric_value)
            
            if run:
                run.log({
                    "early_stop/counter": early_stopper.counter,
                    "early_stop/best_value": early_stopper.best_value,
                    "early_stop/best_epoch": early_stopper.best_epoch,
                }, step=step)
            
            if should_stop:
                print(f"\n🛑 Early stopping triggered!")
                print(f"   Best {monitor_metric}: {early_stopper.best_value:.4f} at epoch {early_stopper.best_epoch}")
                print(f"   No improvement for {early_stopper.patience} epochs")
                break

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    if run:
        # Save final model
        final_path = os.path.join(ckpt_dir, f"emberformer_final_ep{ep+1}.pt")
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'val_f1': val_metrics['val/f1'],
            'val_iou': val_metrics['val/iou'],
            'val_prec': val_metrics['val/prec'],
            'val_rec': val_metrics['val/rec'],
            'val_acc': val_metrics['val/acc'],
            'val_loss': val_loss,
            'config': cfg,
        }, final_path)
        print(f"💾 Saved final model: {final_path}")
        
        if best_model_path:
            print(f"💾 Best model saved at: {best_model_path}")
            best_checkpoint = torch.load(best_model_path)
            save_artifact(run, best_model_path, 
                         name=f"emberformer-best-{run.id}", 
                         type="model",
                         metadata={
                             'epoch': best_checkpoint['epoch'],
                             'val_f1': float(best_checkpoint['val_f1']),
                             'val_iou': float(best_checkpoint['val_iou']),
                             'val_prec': float(best_checkpoint['val_prec']),
                             'val_rec': float(best_checkpoint['val_rec']),
                             'val_acc': float(best_checkpoint['val_acc']),
                             'val_loss': float(best_checkpoint['val_loss']),
                             'type': 'best_model',
                             'monitor_metric': monitor_metric if early_stopper else 'none',
                         })
        
        final_checkpoint = torch.load(final_path)
        save_artifact(run, final_path, 
                     name=f"emberformer-final-{run.id}", 
                     type="model",
                     metadata={
                         'epoch': final_checkpoint['epoch'],
                         'val_f1': float(final_checkpoint['val_f1']),
                         'val_iou': float(final_checkpoint['val_iou']),
                         'val_prec': float(final_checkpoint['val_prec']),
                         'val_rec': float(final_checkpoint['val_rec']),
                         'val_acc': float(final_checkpoint['val_acc']),
                         'val_loss': float(final_checkpoint['val_loss']),
                         'type': 'final_model',
                     })
        run.finish()

if __name__ == "__main__":
    main()
