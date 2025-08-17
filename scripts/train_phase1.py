import os
import yaml
import math
import wandb
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

from data import RawFireDataset, collate_fn
from models import CopyLast, ConvHead2D, Tiny3D

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def make_model(kind, static_channels):
    if kind == "copy_last":
        return CopyLast()
    if kind == "conv2d":
        return ConvHead2D(static_channels=static_channels)
    if kind == "tiny3d":
        return Tiny3D()
    raise ValueError(f"Unknown model kind: {kind}")

@torch.no_grad()
def compute_metrics(logits, target, thr=0.5):
    # logits/target: [B,1,H,W]
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    tgt = target.float()
    inter = (pred * tgt).sum(dim=(1,2,3))
    union = (pred + tgt - pred*tgt).sum(dim=(1,2,3))
    iou = (inter / (union + 1e-6)).mean().item()
    tp = inter
    fp = (pred * (1 - tgt)).sum(dim=(1,2,3))
    fn = ((1 - pred) * tgt).sum(dim=(1,2,3))
    prec = (tp / (tp + fp + 1e-6)).mean().item()
    rec  = (tp / (tp + fn + 1e-6)).mean().item()
    f1 = (2*prec*rec / (prec + rec + 1e-6)) if (prec+rec)>0 else 0.0
    return dict(iou=iou, precision=prec, recall=rec, f1=float(f1))

def to_wandb_image(logits, target, fire_last):
    # all [1,H,W]; create 3-panel for quick eyeballing
    import torchvision.utils as vutils
    pics = []
    for a in [torch.sigmoid(logits), target, fire_last]:
        pics.append(a.clamp(0,1))
    grid = torch.cat(pics, dim=-1)  # [1,H, 3W]
    return wandb.Image(grid.cpu().numpy())

def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    ap.add_argument("--model", default="conv2d", choices=["copy_last","conv2d","tiny3d"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_dir", default="checkpoints")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["global"]["seed"])
    data_cfg = cfg["data"]
    wandb_cfg = cfg["wandb"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / Loader
    ds = RawFireDataset(
        data_dir=data_cfg["data_dir"],
        sequence_length=data_cfg["sequence_length"],
        transform=None
    )
    dl = DataLoader(
        ds,
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        drop_last=data_cfg["drop_last"],
        collate_fn=collate_fn
    )

    # Model
    static_channels = ds.landscape_data.shape[0]
    model = make_model(args.model, static_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # W&B
    run = None
    if wandb_cfg.get("enabled", True) and wandb_cfg.get("mode","online") != "disabled":
        run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=wandb_cfg.get("run_name", f"phase1-{args.model}"),
            tags=wandb_cfg.get("tags", []) + ["phase1", args.model],
            notes=wandb_cfg.get("notes","")
        )
        wandb.config.update({
            "model": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
            "sequence_length": data_cfg["sequence_length"],
            "batch_size": data_cfg["batch_size"]
        }, allow_val_change=True)

    # Train (single split; just to sanity-check loss goes down)
    model.train()
    for epoch in range(1, args.epochs+1):
        total_loss = 0.0
        n = 0
        for fire, static, wind, target, valid in dl:
            fire   = fire.to(device)     # [B,1,H,W,T]
            static = static.to(device)   # [B,Cs,H,W]
            wind   = wind.to(device)     # [B,2,T]
            target = target.to(device)   # [B,1,H,W]

            logits = model(fire, static, wind)      # [B,1,H,W]
            loss = binary_cross_entropy_with_logits(logits, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * fire.size(0)
            n += fire.size(0)

        avg_loss = total_loss / max(1, n)

        # Quick eval on a small batch
        model.eval()
        with torch.no_grad():
            fire, static, wind, target, valid = next(iter(dl))
            fire, static, wind, target = fire.to(device), static.to(device), wind.to(device), target.to(device)
            logits = model(fire, static, wind)
            metrics = compute_metrics(logits, target, thr=args.threshold)
            preview = to_wandb_image(logits[0,0], target[0,0], fire[0,0,...,-1])
        model.train()

        log_dict = {"epoch": epoch, "loss/train": avg_loss, **{f"metrics/{k}": v for k,v in metrics.items()}}
        if run:
            wandb.log(log_dict)
            wandb.log({"preview": preview})

        print(f"[epoch {epoch}] loss={avg_loss:.4f} | iou={metrics['iou']:.4f} f1={metrics['f1']:.4f}")

    # Save
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.save_dir) / f"phase1_{args.model}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)
    if run:
        wandb.save(str(ckpt))
        run.finish()

if __name__ == "__main__":
    main()

