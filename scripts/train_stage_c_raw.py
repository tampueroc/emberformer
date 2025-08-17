import os, time, argparse, yaml, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm

from data import RawFireDataset, collate_fn
from models import UNetS
from utils import (
    init_wandb,
    define_common_metrics,
    save_artifact,
    log_preview_from_batch,
)

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
# Input shaping
# ----------------
def _wind_broadcast(last_wind: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # last_wind: [B,2] -> [B,2,H,W]
    return last_wind[:, :, None, None].expand(-1, -1, H, W)

def _stack_last_k_fire(fire_1hwt: torch.Tensor, k: int) -> torch.Tensor:
    """
    fire_1hwt: [B,1,H,W,T]
    return: [B,k,H,W] (last k real frames; left-padding stays zeros if k > valid length)
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    last_k = fire_1hwt[..., -k:]                  # [B,1,H,W,k]
    last_k = last_k.permute(0, 4, 1, 2, 3)        # [B,k,1,H,W]
    return last_k.squeeze(2)                      # [B,k,H,W]

def _make_x_in(fire, static, wind, valid, k_fire: int) -> torch.Tensor:
    """
    fire:   [B,1,H,W,T]
    static: [B,Cs,H,W]
    wind:   [B,2,T]
    valid:  [B,T]  (unused; we right-align so -1 is last real frame)
    returns x_in: [B, k_fire + Cs + 2, H, W]
    """
    B, _, H, W, _ = fire.shape
    fire_k = _stack_last_k_fire(fire, k_fire)           # [B,k,H,W]
    wind_last = wind[:, :, -1]                          # [B,2]
    wind_maps = _wind_broadcast(wind_last, H, W)        # [B,2,H,W]
    x_in = torch.cat([fire_k, static, wind_maps], dim=1)
    return x_in

# ----------------
# Loss (precision-focused option)
# ----------------
def bce_with_logits_weighted(logits, target, w_pos: float = 1.0, w_neg: float = 1.0, eps: float = 1e-8):
    """
    Per-pixel BCE with class-dependent weights. Setting w_neg > w_pos penalizes FP more,
    which generally increases precision (at the cost of recall).
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(
        target > 0.5,
        torch.as_tensor(w_pos, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(w_neg, device=logits.device, dtype=logits.dtype),
    )
    num = (loss * weights).sum()
    den = weights.sum().clamp_min(eps)
    return num / den

# ----------------
# Main
# ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage_c_raw.yaml")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--k_fire", type=int, default=None, help="how many past fire frames to stack as channels")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _pick_device(args.gpu)
    torch.manual_seed(cfg.get("global", {}).get("seed", 42))
    torch.backends.cudnn.benchmark = cfg.get("global", {}).get("cudnn_benchmark", True)

    # Data
    d = cfg["data"]
    ds = RawFireDataset(d["data_dir"], sequence_length=d.get("sequence_length", 3))
    dl = DataLoader(
        ds,
        batch_size=(args.batch_size or d.get("batch_size", 4)),
        num_workers=d.get("num_workers", 4),
        pin_memory=d.get("pin_memory", True),
        drop_last=d.get("drop_last", False),
        shuffle=d.get("shuffle", True),
        collate_fn=collate_fn,
    )

    # Raw Stage-C knobs
    rcfg = cfg.get("stage_c_raw", {})
    k_fire = int(args.k_fire if args.k_fire is not None else rcfg.get("k_fire", 1))
    mcfg = rcfg.get("metrics", {}) or cfg.get("stage_c", {}).get("metrics", {})
    pred_thresh = float(mcfg.get("pred_thresh", 0.5))

    lcfg = rcfg.get("loss", {})
    w_neg = float(lcfg.get("neg_weight", 2.0))  # ↑ penalize FP → better precision
    w_pos = float(lcfg.get("pos_weight", 1.0))

    # Model/optim
    Cs = ds.landscape_data.shape[0]
    in_ch = k_fire + Cs + 2
    base = int(rcfg.get("base_channels", 32))
    model = UNetS(in_ch=in_ch, base=base).to(device)
    tcfg = cfg.get("train", {})
    epochs = int(args.epochs if args.epochs is not None else tcfg.get("epochs", 5))
    lr = _coerce_float(args.lr, "cli.lr") or _coerce_float(tcfg.get("lr", 3e-4), "train.lr")
    wd = _coerce_float(tcfg.get("weight_decay", 0.0), "train.weight_decay")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # W&B
    run = init_wandb(cfg, context={"script": pathlib.Path(__file__).stem, "model": "unets_raw"})
    if run:
        define_common_metrics()
        run.summary.update({
            "dataset_size": len(ds),
            "model_name": "UNetS-raw",
            "model_params": sum(p.numel() for p in model.parameters()),
            "in_channels": in_ch,
            "k_fire": k_fire,
            "base_channels": base,
            "precision_focus": True,
            "loss_w_neg": w_neg,
            "loss_w_pos": w_pos,
        })

    # Metrics (PIXEL space)
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
    preview_logged_this_epoch = False
    log_every = int(cfg.get("train", {}).get("log_images_every", 200))

    for ep in range(epochs):
        # ----------------
        # Train epoch
        # ----------------
        model.train()
        train_bar = tqdm(dl, desc=f"train e{ep}", unit="batch", leave=False, dynamic_ncols=True)
        for batch in train_bar:
            fire, static, wind, target, valid = [x.to(device, non_blocking=True) for x in batch]
            x_in = _make_x_in(fire, static, wind, valid, k_fire)   # [B,in_ch,H,W]

            opt.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x_in)                               # [B,1,H,W]
                loss = bce_with_logits_weighted(logits, target, w_pos=w_pos, w_neg=w_neg)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                probs = torch.sigmoid(logits.float())
                preds = (probs > pred_thresh).int()
                p = preds.view(-1)
                y = target.int().view(-1)
                Mtr["acc"].update(p, y)
                Mtr["prec"].update(p, y)
                Mtr["rec"].update(p, y)
                Mtr["f1"].update(p, y)
                Mtr["iou"].update(p, y)

            # tqdm live metrics (running)
            try:
                train_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    prec=f"{Mtr['prec'].compute().item():.3f}",
                    iou=f"{Mtr['iou'].compute().item():.3f}",
                )
            except Exception:
                pass

            # WandB per-step logging
            if run:
                payload = {"epoch": ep, "step": step, "train/loss": loss.item()}
                if step % max(1, log_every) == 0:
                    # log running metrics occasionally
                    payload.update({
                        "train/precision_running": Mtr["prec"].compute().item(),
                        "train/iou_running": Mtr["iou"].compute().item(),
                    })
                run.log(payload, step=step)
            step += 1

        train_metrics = {f"train/{k}": v.compute().item() for k, v in Mtr.items()}
        for v in Mtr.values(): v.reset()

        # ----------------
        # "Validation" pass (quick sanity on same loader)
        # ----------------
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        val_bar = tqdm(dl, desc=f"val   e{ep}", unit="batch", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch in val_bar:
                fire, static, wind, target, valid = [x.to(device, non_blocking=True) for x in batch]
                x_in = _make_x_in(fire, static, wind, valid, k_fire)
                with amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(x_in)
                    vloss = bce_with_logits_weighted(logits, target, w_pos=w_pos, w_neg=w_neg)
                val_loss_sum += float(vloss); val_batches += 1

                probs = torch.sigmoid(logits.float())
                preds = (probs > pred_thresh).int()
                p = preds.view(-1); y = target.int().view(-1)
                Mva["acc"].update(p, y)
                Mva["prec"].update(p, y)
                Mva["rec"].update(p, y)
                Mva["f1"].update(p, y)
                Mva["iou"].update(p, y)

                # tqdm live metrics (running)
                try:
                    val_bar.set_postfix(
                        vloss=f"{float(vloss):.4f}",
                        prec=f"{Mva['prec'].compute().item():.3f}",
                        iou=f"{Mva['iou'].compute().item():.3f}",
                    )
                except Exception:
                    pass

                # Log one preview per epoch
                if run and not preview_logged_this_epoch:
                    log_preview_from_batch(
                        run,
                        batch=(fire.detach().cpu(), static.detach().cpu(), wind.detach().cpu(), target.detach().cpu(), valid.detach().cpu()),
                        logits=logits.detach().cpu(),
                        key="val/preview_raw",
                        max_images=cfg["wandb"].get("max_preview_images", 2),
                        size=int(cfg["wandb"].get("preview_size", 400)),
                    )
                    preview_logged_this_epoch = True

        val_metrics = {f"val/{k}": v.compute().item() for k, v in Mva.items()}
        for v in Mva.values(): v.reset()
        val_loss = val_loss_sum / max(1, val_batches)

        if run:
            run.log({"epoch": ep, **train_metrics, "val/loss": val_loss, **val_metrics}, step=step)

        print(
            f"[Stage C RAW][epoch {ep}] "
            f"train_loss={train_metrics.get('train/acc', 0.0):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['val/acc']:.4f} "
            f"val_prec={val_metrics['val/prec']:.4f} "
            f"val_rec={val_metrics['val/rec']:.4f} "
            f"val_f1={val_metrics['val/f1']:.4f} "
            f"val_iou={val_metrics['val/iou']:.4f}"
        )
        preview_logged_this_epoch = False  # reset for next epoch

    # optional artifact save
    if run and cfg["wandb"].get("log_artifacts", True):
        ckpt_dir = "checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(ckpt_dir, f"stageC-raw-unets-{stamp}.pt")
        torch.save(model.state_dict(), path)
        save_artifact(run, path, name="stageC-raw-unets-weights", type="model",
                      metadata={"stage":"C-raw","model":"UNetS","k_fire":k_fire,"in_ch":in_ch})
    if run: run.finish()

if __name__ == "__main__":
    main()

