import os, time, argparse, yaml, pathlib
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import TokenFireDataset
from models import UNetS
from utils import init_wandb, define_common_metrics, save_artifact

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

def _prep_batch(batch):
    """
    batch: default-collated (X, y)
      X["static"] [B,N,Cs], X["fire_last"] [B,N], X["wind_last"] [B,2], X["valid"] [B,N]
      X["meta"] is a list of dicts (len=B) with Gy/Gx (or grid_h/grid_w)
      y [B,N]  (target patch mean)
    returns:
      x_in  [B, 1+Cs+2, Gy, Gx]
      y_map [B, 1, Gy, Gx]
      m_map [B, 1, Gy, Gx] (bool)
    """
    X, y = batch
    static = X["static"]      # [B,N,Cs]
    fire   = X["fire_last"]   # [B,N]
    wind   = X["wind_last"]   # [B,2]
    valid  = X["valid"]       # [B,N]
    metas  = X["meta"]        # list of dicts

    B, N, Cs = static.shape
    # Infer grid
    gH = metas[0].get("Gy") or metas[0].get("grid_h")
    gW = metas[0].get("Gx") or metas[0].get("grid_w")
    assert gH * gW == N, f"N={N} != Gy*Gx={gH*gW}"

    # reshape to [B,C, Gy,Gx]
    static_grid = static.permute(0, 2, 1).contiguous().view(B, Cs, gH, gW)
    fire_grid   = fire.view(B, 1, gH, gW)
    # broadcast wind to maps
    wind_maps = wind.view(B, 2, 1, 1).expand(-1, -1, gH, gW).contiguous()
    x_in = torch.cat([fire_grid, static_grid, wind_maps], dim=1)  # [B,1+Cs+2,Gy,Gx]

    y_map = y.view(B, 1, gH, gW)
    m_map = valid.view(B, 1, gH, gW)
    return x_in, y_map, m_map, (gH, gW)

def _masked_bce(logits, y, mask):
    loss_map = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (loss_map * mask).sum() / denom

def _log_preview(run, logits, y_map, step, key="preview", size=160):
    if run is None: return
    try:
        import wandb, numpy as np
    except Exception:
        return
    with torch.no_grad():
        probs = torch.sigmoid(logits).detach().cpu()
        yb    = y_map.detach().cpu()
        B = min(2, probs.size(0))
        images = []
        for i in range(B):
            # upscale for quick look
            up_p = F.interpolate(probs[i:i+1], size=(size, size), mode="nearest")[0,0]
            up_y = F.interpolate(yb[i:i+1],    size=(size, size), mode="nearest")[0,0]
            images.append(wandb.Image((up_p.numpy()*255).astype(np.uint8), caption=f"pred[{i}]"))
            images.append(wandb.Image((up_y.numpy()*255).astype(np.uint8), caption=f"target[{i}]"))
        run.log({key: images}, step=step)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage_c.yaml")
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
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

    ds = TokenFireDataset(cache_root, raw_root=raw_root, sequence_length=d.get("sequence_length", 3),
                          fire_value=cfg.get("encoding",{}).get("fire_value",231))
    bs = args.batch_size or d.get("batch_size", 8)
    dl = DataLoader(ds, batch_size=bs, shuffle=True,
                    num_workers=d.get("num_workers", 4),
                    pin_memory=d.get("pin_memory", True),
                    drop_last=d.get("drop_last", False))

    # Peek one sample for channel count
    X0, _ = ds[0]
    Cs = X0["static"].shape[1]
    in_ch = 1 + Cs + 2  # last_fire + static + wind

    # W&B
    run = init_wandb(cfg, context={"script": pathlib.Path(__file__).stem})
    if run:
        define_common_metrics()
        run.summary.update({
            "dataset_size": len(ds),
            "grid_h": X0["meta"].get("Gy") or X0["meta"].get("grid_h"),
            "grid_w": X0["meta"].get("Gx") or X0["meta"].get("grid_w"),
            "in_channels": in_ch,
            "stage": "C",
            "model_name": "UNetS",
        })

    # Model/optim
    base = args.base or int(cfg.get("stage_c", {}).get("base_channels", 32))
    model = UNetS(in_ch=in_ch, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=(args.lr or cfg["train"].get("lr", 1e-3)),
                           weight_decay=cfg["train"].get("weight_decay", 0.0))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    step = 0
    epochs = args.epochs or cfg["train"].get("epochs", 1)

    for ep in range(epochs):
        model.train()
        for batch in dl:
            x_in, y_map, mask, _ = _prep_batch(batch)
            x_in, y_map, mask = x_in.to(device, non_blocking=True), y_map.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(x_in)
                loss = _masked_bce(logits, y_map, mask)

            # optional: gradient clipping
            # scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                probs = torch.sigmoid(logits.float())
                preds = (probs > 0.5).float()
                m = mask.bool()
                p = preds[m].flatten()
                y = y_map[m].flatten()
                inter = ((p > 0.5) & (y > 0.5)).sum().float()
                union = ((p > 0.5) | (y > 0.5)).sum().float().clamp_min(1.0)
                iou = (inter / union).item()

            if run:
                run.log({"epoch": ep, "step": step, "train/loss": loss.item(), "train/iou": iou}, step=step)
                if step % max(1, int(cfg["train"].get("log_images_every", 200))) == 0:
                    _log_preview(run, logits, y_map, step, size=int(cfg["wandb"].get("preview_size",160)))

            step += 1

        print(f"[Stage C][epoch {ep}] loss={loss.item():.4f} iou={iou:.4f}")

    # optional artifact save
    if run and cfg["wandb"].get("log_artifacts", False):
        ckpt_dir = "checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(ckpt_dir, f"stageC-unets-{stamp}.pt")
        torch.save(model.state_dict(), path)
        save_artifact(run, path, name="stageC-unets-weights", type="model",
                      metadata={"stage":"C","model":"UNetS"})
    if run: run.finish()

if __name__ == "__main__":
    main()

