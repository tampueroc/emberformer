import os, argparse, yaml, inspect, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import wandb

from data import RawFireDataset, collate_fn
from models import CopyLast, ConvHead2D, Tiny3D

def load_cfg(p):
    with open(p, "r") as f: return yaml.safe_load(f)

def init_wandb(cfg):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", True) or wb.get("mode", "online") == "disabled":
        return None
    os.environ["WANDB_MODE"] = wb.get("mode", "online")
    return wandb.init(
        project=wb.get("project", "emberformer"),
        entity=wb.get("entity") or None,
        name=wb.get("run_name", "phase1"),
        tags=wb.get("tags", []),
        notes=wb.get("notes", ""),
        config=cfg,
    )

def make_loader(cfg):
    d = cfg["data"]
    ds = RawFireDataset(d["data_dir"], sequence_length=d.get("sequence_length", 3))
    dl = DataLoader(
        ds,
        batch_size=d.get("batch_size", 4),
        num_workers=d.get("num_workers", 4),
        pin_memory=d.get("pin_memory", True),
        drop_last=d.get("drop_last", False),
        shuffle=d.get("shuffle", True),
        collate_fn=collate_fn,
    )
    return ds, dl

def build_model(name, static_channels):
    n = name.lower()
    if n in ["copy_last", "copy", "identity"]:
        return CopyLast()
    if n in ["conv_head2d", "conv2d", "head2d"]:
        return ConvHead2D(static_channels=static_channels, hidden=32)
    if n in ["tiny3d", "3d"]:
        return Tiny3D(in_ch=1, hidden=16)
    raise ValueError(f"unknown model {name}")

def call_model(model, fire, static, wind, valid_tokens):
    # Try (fire, static, wind, valid_tokens) then (fire, static, wind)
    sig = inspect.signature(model.forward)
    if len(sig.parameters) >= 4:
        try: return model(fire, static, wind, valid_tokens)
        except TypeError: pass
    return model(fire, static, wind)

def preview(run, batch, logits, max_images=4):
    if run is None: return
    fire, static, wind, target, valid = batch
    probs = torch.sigmoid(logits).detach().cpu()
    preds = (probs > 0.5).float()          # [B,1,H,W]
    last  = fire[..., -1].detach().cpu()   # [B,1,H,W]
    tgt   = target.detach().cpu()          # [B,1,H,W]

    imgs = []
    n = min(max_images, preds.shape[0])
    for i in range(n):
        # NOTE: keep the channel dim -> pass [1,H,W], not [H,W]
        imgs.append(wandb.Image(preds[i], caption=f"pred[{i}]"))
        imgs.append(wandb.Image(tgt[i],   caption=f"target[{i}]"))
        imgs.append(wandb.Image(last[i],  caption=f"last_fire[{i}]"))
    run.log({"preview": imgs})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    ap.add_argument("--model", default=None, help="copy_last | conv_head2d | tiny3d")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.get("global", {}).get("seed", 42))
    torch.backends.cudnn.benchmark = cfg.get("global", {}).get("cudnn_benchmark", True)

    run = init_wandb(cfg)
    ds, dl = make_loader(cfg)
    Cs = ds.landscape_data.shape[0]

    tcfg = cfg.get("train", {})
    model_name = args.model or tcfg.get("model", "conv_head2d")
    epochs = args.epochs or tcfg.get("epochs", 1)
    lr = tcfg.get("lr", 1e-3); wd = tcfg.get("weight_decay", 0.0)
    log_every = tcfg.get("log_images_every", 200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, Cs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = None if isinstance(model, CopyLast) else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    acc = torchmetrics.classification.BinaryAccuracy().to(device)
    iou = torchmetrics.classification.BinaryJaccardIndex().to(device)

    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in dl:
            fire, static, wind, target, valid = [x.to(device) for x in batch]
            logits = call_model(model, fire, static, wind, valid)     # tolerant call
            loss = criterion(logits, target)

            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).int()
                acc.update(preds, target.int()); iou.update(preds, target.int())

            if run:
                wandb.log({
                    "loss/train": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"] if optimizer else 0.0,
                    "epoch": epoch,
                    "step": step,
                }, step=step)

            if run and (step == 0 or (log_every and step % log_every == 0)):
                preview(run, batch, logits, max_images=cfg["wandb"].get("max_preview_images", 8))

            step += 1

        epoch_acc, epoch_iou = acc.compute().item(), iou.compute().item()
        acc.reset(); iou.reset()
        if run: wandb.log({"metrics/acc": epoch_acc, "metrics/iou": epoch_iou, "epoch": epoch}, step=step)
        print(f"[epoch {epoch}] loss ~ {loss.item():.4f} | acc {epoch_acc:.4f} | IoU {epoch_iou:.4f}")

    if run: run.finish()

if __name__ == "__main__":
    main()

