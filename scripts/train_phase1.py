import os, time, argparse, yaml, inspect, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import wandb

from data import RawFireDataset, collate_fn
from models import CopyLast, ConvHead2D, Tiny3D


def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def init_wandb(cfg):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", True) or wb.get("mode", "online") == "disabled":
        return None
    mode = wb.get("mode", "online")
    os.environ["WANDB_MODE"] = mode  # respect offline/online
    run = wandb.init(
        project=wb.get("project", "emberformer"),
        entity=wb.get("entity") or None,
        name=wb.get("run_name", "phase1"),
        tags=wb.get("tags", []),
        notes=wb.get("notes", ""),
        config=cfg,  # capture full YAML
        mode=mode if mode in ("online", "offline", "disabled") else "online",
    )

    # nice axes: epoch on x for train/* and val/* metrics
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    return run


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
    n = (name or "").lower()
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
        try:
            return model(fire, static, wind, valid_tokens)
        except TypeError:
            pass
    return model(fire, static, wind)


def preview(run, batch, logits, max_images=4):
    """Log a small triplet preview: pred / target / last_fire."""
    if run is None:
        return
    fire, static, wind, target, valid = batch
    probs = torch.sigmoid(logits).detach().cpu()
    preds = (probs > 0.5).float()  # [B,1,H,W]
    last = fire[..., -1].detach().cpu()  # [B,1,H,W]
    tgt = target.detach().cpu()  # [B,1,H,W]

    imgs = []
    n = min(max_images, preds.shape[0])
    for i in range(n):
        # Keep channel dim for wandb.Image: [1,H,W] is OK
        imgs.append(wandb.Image(preds[i], caption=f"pred[{i}]"))
        imgs.append(wandb.Image(tgt[i], caption=f"target[{i}]"))
        imgs.append(wandb.Image(last[i], caption=f"last_fire[{i}]"))
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
    lr = tcfg.get("lr", 1e-3)
    wd = tcfg.get("weight_decay", 0.0)
    log_every = tcfg.get("log_images_every", 200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, Cs).to(device)

    # Optional W&B watch
    if run and cfg.get("wandb", {}).get("watch", True):
        wandb.watch(model, log="gradients", log_freq=100)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = None if isinstance(model, CopyLast) else torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd
    )

    acc = torchmetrics.classification.BinaryAccuracy().to(device)
    iou = torchmetrics.classification.BinaryJaccardIndex().to(device)

    # Summaries
    if run:
        run.summary["dataset_size"] = len(ds)

    step = 0
    H = W = T_max = None

    for epoch in range(epochs):
        model.train()
        for batch in dl:
            fire, static, wind, target, valid = [x.to(device) for x in batch]

            if H is None:
                # Record shapes once
                _, _, H, W, T_max = fire.shape
                if run:
                    wandb.log(
                        {"batch/H": H, "batch/W": W, "batch/T_max": T_max, "epoch": epoch},
                        step=step,
                    )
                    run.summary["H"] = H
                    run.summary["W"] = W
                    run.summary["T_max"] = T_max

            logits = call_model(model, fire, static, wind, valid)  # tolerant call
            loss = criterion(logits, target)

            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).int()
                acc.update(preds, target.int())
                iou.update(preds, target.int())

            if run:
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"] if optimizer else 0.0,
                    },
                    step=step,
                )

            if run and cfg["wandb"].get("log_batch_preview", True) and (
                step == 0 or (log_every and step % log_every == 0)
            ):
                preview(run, batch, logits, max_images=cfg["wandb"].get("max_preview_images", 8))

            step += 1

        # end epoch
        epoch_acc, epoch_iou = acc.compute().item(), iou.compute().item()
        acc.reset()
        iou.reset()

        if run:
            wandb.log(
                {"epoch": epoch, "val/acc": epoch_acc, "val/iou": epoch_iou},
                step=step,
            )

        print(f"[epoch {epoch}] loss ~ {loss.item():.4f} | acc {epoch_acc:.4f} | IoU {epoch_iou:.4f}")

    # Save & log model artifact
    if run and cfg["wandb"].get("log_artifacts", True):
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ckpt_path = os.path.join(
            ckpt_dir, f"{cfg['wandb'].get('run_name','run')}-{model_name}-e{epochs}-{stamp}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)

        art = wandb.Artifact(
            name=f"{model_name}-weights",
            type="model",
            metadata={"epochs": epochs, "model": model_name},
        )
        art.add_file(ckpt_path)
        run.log_artifact(art)

        run.summary["epochs"] = epochs

    if run:
        run.finish()


if __name__ == "__main__":
    main()

