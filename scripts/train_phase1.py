import os, time, argparse, yaml, inspect, torch, pathlib
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

from data import RawFireDataset, collate_fn
from models import CopyLast, ConvHead2D, Tiny3D
from utils import (
    init_wandb,
    define_common_metrics,
    watch_model,
    log_batch_shapes,
    log_preview_from_batch,
    save_artifact,
)

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

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
    sig = inspect.signature(model.forward)
    if len(sig.parameters) >= 4:
        try:
            return model(fire, static, wind, valid_tokens)
        except TypeError:
            pass
    return model(fire, static, wind)

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

def _pick_device(gpu_arg: int | None):
    if torch.cuda.is_available():
        if gpu_arg is None:
            return torch.device("cuda")
        n = torch.cuda.device_count()
        if gpu_arg < 0 or gpu_arg >= n:
            print(f"[warn] --gpu {gpu_arg} out of range (available: 0..{n-1}); falling back to cuda:0")
            gpu_arg = 0
        torch.cuda.set_device(gpu_arg)
        return torch.device(f"cuda:{gpu_arg}")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    ap.add_argument("--model", default=None, help="copy_last | conv_head2d | tiny3d")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None, help="GPU index to use (e.g., 0 or 1)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.get("global", {}).get("seed", 42))
    torch.backends.cudnn.benchmark = cfg.get("global", {}).get("cudnn_benchmark", True)

    # Resolve effective training params BEFORE W&B init (so name/tags reflect CLI overrides)
    tcfg = cfg.get("train", {})
    if args.model is not None:
        tcfg["model"] = args.model
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    # Keep lr/wd in cfg as numbers
    if "lr" in tcfg:
        try:
            tcfg["lr"] = float(tcfg["lr"])
        except Exception:
            pass

    # Init W&B with short, informative name + auto-tags
    script_name = pathlib.Path(__file__).stem  # e.g. "train_phase1"
    device = _pick_device(args.gpu)
    run = init_wandb(cfg, context={"script": script_name})
    if run:
        define_common_metrics()

    ds, dl = make_loader(cfg)
    Cs = ds.landscape_data.shape[0]

    model_name = tcfg.get("model", "conv_head2d")
    epochs = tcfg.get("epochs", 1)
    lr = _coerce_float(tcfg.get("lr", 1e-3), "train.lr")
    wd = _coerce_float(tcfg.get("weight_decay", 0.0), "train.weight_decay")

    print(f"[phase1] using device: {device}")
    model = build_model(model_name, Cs).to(device)
    if run:
        watch_model(run, model, cfg)
        run.summary.update({
            "dataset_size": len(ds),
            "model_name": model_name,
            "model_params": sum(p.numel() for p in model.parameters()),
            "sequence_length": cfg["data"].get("sequence_length", 3),
            "batch_size": cfg["data"].get("batch_size", 4),
        })

    criterion = nn.BCEWithLogitsLoss()
    optimizer = None if isinstance(model, CopyLast) else torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd
    )

    # Metrics (deep-crowns set + IoU)
    acc  = torchmetrics.classification.BinaryAccuracy().to(device)
    prec = torchmetrics.classification.BinaryPrecision().to(device)
    rec  = torchmetrics.classification.BinaryRecall().to(device)
    f1   = torchmetrics.classification.BinaryF1Score().to(device)
    iou  = torchmetrics.classification.BinaryJaccardIndex().to(device)

    step = 0
    shapes_logged = False
    last_batch_cpu = None
    last_logits_cpu = None

    for epoch in range(epochs):
        model.train()
        for batch in dl:
            fire, static, wind, target, valid = [x.to(device) for x in batch]

            if not shapes_logged and run:
                log_batch_shapes(run, fire, epoch, step)
                shapes_logged = True

            logits = call_model(model, fire, static, wind, valid)
            loss = criterion(logits, target)

            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).int()
                p = preds.view(-1)
                y = target.int().view(-1)
                acc.update(p, y)
                prec.update(p, y)
                rec.update(p, y)
                f1.update(p, y)
                iou.update(p, y)

            if run:
                run.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"] if optimizer else 0.0,
                    },
                    step=step,
                )

            last_batch_cpu = (fire.detach().cpu(), static.detach().cpu(),
                              wind.detach().cpu(), target.detach().cpu(), valid.detach().cpu())
            last_logits_cpu = logits.detach().cpu()
            step += 1

        epoch_metrics = {
            "val/acc":       acc.compute().item(),
            "val/precision": prec.compute().item(),
            "val/recall":    rec.compute().item(),
            "val/f1":        f1.compute().item(),
            "val/iou":       iou.compute().item(),
        }
        for m in (acc, prec, rec, f1, iou):
            m.reset()

        if run:
            run.log({"epoch": epoch, **epoch_metrics}, step=step)
            if cfg["wandb"].get("log_batch_preview", True) and last_batch_cpu is not None:
                log_preview_from_batch(
                    run,
                    batch=last_batch_cpu,
                    logits=last_logits_cpu,
                    key="preview",
                    max_images=cfg["wandb"].get("max_preview_images", 8),
                    size=int(cfg["wandb"].get("preview_size", 160)),
                )

        print(
            f"[epoch {epoch}] "
            f"loss ~ {loss.item():.4f} | "
            f"acc {epoch_metrics['val/acc']:.4f} | "
            f"prec {epoch_metrics['val/precision']:.4f} | "
            f"rec {epoch_metrics['val/recall']:.4f} | "
            f"f1 {epoch_metrics['val/f1']:.4f} | "
            f"IoU {epoch_metrics['val/iou']:.4f}"
        )

    if run and cfg["wandb"].get("log_artifacts", True):
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ckpt_path = os.path.join(
            ckpt_dir, f"{(cfg['wandb'].get('run_name') or run.name)}-{model_name}-e{epochs}-{stamp}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)
        save_artifact(
            run,
            filepath=ckpt_path,
            name=f"{model_name}-weights",
            type="model",
            metadata={"epochs": epochs, "model": model_name},
        )
        run.summary["epochs"] = epochs

    if run:
        run.finish()

if __name__ == "__main__":
    main()

