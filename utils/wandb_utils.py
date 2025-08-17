import os
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F

def _maybe_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None

# -----------------------------
# Core init + common definitions
# -----------------------------
def init_wandb(cfg: Dict[str, Any]):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", True) or wb.get("mode", "online") == "disabled":
        return None
    wandb = _maybe_import_wandb()
    if wandb is None:
        print("[wandb] not installed; continuing without logging.")
        return None

    mode = wb.get("mode", "online")
    os.environ["WANDB_MODE"] = mode  # respect offline/online

    run = wandb.init(
        project=wb.get("project", "emberformer"),
        entity=wb.get("entity") or None,
        name=wb.get("run_name", "phase"),
        tags=wb.get("tags", []),
        notes=wb.get("notes", ""),
        config=cfg,
        mode=mode if mode in ("online", "offline", "disabled") else "online",
    )
    return run

def define_common_metrics():
    wandb = _maybe_import_wandb()
    if wandb is None:
        return
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

def watch_model(run, model, cfg):
    if run is None:
        return
    wandb = _maybe_import_wandb()
    if wandb is None:
        return
    if cfg.get("wandb", {}).get("watch", True):
        wandb.watch(model, log="gradients", log_freq=100)

# -------------
# Handy logging
# -------------
def log_batch_shapes(run, fire_batch: torch.Tensor, epoch: int, step: int):
    if run is None:
        return
    wandb = _maybe_import_wandb()
    if wandb is None:
        return
    _, _, H, W, T_max = fire_batch.shape
    wandb.log({"batch/H": H, "batch/W": W, "batch/T_max": T_max, "epoch": epoch}, step=step)
    run.summary["H"] = H
    run.summary["W"] = W
    run.summary["T_max"] = T_max

def _shrink_1hw(img_1hw: torch.Tensor, size: int = 160) -> torch.Tensor:
    """
    Downsample a single [1,H,W] tensor to [1,size,size] using nearest (keeps binary masks crisp).
    """
    x = img_1hw
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("Expected [1,H,W] tensor.")
    x = x.unsqueeze(0).float()                                # [1,1,H,W]
    x = F.interpolate(x, size=(size, size), mode="nearest")   # [1,1,s,s]
    return x.squeeze(0)                                       # [1,s,s]

def log_preview_from_batch(
    run,
    batch: tuple,
    logits: Optional[torch.Tensor] = None,
    *,
    key: str = "preview",
    max_images: int = 8,
    size: int = 160,
):
    """
    Logs a preview panel. If `logits` is provided, logs pred/target/last_fire.
    Otherwise logs target/last_fire (useful for phase0).
    batch: (fire, static, wind, target, valid)
    """
    if run is None:
        return
    wandb = _maybe_import_wandb()
    if wandb is None:
        return

    fire, static, wind, target, valid = batch
    fire = fire.detach().cpu()
    target = target.detach().cpu()

    images = []
    B = fire.shape[0]
    n = min(max_images, B)

    if logits is not None:
        probs = torch.sigmoid(logits).detach().cpu()
        preds = (probs > 0.5).float()  # [B,1,H,W]
        for i in range(n):
            images.append(wandb.Image(_shrink_1hw(preds[i], size),  caption=f"pred[{i}]"))
            images.append(wandb.Image(_shrink_1hw(target[i], size), caption=f"target[{i}]"))
            images.append(wandb.Image(_shrink_1hw(fire[i, :, :, :, -1], size), caption=f"last_fire[{i}]"))
    else:
        for i in range(n):
            images.append(wandb.Image(_shrink_1hw(target[i], size), caption=f"target[{i}]"))
            images.append(wandb.Image(_shrink_1hw(fire[i, :, :, :, -1], size), caption=f"last_fire[{i}]"))

    run.log({key: images})

# ---------------
# Artifacts helper
# ---------------
def save_artifact(run, filepath: str, name: str, *, type: str = "model", metadata: Optional[Dict[str, Any]] = None):
    if run is None:
        return
    wandb = _maybe_import_wandb()
    if wandb is None:
        return
    art = wandb.Artifact(name=name, type=type, metadata=metadata or {})
    art.add_file(filepath)
    run.log_artifact(art)

