import os
import datetime as _dt
from typing import Optional, Dict, Any, List
import torch
import torch.nn.functional as F

def _maybe_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None

# -----------------------------
# Helpers for naming & tags
# -----------------------------
def _fmt_lr(val) -> str:
    try:
        v = float(val)
    except Exception:
        return str(val)
    return f"{v:.0e}" if v < 1e-2 else f"{v:g}"

def _default_run_name(cfg: Dict[str, Any], context: Dict[str, Any]) -> str:
    tcfg = cfg.get("train", {})
    dcfg = cfg.get("data", {})
    script = context.get("script", "train")
    model  = tcfg.get("model", "unknown")
    seq    = dcfg.get("sequence_length", "?")
    lr     = _fmt_lr(tcfg.get("lr", "1e-3"))
    now    = _dt.datetime.now().strftime("%m%d-%H%M")
    return f"{script}-{model}-T{seq}-lr{lr}-{now}"

def _build_run_name(cfg: Dict[str, Any], context: Dict[str, Any]) -> str:
    wb   = cfg.get("wandb", {})
    tmpl = wb.get("run_name_template")
    if not tmpl:
        return _default_run_name(cfg, context)
    tcfg = cfg.get("train", {})
    dcfg = cfg.get("data", {})
    fmtdict = {
        "script": context.get("script", "train"),
        "model":  tcfg.get("model", "unknown"),
        "seq":    dcfg.get("sequence_length", "?"),
        "lr":     _fmt_lr(tcfg.get("lr", "1e-3")),
        "time":   _dt.datetime.now().strftime("%m%d-%H%M"),
    }
    try:
        return tmpl.format(**fmtdict)
    except Exception:
        return _default_run_name(cfg, context)

def _auto_tags(cfg: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    # Minimal, high-signal tags for filtering
    tcfg = cfg.get("train", {})
    dcfg = cfg.get("data", {})
    return [
        f"script:{context.get('script','train')}",
        f"model:{tcfg.get('model','unknown')}",
        f"T:{dcfg.get('sequence_length','?')}",
    ]

# -----------------------------
# Core init + common definitions
# -----------------------------
def init_wandb(cfg: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", True) or wb.get("mode", "online") == "disabled":
        return None
    wandb = _maybe_import_wandb()
    if wandb is None:
        print("[wandb] not installed; continuing without logging.")
        return None

    context = dict(context or {})
    mode = wb.get("mode", "online")
    os.environ["WANDB_MODE"] = mode

    name = wb.get("run_name") or _build_run_name(cfg, context)
    tags = list(wb.get("tags", []))
    if wb.get("auto_tags", False):
        tags.extend(_auto_tags(cfg, context))

    run = wandb.init(
        project=wb.get("project", "emberformer"),
        entity=wb.get("entity") or None,
        name=name,
        tags=tags,
        notes=wb.get("notes", ""),
        config=cfg,
        mode=mode if mode in ("online", "offline", "disabled") else "online",
        group=wb.get("group") or None,
        job_type=wb.get("job_type") or None,
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
    _, _, _, _, T_max = fire_batch.shape
    wandb.log({"batch/T_max": T_max, "epoch": epoch}, step=step)
    run.summary["T_max"] = T_max

def _shrink_1hw(img_1hw: torch.Tensor, size: int = 160) -> torch.Tensor:
    if img_1hw.ndim != 3 or img_1hw.shape[0] != 1:
        raise ValueError("Expected [1,H,W] tensor.")
    x = img_1hw.unsqueeze(0).float()                 # [1,1,H,W]
    x = F.interpolate(x, size=(size, size), mode="nearest")
    return x.squeeze(0)                              # [1,s,s]

def log_preview_from_batch(
    run,
    batch: tuple,
    logits: Optional[torch.Tensor] = None,
    *,
    key: str = "preview",
    max_images: int = 8,
    size: int = 160,
):
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
        preds = (probs > 0.5).float()
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

