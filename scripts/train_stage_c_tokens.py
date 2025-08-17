import os, time, argparse, yaml, pathlib, json, random
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from torch import amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from data import TokenFireDataset
from models import UNetS
from utils import init_wandb, define_common_metrics, save_artifact, log_grid_preview

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

def _prep_batch(batch):
    X, y = batch
    static = X["static"]      # [B,N,Cs]
    fire   = X["fire_last"]   # [B,N]
    wind   = X["wind_last"]   # [B,2]
    valid  = X["valid"]       # [B,N]
    metas  = X["meta"]

    B, N, Cs = static.shape
    if torch.is_tensor(metas):
        if metas.ndim == 2:               # [B,2]
            gH, gW = map(int, metas[0].tolist())
            # (optional) sanity: assert all equal
            # assert torch.all(metas == metas[0]), "Mixed grid sizes in batch"
        else:                              # [2] (e.g., batch size 1 without collation)
            gH, gW = map(int, metas.tolist())
    else:
        raise TypeError(f"Unexpected meta type: {type(metas)}")
    gH, gW = metas[0]
    assert gH * gW == N, f"N={N} != Gy*Gx={gH*gW}"

    static_grid = static.permute(0, 2, 1).contiguous().view(B, Cs, gH, gW)
    fire_grid   = fire.view(B, 1, gH, gW)
    wind_maps = wind.view(B, 2, 1, 1).expand(-1, -1, gH, gW).contiguous()
    x_in = torch.cat([fire_grid, static_grid, wind_maps], dim=1)  # [B,1+Cs+2,Gy,Gx]

    y_map = y.view(B, 1, gH, gW)
    m_map = valid.view(B, 1, gH, gW)
    return x_in, y_map, m_map, (gH, gW)

def _masked_bce(logits, y, mask):
    loss_map = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    denom = mask.float().sum().clamp_min(1.0)
    return (loss_map * mask.float()).sum() / denom

# ---------- split helpers (persisted) ----------
def _split_file_path(cache_root: str, seed: int) -> str:
    split_dir = os.path.join(cache_root, "_splits")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"stageC-seq-split-seed{seed}.json")

def _make_or_load_sequence_split(ds, cache_root: str, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    path = _split_file_path(cache_root, seed)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f), path

    seqs = sorted(ds.seq_dirs)  # ["sequence_001", ...]
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
# -----------------------------------------------

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

    # Peek one sample for channel count
    X0, _ = ds[0]
    Cs = X0["static"].shape[1]
    in_ch = 1 + Cs + 2

    # W&B
    run = init_wandb(cfg, context={"script": pathlib.Path(__file__).stem})
    if run:
        define_common_metrics()

    # deterministic split
    split_cfg = cfg.get("split", {})
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac   = float(split_cfg.get("val",   0.1))
    test_frac  = float(split_cfg.get("test",  0.1))
    seed_split = int(split_cfg.get("seed", cfg.get("global", {}).get("seed", 42)))

    split, split_path = _make_or_load_sequence_split(ds, cache_root, train_frac, val_frac, test_frac, seed_split)
    train_idxs = _subset_indices_by_sequence(ds, split["train"])
    val_idxs   = _subset_indices_by_sequence(ds, split["val"])
    test_idxs  = _subset_indices_by_sequence(ds, split["test"])  # saved for later

    train_set = torch.utils.data.Subset(ds, train_idxs)
    val_set   = torch.utils.data.Subset(ds, val_idxs)

    bs = args.batch_size or d.get("batch_size", 32)
    dl_train = DataLoader(train_set, batch_size=bs, shuffle=True,
                          num_workers=d.get("num_workers", 8),
                          pin_memory=d.get("pin_memory", True),
                          drop_last=d.get("drop_last", False))
    dl_val = DataLoader(val_set, batch_size=bs, shuffle=False,
                        num_workers=d.get("num_workers", 8),
                        pin_memory=d.get("pin_memory", True),
                        drop_last=False)

    if run:
        gh0, gw0 = X0["meta"]
        run.summary.update({
            "dataset_size": len(ds),
            "grid_h": gh0,
            "grid_w": gw0,
            "in_channels": in_ch,
            "stage": "C",
            "model_name": "UNetS",
            "split_file": split_path,
            "split/train_seqs": len(split["train"]),
            "split/val_seqs":   len(split["val"]),
            "split/test_seqs":  len(split["test"]),
            "train_samples":    len(train_set),
            "val_samples":      len(val_set),
            "test_samples":     len(test_idxs),
        })

    # Model/optim
    base = args.base or int(cfg.get("stage_c", {}).get("base_channels", 32))
    model = UNetS(in_ch=in_ch, base=base).to(device)
    lr = _coerce_float(args.lr, "cli.lr") or _coerce_float(cfg["train"].get("lr", 3e-4), "train.lr")
    wd = _coerce_float(cfg["train"].get("weight_decay", 0.0), "train.weight_decay")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # torchmetrics (train & val)
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
    epochs = args.epochs or cfg["train"].get("epochs", 5)
    log_imgs_every = int(cfg["train"].get("log_images_every", 200))
    preview_size = int(cfg["wandb"].get("preview_size", 160))

    for ep in range(epochs):
        # ----- train -----
        model.train()
        for batch in dl_train:
            x_in, y_map, mask, _ = _prep_batch(batch)
            x_in = x_in.to(device, non_blocking=True)
            y_map = y_map.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x_in)
                loss = _masked_bce(logits, y_map, mask)

            scaler.scale(loss).backward()
            # scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                probs = torch.sigmoid(logits.float())
                preds_bin = (probs > 0.5).int()
                target_bin = (y_map > 0.5).int()
                m = mask.bool()
                p = preds_bin[m].flatten()
                yb = target_bin[m].flatten()
                Mtr["acc"].update(p, yb)
                Mtr["prec"].update(p, yb)
                Mtr["rec"].update(p, yb)
                Mtr["f1"].update(p, yb)
                Mtr["iou"].update(p, yb)

            if run:
                run.log({"epoch": ep, "step": step, "train/loss": loss.item()}, step=step)
                if step % max(1, log_imgs_every) == 0:
                    log_grid_preview(run, probs, y_map, key="train/preview", size=preview_size)

            step += 1

        # epoch-end train metrics
        train_metrics = {f"train/{k}": v.compute().item() for k, v in Mtr.items()}
        for v in Mtr.values(): v.reset()

        # ----- validation -----
        model.eval()
        val_loss_total = 0.0
        nb = 0
        with torch.no_grad():
            for batch in dl_val:
                x_in, y_map, mask, _ = _prep_batch(batch)
                x_in = x_in.to(device, non_blocking=True)
                y_map = y_map.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(x_in)
                    vloss = _masked_bce(logits, y_map, mask)
                val_loss_total += float(vloss)
                nb += 1

                probs = torch.sigmoid(logits.float())
                preds_bin = (probs > 0.5).int()
                target_bin = (y_map > 0.5).int()
                m = mask.bool()
                p = preds_bin[m].flatten()
                yb = target_bin[m].flatten()
                Mva["acc"].update(p, yb)
                Mva["prec"].update(p, yb)
                Mva["rec"].update(p, yb)
                Mva["f1"].update(p, yb)
                Mva["iou"].update(p, yb)

        val_metrics = {f"val/{k}": v.compute().item() for k, v in Mva.items()}
        for v in Mva.values(): v.reset()
        val_loss = val_loss_total / max(1, nb)

        if run:
            run.log({"epoch": ep, **train_metrics, "val/loss": val_loss, **val_metrics}, step=step)

        print(
            f"[Stage C][epoch {ep}] "
            f"train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['val/acc']:.4f} "
            f"val_prec={val_metrics['val/prec']:.4f} "
            f"val_rec={val_metrics['val/rec']:.4f} "
            f"val_f1={val_metrics['val/f1']:.4f} "
            f"val_iou={val_metrics['val/iou']:.4f}"
        )

    # optional artifact save
    if run and cfg["wandb"].get("log_artifacts", True):
        ckpt_dir = "checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(ckpt_dir, f"stageC-unets-{stamp}.pt")
        torch.save(model.state_dict(), path)
        save_artifact(run, path, name="stageC-unets-weights", type="model",
                      metadata={"stage":"C","model":"UNetS"})
    if run: run.finish()

if __name__ == "__main__":
    main()

