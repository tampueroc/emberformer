import time, argparse, yaml, os
import torch
from torch.utils.data import DataLoader

from data import RawFireDataset, TokenFireDataset, collate_fn

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def time_loader(dl, iters=50):
    t0 = time.time()
    iters = min(iters, len(dl))
    it = iter(dl)
    for _ in range(iters):
        batch = next(it)
        _ = sum(x.numel() if torch.is_tensor(x) else 0 for x in (batch if isinstance(batch, tuple) else [batch]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - t0, iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emberformer.yaml")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    d = cfg["data"]
    pcfg = cfg.get("patchify_on_disk", {})
    cache_root = os.path.expanduser(pcfg["cache_dir"])

    raw_ds = RawFireDataset(d["data_dir"], sequence_length=d.get("sequence_length",3))
    tok_ds = TokenFireDataset(cache_root, raw_root=d["data_dir"], sequence_length=d.get("sequence_length",3))

    raw_dl = DataLoader(raw_ds, batch_size=d.get("batch_size",4), shuffle=True,
                        num_workers=d.get("num_workers",4), pin_memory=d.get("pin_memory",True),
                        drop_last=d.get("drop_last",False), collate_fn=collate_fn)

    # Token dataset yields (X,y) already — no custom collate
    tok_dl = DataLoader(tok_ds, batch_size=d.get("batch_size",4), shuffle=True,
                        num_workers=d.get("num_workers",4), pin_memory=d.get("pin_memory",True),
                        drop_last=d.get("drop_last",False))

    dt_raw, n = time_loader(raw_dl, iters=args.iters)
    dt_tok, _ = time_loader(tok_dl,  iters=args.iters)

    print(f"[bench] raw:    {dt_raw:.2f}s / {n} iters  -> {dt_raw/max(1,n):.3f}s/iter")
    print(f"[bench] tokens: {dt_tok:.2f}s / {n} iters  -> {dt_tok/max(1,n):.3f}s/iter")
    speedup = dt_raw / max(dt_tok, 1e-6)
    print(f"[bench] speedup x{speedup:.2f}")

if __name__ == "__main__":
    main()

