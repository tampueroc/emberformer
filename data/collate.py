import torch

def collate_fn(batch):
    """
    Left-pad variable-length sequences and emit a validity mask.

    Args:
        batch: list of tuples (fire, static, wind, target) with shapes:
            fire   : [1, H, W, T_i]
            static : [Cs, H, W]
            wind   : [T_i, 2]
            target : [1, H, W]

    Returns:
        fire_padded   : [B, 1, H, W, T_max]
        static_batch  : [B, Cs, H, W]
        wind_padded   : [B, 2, T_max]
        target_batch  : [B, 1, H, W]
        valid_tokens  : [B, T_max] (bool; True=valid, False=pad)
    """
    fires, statics, winds, targets = zip(*batch)

    B = len(fires)
    # Max time across the batch
    T_max = max(f.shape[-1] for f in fires)

    # Common H,W,Cs from first sample
    _, H, W, _ = fires[0].shape
    Cs = statics[0].shape[0]

    # Respect dtype/device of inputs
    fire_dtype, fire_device = fires[0].dtype, fires[0].device
    static_dtype, static_device = statics[0].dtype, statics[0].device
    wind_dtype, wind_device = winds[0].dtype, winds[0].device
    target_dtype, target_device = targets[0].dtype, targets[0].device

    # Allocate padded tensors
    fire_padded  = torch.zeros((B, 1, H, W, T_max), dtype=fire_dtype, device=fire_device)
    wind_padded  = torch.zeros((B, 2, T_max),       dtype=wind_dtype, device=wind_device)
    static_batch = torch.empty((B, Cs, H, W),       dtype=static_dtype, device=static_device)
    target_batch = torch.empty((B, 1, H, W),        dtype=target_dtype, device=target_device)
    valid_tokens = torch.zeros((B, T_max),          dtype=torch.bool,   device=fire_device)

    for i, (f, s, w, t) in enumerate(zip(fires, statics, winds, targets)):
        Ti = f.shape[-1]
        # Right-align (left-pad) along time
        fire_padded[i, ..., -Ti:] = f
        wind_padded[i, :, -Ti:] = w.T  # w: [T_i,2] -> [2,T_i]
        valid_tokens[i, -Ti:] = True

        static_batch[i] = s
        target_batch[i] = t

    return (
        fire_padded.contiguous(),
        static_batch.contiguous(),
        wind_padded.contiguous(),
        target_batch.contiguous(),
        valid_tokens,
    )

