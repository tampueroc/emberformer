import torch

def collate_tokens_temporal(batch):
    """
    Collate function for TokenFireDataset with use_history=True.
    Left-pads temporal dimension to T_max across batch.

    Args:
        batch: list of tuples (X, y) where X is a dict with:
            static:    [N, Cs]
            fire_hist: [T_hist_i, N]
            wind_hist: [T_hist_i, 2]
            valid:     [N] (spatial validity)
            meta:      [2] (Gy, Gx)

    Returns:
        X_batch: dict with
            static:    [B, N, Cs]
            fire_hist: [B, N, T_max]
            wind_hist: [B, T_max, 2]
            valid:     [B, N]
            valid_t:   [B, T_max] (temporal validity: True=real, False=pad)
            meta:      [B, 2] or list of [2] tensors
        y_batch: [B, N]
    """
    X_list, y_list = zip(*batch)
    B = len(X_list)
    
    # Determine T_max
    T_max = max(X["fire_hist"].shape[0] for X in X_list)
    
    # Get dimensions from first sample
    N, Cs = X_list[0]["static"].shape
    
    # Infer dtype/device from first sample
    static_dtype, static_device = X_list[0]["static"].dtype, X_list[0]["static"].device
    fire_dtype, fire_device = X_list[0]["fire_hist"].dtype, X_list[0]["fire_hist"].device
    wind_dtype, wind_device = X_list[0]["wind_hist"].dtype, X_list[0]["wind_hist"].device
    valid_dtype, valid_device = X_list[0]["valid"].dtype, X_list[0]["valid"].device
    
    # Allocate batch tensors
    static_batch = torch.empty((B, N, Cs), dtype=static_dtype, device=static_device)
    fire_hist_batch = torch.zeros((B, N, T_max), dtype=fire_dtype, device=fire_device)
    wind_hist_batch = torch.zeros((B, T_max, 2), dtype=wind_dtype, device=wind_device)
    valid_batch = torch.empty((B, N), dtype=valid_dtype, device=valid_device)
    valid_t_batch = torch.zeros((B, T_max), dtype=torch.bool, device=fire_device)
    y_batch = torch.empty((B, N), dtype=y_list[0].dtype, device=y_list[0].device)
    
    meta_list = []
    
    for i, (X, y) in enumerate(batch):
        T_hist_i = X["fire_hist"].shape[0]
        
        # Left-pad: place real data at the end
        fire_hist_batch[i, :, -T_hist_i:] = X["fire_hist"].T  # [T_hist,N] -> [N,T_hist]
        wind_hist_batch[i, -T_hist_i:, :] = X["wind_hist"]    # [T_hist,2]
        valid_t_batch[i, -T_hist_i:] = True
        
        static_batch[i] = X["static"]
        valid_batch[i] = X["valid"]
        y_batch[i] = y
        meta_list.append(X["meta"])
    
    X_batch = {
        "static": static_batch,
        "fire_hist": fire_hist_batch,
        "wind_hist": wind_hist_batch,
        "valid": valid_batch,
        "valid_t": valid_t_batch,
        "meta": meta_list,  # Keep as list, or stack if needed
    }
    
    return X_batch, y_batch


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

