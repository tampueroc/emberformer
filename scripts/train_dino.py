"""
Training script for EmberFormer-DINO

Mirrors train_emberformer.py but uses:
- RawFireDataset with pixel-level fire frames
- EmberFormerDINO model (DINO encoder + temporal transformer + simple decoder)
- Frozen DINO in Phase 1, optional fine-tuning in Phase 2
- Differential learning rates (DINO vs trainable components)
"""
import os, time, argparse, yaml, pathlib, json, random
from collections import defaultdict
import torch
import torch.nn as nn
from torch import amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm

from data import RawFireDataset
from models.emberformer import EmberFormerDINO
from utils import init_wandb, save_artifact

# ----------------
# Early Stopping
# ----------------
class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (F1, IoU), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch, value):
        """Returns True if should stop training"""
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def state_dict(self):
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
        }

# ----------------
# Config helpers
# ----------------
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
    """Safely convert config value to float"""
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

# ----------------
# Loss Functions (matching train_emberformer.py)
# ----------------
def _masked_bce(logits, y, mask, pos_weight=None):
    """BCE loss over masked pixels"""
    loss_map = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    loss_map = loss_map * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss_map.sum() / denom

def _auto_pos_weight(y_map, mask):
    """Compute pos_weight as (neg_mass / pos_mass)"""
    y = (y_map * mask).float()
    pos_mass = y.sum().clamp_min(1.0)
    neg_mass = (mask.float().sum() - pos_mass).clamp_min(1.0)
    return (neg_mass / pos_mass)

def _masked_soft_dice_loss(logits, y, mask, smooth=1.0, ignore_empty=True):
    """Soft Dice loss over masked pixels"""
    p = torch.sigmoid(logits).float()
    y = y.float()
    m = mask.float()
    
    p = p * m
    y = y * m
    
    dims = (1, 2, 3)
    intersection = (p * y).sum(dims)
    p_sum = p.sum(dims)
    y_sum = y.sum(dims)
    
    dice = (2.0 * intersection + smooth) / (p_sum + y_sum + smooth)
    
    if ignore_empty:
        valid = (y_sum > 0)
        if valid.any():
            dice_loss = (1.0 - dice[valid]).mean()
        else:
            return torch.zeros([], device=logits.device, dtype=logits.dtype)
    else:
        dice_loss = (1.0 - dice).mean()
    
    return torch.sqrt(dice_loss + 1e-7)

def _masked_focal_loss(logits, y, mask, alpha=0.25, gamma=2.0, pos_weight=None):
    """Focal Loss for handling class imbalance"""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none', pos_weight=pos_weight)
    
    probs = torch.sigmoid(logits)
    p_t = probs * y + (1 - probs) * (1 - y)
    focal_weight = (1 - p_t) ** gamma
    
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    loss = alpha_t * focal_weight * bce
    loss = loss * mask.float()
    
    return loss.sum() / mask.float().sum().clamp_min(1.0)

def _masked_tversky_loss(logits, y, mask, alpha=0.7, beta=0.3, smooth=1.0):
    """Tversky Loss for precision/recall control"""
    p = torch.sigmoid(logits).float()
    y = y.float()
    m = mask.float()
    
    p = p * m
    y = y * m
    
    dims = (1, 2, 3)
    tp = (p * y).sum(dims)
    fp = (p * (1 - y)).sum(dims)
    fn = ((1 - p) * y).sum(dims)
    
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - tversky).mean()

# ----------------
# Custom collate for RawFireDataset
# ----------------
def collate_raw_dino(batch):
    """
    Collate function for DINO training with variable-length sequences
    
    Input: List of (fire_seq, static, wind, target) tuples
    Output: Batched tensors with left-padding
    """
    T_max = max(item[0].shape[-1] for item in batch)
    B = len(batch)
    
    _, H, W, _ = batch[0][0].shape
    Cs = batch[0][1].shape[0]
    
    fire_hist = torch.zeros((B, T_max, 1, H, W), dtype=batch[0][0].dtype)
    static_batch = torch.zeros((B, Cs, H, W), dtype=batch[0][1].dtype)
    wind_batch = torch.zeros((B, T_max, 2), dtype=batch[0][2].dtype)
    targets = torch.zeros((B, 1, H, W), dtype=batch[0][3].dtype)
    valid_t = torch.zeros((B, T_max), dtype=torch.bool)
    
    for i, (fire_seq, static, wind, target) in enumerate(batch):
        T = fire_seq.shape[-1]
        
        # Left-pad fire sequence: [1, H, W, T] -> [T_max, 1, H, W]
        fire_seq = fire_seq.permute(3, 0, 1, 2)  # [T, 1, H, W]
        fire_hist[i, -T:] = fire_seq
        
        static_batch[i] = static
        wind_batch[i, -T:] = wind
        targets[i] = target
        valid_t[i, -T:] = True
    
    return fire_hist, static_batch, wind_batch, targets, valid_t

# ----------------
# Training & Validation
# ----------------
def train_one_epoch(model, loader, optimizer, scaler, metrics, device, cfg, epoch, step_offset):
    """Train for one epoch with detailed logging"""
    model.train()
    
    # Reset metrics
    for m in metrics.values():
        m.reset()
    
    # Loss config
    loss_cfg = cfg['model']['loss']
    loss_type = loss_cfg['type']
    
    if loss_type == "focal_tversky":
        focal_weight = _coerce_float(loss_cfg['focal_weight'], 'focal_weight')
        tversky_weight = _coerce_float(loss_cfg['tversky_weight'], 'tversky_weight')
        focal_alpha = _coerce_float(loss_cfg['focal_alpha'], 'focal_alpha')
        focal_gamma = _coerce_float(loss_cfg['focal_gamma'], 'focal_gamma')
        tversky_alpha = _coerce_float(loss_cfg['tversky_alpha'], 'tversky_alpha')
        tversky_beta = _coerce_float(loss_cfg['tversky_beta'], 'tversky_beta')
    elif loss_type == "bce_dice":
        use_dice = loss_cfg.get('use_dice', True)
        bce_weight = _coerce_float(loss_cfg.get('bce_weight', 0.9), 'bce_weight')
        dice_weight = _coerce_float(loss_cfg.get('dice_weight', 0.1), 'dice_weight')
    
    pred_thresh = _coerce_float(cfg['model']['metrics']['pred_thresh'], 'pred_thresh')
    target_thresh = _coerce_float(cfg['model']['metrics'].get('target_thresh', 0.5), 'target_thresh')
    posw_cfg = cfg['model']['metrics'].get('pos_weight', 'auto')
    
    log_metrics_every = cfg['train'].get('log_metrics_every', 50)
    
    total_loss = 0.0
    batch_t_lengths = []
    step = step_offset
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (fire_hist, static, wind, targets, valid_t) in enumerate(pbar):
        # Move to device
        fire_hist = fire_hist.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        wind = wind.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid_t = valid_t.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits = model(fire_hist, static, wind, valid_t)
            
            # Create mask (all pixels valid for raw data)
            mask = torch.ones_like(targets, dtype=torch.bool)
            
            # Compute loss
            if loss_type == "focal_tversky":
                focal = _masked_focal_loss(logits, targets, mask, 
                                          alpha=focal_alpha, gamma=focal_gamma)
                tversky = _masked_tversky_loss(logits, targets, mask,
                                               alpha=tversky_alpha, beta=tversky_beta)
                loss = focal_weight * focal + tversky_weight * tversky
                loss_component_1, loss_component_2 = focal, tversky
                loss_name_1, loss_name_2 = "focal", "tversky"
            elif loss_type == "bce_dice":
                if use_dice:
                    if posw_cfg == "auto":
                        pos_weight = _auto_pos_weight(targets, mask).detach()
                    elif isinstance(posw_cfg, (int, float)):
                        pos_weight = torch.tensor(float(posw_cfg), device=device)
                    else:
                        pos_weight = None
                    
                    bce = _masked_bce(logits, targets, mask, pos_weight=pos_weight)
                    dice = _masked_soft_dice_loss(logits, targets, mask)
                    loss = bce_weight * bce + dice_weight * dice
                    loss_component_1, loss_component_2 = bce, dice
                    loss_name_1, loss_name_2 = "bce", "dice"
                else:
                    if posw_cfg == "auto":
                        pos_weight = _auto_pos_weight(targets, mask).detach()
                    else:
                        pos_weight = None
                    loss = _masked_bce(logits, targets, mask, pos_weight)
                    loss_component_1, loss_component_2 = loss, None
                    loss_name_1, loss_name_2 = "bce", None
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update metrics
        with torch.no_grad():
            preds = torch.sigmoid(logits)
            preds_bin = (preds > pred_thresh).int()
            targets_bin = (targets > target_thresh).int()
            
            p = preds_bin[mask].flatten()
            yb = targets_bin[mask].flatten()
            
            for m in metrics.values():
                m.update(p, yb)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Track sequence length
        batch_t_len = valid_t.sum(dim=1).float().mean().item()
        batch_t_lengths.append(batch_t_len)
        
        # Per-step logging
        if cfg['wandb']['enabled']:
            import wandb
            log_dict = {
                "epoch": epoch,
                "train/loss_step": loss.item(),
                "train/batch_T_avg": batch_t_len,
            }
            
            if loss_component_1 is not None:
                log_dict[f"train/{loss_name_1}_step"] = loss_component_1.item()
            if loss_component_2 is not None:
                log_dict[f"train/{loss_name_2}_step"] = loss_component_2.item()
            
            if step % log_metrics_every == 0:
                step_metrics = {
                    f"train/{k}_step": v.compute().item() 
                    for k, v in metrics.items()
                }
                log_dict.update(step_metrics)
            
            wandb.log(log_dict, step=step)
        
        step += 1
    
    # Compute epoch metrics
    avg_loss = total_loss / len(loader)
    metric_dict = {k: v.compute().item() for k, v in metrics.items()}
    avg_t_len = sum(batch_t_lengths) / len(batch_t_lengths) if batch_t_lengths else 0
    
    return avg_loss, metric_dict, avg_t_len, step

@torch.no_grad()
def validate(model, loader, device, cfg, epoch):
    """Validate the model"""
    model.eval()
    
    # Create metrics
    metrics = {
        'acc': torchmetrics.classification.BinaryAccuracy().to(device),
        'precision': torchmetrics.classification.BinaryPrecision().to(device),
        'recall': torchmetrics.classification.BinaryRecall().to(device),
        'f1': torchmetrics.classification.BinaryF1Score().to(device),
        'iou': torchmetrics.classification.BinaryJaccardIndex().to(device),
    }
    
    # Loss config
    loss_cfg = cfg['model']['loss']
    loss_type = loss_cfg['type']
    
    if loss_type == "focal_tversky":
        focal_weight = _coerce_float(loss_cfg['focal_weight'], 'focal_weight')
        tversky_weight = _coerce_float(loss_cfg['tversky_weight'], 'tversky_weight')
        focal_alpha = _coerce_float(loss_cfg['focal_alpha'], 'focal_alpha')
        focal_gamma = _coerce_float(loss_cfg['focal_gamma'], 'focal_gamma')
        tversky_alpha = _coerce_float(loss_cfg['tversky_alpha'], 'tversky_alpha')
        tversky_beta = _coerce_float(loss_cfg['tversky_beta'], 'tversky_beta')
    elif loss_type == "bce_dice":
        use_dice = loss_cfg.get('use_dice', True)
        bce_weight = _coerce_float(loss_cfg.get('bce_weight', 0.9), 'bce_weight')
        dice_weight = _coerce_float(loss_cfg.get('dice_weight', 0.1), 'dice_weight')
    
    pred_thresh = _coerce_float(cfg['model']['metrics']['pred_thresh'], 'pred_thresh')
    target_thresh = _coerce_float(cfg['model']['metrics'].get('target_thresh', 0.5), 'target_thresh')
    
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for fire_hist, static, wind, targets, valid_t in pbar:
        fire_hist = fire_hist.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        wind = wind.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid_t = valid_t.to(device, non_blocking=True)
        
        with amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits = model(fire_hist, static, wind, valid_t)
            
            mask = torch.ones_like(targets, dtype=torch.bool)
            
            # Compute loss
            if loss_type == "focal_tversky":
                focal_val = _masked_focal_loss(logits, targets, mask,
                                               alpha=focal_alpha, gamma=focal_gamma)
                tversky_val = _masked_tversky_loss(logits, targets, mask,
                                                   alpha=tversky_alpha, beta=tversky_beta)
                vloss = focal_weight * focal_val + tversky_weight * tversky_val
            elif loss_type == "bce_dice":
                if use_dice:
                    bce_val = _masked_bce(logits, targets, mask, None)
                    dice_val = _masked_soft_dice_loss(logits, targets, mask)
                    vloss = bce_weight * bce_val + dice_weight * dice_val
                else:
                    vloss = _masked_bce(logits, targets, mask, None)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        total_loss += vloss.item()
        
        # Update metrics
        preds = torch.sigmoid(logits)
        preds_bin = (preds > pred_thresh).int()
        targets_bin = (targets > target_thresh).int()
        
        p = preds_bin[mask].flatten()
        yb = targets_bin[mask].flatten()
        
        for m in metrics.values():
            m.update(p, yb)
        
        pbar.set_postfix({"loss": f"{vloss.item():.4f}"})
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    metric_dict = {k: v.compute().item() for k, v in metrics.items()}
    
    return avg_loss, metric_dict

# ----------------
# Main
# ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/emberformer_dino.yaml")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                       help="Phase 1: frozen DINO, Phase 2: fine-tune DINO")
    args = parser.parse_args()
    
    # Load config
    cfg = load_cfg(args.config)
    device = _pick_device(args.gpu)
    
    print(f"\n{'='*60}")
    print(f"EmberFormer-DINO Training - Phase {args.phase}")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Set seeds
    seed = cfg['global']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Initialize wandb
    wandb_run = None
    if cfg['wandb']['enabled']:
        cfg['wandb']['run_name'] = f"dino-phase{args.phase}-{time.strftime('%m%d-%H%M')}"
        cfg['wandb']['tags'] = cfg['wandb'].get('tags', []) + [f"phase{args.phase}"]
        
        context = {
            'script': 'train_dino',
            'model': cfg['train']['model'],
            'phase': args.phase,
        }
        
        wandb_run = init_wandb(cfg=cfg, context=context)
        print(f"✓ W&B initialized: {wandb_run.name}\n")
    
    # Load datasets
    data_dir = os.path.expanduser(cfg['data']['data_dir'])
    
    print("Loading datasets...")
    full_dataset = RawFireDataset(data_dir, sequence_length=cfg['data']['sequence_length'])
    
    # Split into train/val
    total_samples = len(full_dataset.samples)
    train_size = int(cfg['split']['train'] * total_samples)
    
    train_dataset = RawFireDataset(data_dir, sequence_length=cfg['data']['sequence_length'])
    train_dataset.samples = full_dataset.samples[:train_size]
    
    val_dataset = RawFireDataset(data_dir, sequence_length=cfg['data']['sequence_length'])
    val_dataset.samples = full_dataset.samples[train_size:]
    
    print(f"  Total: {total_samples} samples")
    print(f"  Train: {len(train_dataset.samples)} samples")
    print(f"  Val: {len(val_dataset.samples)} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=cfg['data']['shuffle'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        collate_fn=collate_raw_dino,
        drop_last=cfg['data'].get('drop_last', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        collate_fn=collate_raw_dino
    )
    
    # Get static channels from first sample
    first_sample = train_dataset[0]
    static_channels = first_sample[1].shape[0]
    print(f"Static channels: {static_channels}")
    
    # Create model
    print("\nCreating EmberFormerDINO...")
    
    # Determine freeze strategy and LR based on phase
    if args.phase == 1:
        freeze_dino = cfg['model']['dino']['freeze_fire']
        lr_main = _coerce_float(cfg['train']['lr'], 'lr')
        lr_dino = 0.0  # Not used when frozen
    else:
        freeze_dino = False
        lr_main = _coerce_float(cfg['train']['lr'], 'lr')
        lr_dino = _coerce_float(cfg['train'].get('finetune', {}).get('lr', 1e-5), 'finetune.lr')
    
    model = EmberFormerDINO(
        dino_model=cfg['model']['dino']['model_name'],
        freeze_dino=freeze_dino,
        d_model=cfg['model']['temporal']['d_model'],
        nhead=cfg['model']['temporal']['nhead'],
        num_layers=cfg['model']['temporal']['num_layers'],
        dim_feedforward=cfg['model']['temporal']['dim_feedforward'],
        dropout=_coerce_float(cfg['model']['temporal']['dropout'], 'dropout'),
        spatial_hidden=cfg['model']['spatial']['hidden_channels'],
        patch_size=cfg['model']['refinement']['patch_size'],
        static_channels=static_channels,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params: {total_params - trainable_params:,}")
    print(f"  Phase: {args.phase} ({'Frozen DINO' if freeze_dino else 'Fine-tuning DINO'})")
    
    # Create optimizer with differential learning rates
    weight_decay = _coerce_float(cfg['train']['weight_decay'], 'weight_decay')
    
    if args.phase == 2:
        # Phase 2: Differential LRs for DINO vs other components
        dino_params = list(model.fire_encoder.parameters())
        other_params = [p for n, p in model.named_parameters() 
                       if not n.startswith('fire_encoder.') and p.requires_grad]
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': lr_main},
            {'params': dino_params, 'lr': lr_dino}
        ], weight_decay=weight_decay)
        
        print(f"  LR (main components): {lr_main}")
        print(f"  LR (DINO fine-tune): {lr_dino}")
    else:
        # Phase 1: Single LR for all trainable params
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_main,
            weight_decay=weight_decay
        )
        print(f"  LR: {lr_main}")
    
    # Loss function info
    loss_cfg = cfg['model']['loss']
    print(f"  Loss: {loss_cfg['type']}\n")
    
    # Create training metrics
    train_metrics = {
        'acc': torchmetrics.classification.BinaryAccuracy().to(device),
        'precision': torchmetrics.classification.BinaryPrecision().to(device),
        'recall': torchmetrics.classification.BinaryRecall().to(device),
        'f1': torchmetrics.classification.BinaryF1Score().to(device),
        'iou': torchmetrics.classification.BinaryJaccardIndex().to(device),
    }
    
    # Early stopping
    early_stopping = None
    if cfg['train']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=cfg['train']['early_stopping']['patience'],
            min_delta=_coerce_float(cfg['train']['early_stopping']['min_delta'], 'min_delta'),
            mode=cfg['train']['early_stopping']['mode']
        )
        monitor_metric = cfg['train']['early_stopping']['monitor']
        print(f"Early stopping: patience={early_stopping.patience}, monitoring {monitor_metric}")
    
    # Checkpoint setup
    save_checkpoints = cfg['train'].get('save_checkpoints', True)
    if save_checkpoints:
        ckpt_dir = pathlib.Path(cfg['train'].get('checkpoint_dir', 'checkpoints'))
        ckpt_dir.mkdir(exist_ok=True)
        print(f"Checkpoints: {ckpt_dir}\n")
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # Training loop
    epochs = cfg['train']['epochs']
    best_f1 = 0.0
    global_step = 0
    
    print(f"Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_metric_dict, avg_t_len, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, train_metrics,
            device, cfg, epoch, global_step
        )
        
        # Validate
        val_loss, val_metric_dict = validate(
            model, val_loader, device, cfg, epoch
        )
        
        # Format loss name for console output
        loss_type = cfg['model']['loss']['type']
        if loss_type == "focal_tversky":
            loss_str = "Focal+Tversky"
        elif cfg['model']['loss'].get('use_dice', False):
            loss_str = "BCE+Dice"
        else:
            loss_str = "BCE"
        
        # Console output
        print(f"[Epoch {epoch+1:2d}/{epochs}] {loss_str} | "
              f"train: loss={train_loss:.4f} f1={train_metric_dict['f1']:.3f} prec={train_metric_dict['precision']:.3f} | "
              f"val: loss={val_loss:.4f} f1={val_metric_dict['f1']:.3f} prec={val_metric_dict['precision']:.3f} "
              f"iou={val_metric_dict['iou']:.3f}")
        
        # Log to wandb
        if wandb_run:
            import wandb
            
            all_metrics = {
                'epoch': epoch,
                'train/loss_epoch': train_loss,
                'train/T_avg_epoch': avg_t_len,
                'val/loss_epoch': val_loss,
            }
            
            # Add train metrics with epoch suffix
            for k, v in train_metric_dict.items():
                all_metrics[f'train/{k}_epoch'] = v
            
            # Add val metrics with epoch suffix
            for k, v in val_metric_dict.items():
                all_metrics[f'val/{k}_epoch'] = v
            
            # Log learning rates
            all_metrics['train/lr_main'] = optimizer.param_groups[0]['lr']
            if len(optimizer.param_groups) > 1:
                all_metrics['train/lr_dino'] = optimizer.param_groups[1]['lr']
            
            wandb.log(all_metrics, step=global_step)
        
        # Save best model
        if val_metric_dict['f1'] > best_f1:
            best_f1 = val_metric_dict['f1']
            if save_checkpoints:
                checkpoint_path = ckpt_dir / f"dino_phase{args.phase}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_f1,
                    'val_iou': val_metric_dict['iou'],
                    'config': cfg,
                    'phase': args.phase,
                }, checkpoint_path)
                print(f"  ✓ Saved best checkpoint: {checkpoint_path}")
        
        # Early stopping
        if early_stopping:
            # Extract monitored metric
            if monitor_metric == "val/loss":
                metric_value = val_loss
            elif monitor_metric == "val/f1":
                metric_value = val_metric_dict['f1']
            elif monitor_metric == "val/iou":
                metric_value = val_metric_dict['iou']
            else:
                metric_value = val_metric_dict['f1']
            
            if early_stopping(epoch, metric_value):
                print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best {monitor_metric}: {early_stopping.best_value:.4f} at epoch {early_stopping.best_epoch+1}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"{'='*60}\n")
    
    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()
