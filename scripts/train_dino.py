"""
Training script for EmberFormer-DINO

Uses:
- RawFireDataset with pixel-level fire frames
- EmberFormerDINO model (DINO encoder + temporal transformer + simple decoder)
- Frozen DINO in Phase 1, optional fine-tuning in Phase 2
- Single learning rate (or differential for DINO fine-tuning)
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
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch, value):
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

# ----------------
# Loss Functions
# ----------------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * bce_loss
        return loss.mean()

class TverskyLoss(nn.Module):
    """Tversky Loss - generalizes Dice loss with control over FP/FN"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    """Focal + Tversky combined loss"""
    def __init__(self, focal_weight=0.7, tversky_weight=0.3, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
    
    def forward(self, logits, targets):
        return (self.focal_weight * self.focal(logits, targets) + 
                self.tversky_weight * self.tversky(logits, targets))

# ----------------
# Custom collate for RawFireDataset
# ----------------
def collate_raw_dino(batch):
    """
    Collate function for DINO training with variable-length sequences
    
    Input: List of (fire_seq, static, wind, target) tuples
    - fire_seq: [1, H, W, T] variable T
    - static: [Cs, H, W]
    - wind: [T, 2]
    - target: [1, H, W]
    
    Output: Batched tensors with left-padding
    - fire_hist: [B, T_max, 1, H, W]
    - static: [B, Cs, H, W]
    - wind: [B, T_max, 2]
    - targets: [B, 1, H, W]
    - valid_t: [B, T_max] boolean mask
    """
    # Find max sequence length
    T_max = max(item[0].shape[-1] for item in batch)
    B = len(batch)
    
    # Get dimensions from first sample
    _, H, W, _ = batch[0][0].shape
    Cs = batch[0][1].shape[0]
    
    # Preallocate tensors
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
        
        # Static (no padding needed)
        static_batch[i] = static
        
        # Left-pad wind: [T, 2] -> [T_max, 2]
        wind_batch[i, -T:] = wind
        
        # Target
        targets[i] = target
        
        # Valid mask (1 for real timesteps, 0 for padding)
        valid_t[i, -T:] = True
    
    return fire_hist, static_batch, wind_batch, targets, valid_t

# ----------------
# Training & Validation
# ----------------
def train_one_epoch(model, loader, optimizer, criterion, metrics, device, scaler, cfg):
    """Train for one epoch"""
    model.train()
    
    # Reset metrics
    for m in metrics.values():
        m.reset()
    
    total_loss = 0.0
    
    for batch_idx, (fire_hist, static, wind, targets, valid_t) in enumerate(tqdm(loader, desc="Training")):
        # Move to device
        fire_hist = fire_hist.to(device)
        static = static.to(device)
        wind = wind.to(device)
        targets = targets.to(device)
        valid_t = valid_t.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits = model(fire_hist, static, wind, valid_t)
            loss = criterion(logits, targets)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update metrics
        preds = torch.sigmoid(logits)
        for m in metrics.values():
            m.update(preds, targets.long())
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    metric_dict = {k: v.compute().item() for k, v in metrics.items()}
    
    return avg_loss, metric_dict

@torch.no_grad()
def validate(model, loader, criterion, metrics, device, cfg):
    """Validate the model"""
    model.eval()
    
    # Reset metrics
    for m in metrics.values():
        m.reset()
    
    total_loss = 0.0
    
    for fire_hist, static, wind, targets, valid_t in tqdm(loader, desc="Validation"):
        # Move to device
        fire_hist = fire_hist.to(device)
        static = static.to(device)
        wind = wind.to(device)
        targets = targets.to(device)
        valid_t = valid_t.to(device)
        
        # Forward pass
        with amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits = model(fire_hist, static, wind, valid_t)
            loss = criterion(logits, targets)
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update metrics
        preds = torch.sigmoid(logits)
        for m in metrics.values():
            m.update(preds, targets.long())
    
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
        # Update config for wandb
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
    train_dataset = RawFireDataset(data_dir, sequence_length=cfg['data']['sequence_length'])
    
    # Split into train/val
    total_samples = len(train_dataset.samples)
    train_size = int(cfg['split']['train'] * total_samples)
    val_size = total_samples - train_size
    
    train_dataset.samples = train_dataset.samples[:train_size]
    val_dataset = RawFireDataset(data_dir, sequence_length=cfg['data']['sequence_length'])
    val_dataset.samples = train_dataset.samples[train_size:]
    
    print(f"  Train: {len(train_dataset.samples)} samples")
    print(f"  Val: {len(val_dataset.samples)} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=cfg['data']['shuffle'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        collate_fn=collate_raw_dino
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
    
    # Determine if we should freeze DINO based on phase
    if args.phase == 1:
        freeze_dino = cfg['model']['dino']['freeze_fire']
        lr = cfg['train']['lr']
    else:
        freeze_dino = False
        lr = cfg['train'].get('finetune', {}).get('lr', 1e-5)
    
    model = EmberFormerDINO(
        dino_model=cfg['model']['dino']['model_name'],
        freeze_dino=freeze_dino,
        d_model=cfg['model']['temporal']['d_model'],
        nhead=cfg['model']['temporal']['nhead'],
        num_layers=cfg['model']['temporal']['num_layers'],
        dim_feedforward=cfg['model']['temporal']['dim_feedforward'],
        dropout=cfg['model']['temporal']['dropout'],
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
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg['train']['weight_decay']
    )
    
    # Create loss function
    loss_cfg = cfg['model']['loss']
    if loss_cfg['type'] == 'focal_tversky':
        criterion = CombinedLoss(
            focal_weight=loss_cfg['focal_weight'],
            tversky_weight=loss_cfg['tversky_weight'],
            focal_alpha=loss_cfg['focal_alpha'],
            focal_gamma=loss_cfg['focal_gamma'],
            tversky_alpha=loss_cfg['tversky_alpha'],
            tversky_beta=loss_cfg['tversky_beta'],
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_cfg['type']}")
    
    print(f"  Loss: {loss_cfg['type']}\n")
    
    # Create metrics
    def _make_metrics():
        return {
            'acc': torchmetrics.classification.BinaryAccuracy().to(device),
            'precision': torchmetrics.classification.BinaryPrecision().to(device),
            'recall': torchmetrics.classification.BinaryRecall().to(device),
            'f1': torchmetrics.classification.BinaryF1Score().to(device),
            'iou': torchmetrics.classification.BinaryJaccardIndex().to(device),
        }
    
    metrics_train = _make_metrics()
    metrics_val = _make_metrics()
    
    # Early stopping
    early_stopping = None
    if cfg['train']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=cfg['train']['early_stopping']['patience'],
            min_delta=cfg['train']['early_stopping']['min_delta'],
            mode=cfg['train']['early_stopping']['mode']
        )
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # Training loop
    epochs = cfg['train']['epochs']
    best_f1 = 0.0
    
    print(f"Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            metrics_train, device, scaler, cfg
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion,
            metrics_val, device, cfg
        )
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f} | F1: {train_metrics['f1']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f}")
        
        # Log to wandb
        if wandb_run:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/f1': train_metrics['f1'],
                'train/iou': train_metrics['iou'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'val/loss': val_loss,
                'val/f1': val_metrics['f1'],
                'val/iou': val_metrics['iou'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            if cfg['train'].get('save_checkpoints', False):
                checkpoint_dir = pathlib.Path(cfg['train'].get('checkpoint_dir', 'checkpoints'))
                checkpoint_dir.mkdir(exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"dino_phase{args.phase}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_f1,
                    'config': cfg,
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if early_stopping:
            monitor_metric = val_metrics[cfg['train']['early_stopping']['monitor'].split('/')[-1]]
            if early_stopping(epoch, monitor_metric):
                print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best {cfg['train']['early_stopping']['monitor']}: {early_stopping.best_value:.4f} at epoch {early_stopping.best_epoch+1}")
                break
        
        print()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"{'='*60}\n")
    
    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()
