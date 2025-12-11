#!/usr/bin/env python3
"""
W&B Sweep-compatible threshold evaluation script.

This runs inference ONCE per threshold (inefficient but works with wandb sweep).
For faster results, use threshold_sweep.py instead.

Usage with W&B Sweep:
    wandb sweep configs/sweep_threshold.yaml
    wandb agent <sweep-id>

Manual usage:
    uv run python scripts/threshold_sweep_wandb.py \\
        --checkpoint ~/data/emberformer/checkpoints/dino_phase2_expanding_best.pt \\
        --config configs/emberformer_dino_expanding.yaml \\
        --threshold 0.65 \\
        --gpu 0
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import RawFireDataset
from models import EmberFormerDINO


def collate_raw_dino(batch):
    """Collate function for DINO training with variable-length sequences"""
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
        fire_seq = fire_seq.permute(3, 0, 1, 2)
        fire_hist[i, -T:] = fire_seq
        static_batch[i] = static
        wind_batch[i, -T:] = wind
        targets[i] = target
        valid_t[i, -T:] = True
    
    return fire_hist, static_batch, wind_batch, targets, valid_t


def main():
    parser = argparse.ArgumentParser(description="Evaluate single threshold (W&B sweep compatible)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml file")
    parser.add_argument("--threshold", type=float, required=True, help="Classification threshold to evaluate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Initialize W&B
    checkpoint_name = Path(args.checkpoint).stem
    wandb.init(
        project=cfg['wandb'].get('project', 'emberformer'),
        entity=cfg['wandb'].get('entity', None),
        name=f"thresh-{args.threshold:.2f}-{checkpoint_name}",
        tags=["threshold-sweep", "post-training"],
        config={
            'checkpoint': args.checkpoint,
            'threshold': args.threshold,
        }
    )
    
    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Build model
    print(f"\n🏗️  Building model...")
    model = EmberFormerDINO(
        dino_model=cfg['model']['dino']['model_name'],
        freeze_dino=cfg['model']['dino']['freeze_fire'],
        d_model=cfg['model']['temporal']['d_model'],
        nhead=cfg['model']['temporal']['nhead'],
        num_layers=cfg['model']['temporal']['num_layers'],
        dim_feedforward=cfg['model']['temporal']['dim_feedforward'],
        dropout=cfg['model']['temporal']['dropout'],
        spatial_hidden=cfg['model']['spatial']['hidden_channels'],
        patch_size=cfg['model']['refinement']['patch_size'],
        static_channels=cfg['static']['num_channels'],
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.expanduser(args.checkpoint)
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation dataset
    print(f"📂 Loading validation dataset...")
    data_dir = os.path.expanduser(cfg['data']['data_dir'])
    sequence_length = cfg['data']['sequence_length']
    target_size = cfg['data']['resize_to']
    
    import torchvision.transforms.functional as TF
    
    class ResizeTransform:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return TF.resize(img, [self.size, self.size],
                           interpolation=TF.InterpolationMode.BILINEAR,
                           antialias=True)
    
    transform = ResizeTransform(target_size)
    
    full_dataset = RawFireDataset(
        data_dir,
        sequence_length=sequence_length,
        transform=transform,
    )
    
    # Split dataset
    from torch.utils.data import random_split
    train_size = int(cfg['split']['train'] * len(full_dataset))
    val_size = int(cfg['split']['val'] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(cfg['split']['seed'])
    _, dataset, _ = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_raw_dino,
    )
    
    print(f"  ✓ Validation samples: {len(dataset)}")
    
    # Create metrics
    metrics = {
        'acc': BinaryAccuracy().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'f1': BinaryF1Score().to(device),
        'iou': BinaryJaccardIndex().to(device),
    }
    
    # Run inference
    print(f"\n🔮 Evaluating with threshold={args.threshold:.3f}...")
    target_thresh = cfg['model']['metrics'].get('target_thresh', 0.05)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            fire_hist, static, wind_hist, targets, valid_t = batch
            
            # Move to device
            fire_hist = fire_hist.to(device)
            static = static.to(device)
            wind_hist = wind_hist.to(device)
            valid_t = valid_t.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(fire_hist, static, wind_hist, valid_t)
            probs = torch.sigmoid(logits)
            
            # Binarize with threshold
            preds_bin = (probs > args.threshold).int()
            targets_bin = (targets > target_thresh).int()
            
            # Flatten and update metrics
            p = preds_bin.flatten()
            t = targets_bin.flatten()
            
            for m in metrics.values():
                m.update(p, t)
    
    # Compute final metrics
    results = {
        'val/accuracy': metrics['acc'].compute().item(),
        'val/precision': metrics['precision'].compute().item(),
        'val/recall': metrics['recall'].compute().item(),
        'val/f1': metrics['f1'].compute().item(),
        'val/iou': metrics['iou'].compute().item(),
        'threshold': args.threshold,
    }
    
    # Log to W&B
    wandb.log(results)
    
    print(f"\n📊 Results:")
    print(f"  F1:        {results['val/f1']:.4f}")
    print(f"  Precision: {results['val/precision']:.4f}")
    print(f"  Recall:    {results['val/recall']:.4f}")
    print(f"  IoU:       {results['val/iou']:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
