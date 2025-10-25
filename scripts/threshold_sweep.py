#!/usr/bin/env python3
"""
Post-training threshold sweep on validation set.

Loads a trained checkpoint, runs inference once to get probabilities,
then sweeps classification thresholds to find optimal precision/recall tradeoff.

Usage:
    uv run python scripts/threshold_sweep.py \\
        --checkpoint ~/data/emberformer/checkpoints/dino_phase2_expanding_best.pt \\
        --config configs/emberformer_dino_expanding.yaml \\
        --threshold-min 0.5 \\
        --threshold-max 0.75 \\
        --threshold-step 0.05 \\
        --gpu 0
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import RawFireDataset
from models import EmberFormerDINO


def collate_raw_dino(batch):
    """
    Collate function for DINO training with variable-length sequences
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


def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_f1' in checkpoint:
        print(f"  ✓ Checkpoint Val F1: {checkpoint['val_f1']:.4f}")
    
    return model


def collect_predictions(model, dataloader, device, target_thresh=0.05):
    """
    Run inference on entire dataset and collect probabilities + targets.
    
    Returns:
        all_probs: List of probability maps [H, W]
        all_targets: List of binary target maps [H, W]
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    print(f"\n🔮 Running inference on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch from collate_raw_dino
            fire_hist, static, wind_hist, targets, valid_t = batch
            
            # Move to device
            fire_hist = fire_hist.to(device)    # [B, T, 1, H, W]
            static = static.to(device)          # [B, Cs, H, W]
            wind_hist = wind_hist.to(device)    # [B, T, 2]
            valid_t = valid_t.to(device)        # [B, T]
            targets = targets.to(device)        # [B, 1, H, W]
            
            # Forward pass
            logits = model(fire_hist, static, wind_hist, valid_t)  # [B, 1, H, W]
            probs = torch.sigmoid(logits)                          # [B, 1, H, W]
            
            # Collect probabilities and targets
            for i in range(probs.size(0)):
                all_probs.append(probs[i, 0].cpu().numpy())      # [H, W]
                all_targets.append((targets[i, 0].cpu().numpy() > target_thresh).astype(np.float32))  # [H, W]
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    print(f"  ✓ Collected {len(all_probs)} predictions")
    return all_probs, all_targets


def sweep_thresholds(all_probs, all_targets, thresholds, device):
    """
    Sweep through thresholds and compute metrics.
    
    Args:
        all_probs: List of probability maps [H, W]
        all_targets: List of binary target maps [H, W]
        thresholds: List of thresholds to try
        device: torch device
        
    Returns:
        results: List of dicts with metrics per threshold
    """
    print(f"\n📊 Sweeping {len(thresholds)} thresholds...")
    
    results = []
    
    for thresh in thresholds:
        # Create metrics
        metrics = {
            'acc': BinaryAccuracy().to(device),
            'precision': BinaryPrecision().to(device),
            'recall': BinaryRecall().to(device),
            'f1': BinaryF1Score().to(device),
            'iou': BinaryJaccardIndex().to(device),
        }
        
        # Apply threshold and compute metrics
        for prob_map, target_map in zip(all_probs, all_targets):
            # Binarize predictions
            pred_bin = (prob_map > thresh).astype(np.float32)  # [H, W]
            
            # Convert to tensors and flatten
            pred_tensor = torch.from_numpy(pred_bin).flatten().to(device)
            target_tensor = torch.from_numpy(target_map).flatten().to(device)
            
            # Update metrics
            for m in metrics.values():
                m.update(pred_tensor, target_tensor)
        
        # Compute final metrics
        result = {
            'threshold': thresh,
            'accuracy': metrics['acc'].compute().item(),
            'precision': metrics['precision'].compute().item(),
            'recall': metrics['recall'].compute().item(),
            'f1': metrics['f1'].compute().item(),
            'iou': metrics['iou'].compute().item(),
        }
        results.append(result)
        
        print(f"  thresh={thresh:.3f}: F1={result['f1']:.4f}, Prec={result['precision']:.4f}, Rec={result['recall']:.4f}")
    
    return results


def save_results(results, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved results to: {output_path}")
    return df


def print_summary(df):
    """Print summary of best thresholds."""
    print("\n" + "="*80)
    print("THRESHOLD SWEEP SUMMARY")
    print("="*80)
    
    # Best F1
    best_f1_row = df.loc[df['f1'].idxmax()]
    print(f"\n🏆 Best F1 Score: {best_f1_row['f1']:.4f} at threshold={best_f1_row['threshold']:.3f}")
    print(f"   Precision: {best_f1_row['precision']:.4f}")
    print(f"   Recall: {best_f1_row['recall']:.4f}")
    print(f"   IoU: {best_f1_row['iou']:.4f}")
    
    # Best Precision
    best_prec_row = df.loc[df['precision'].idxmax()]
    print(f"\n🎯 Best Precision: {best_prec_row['precision']:.4f} at threshold={best_prec_row['threshold']:.3f}")
    print(f"   F1: {best_prec_row['f1']:.4f}")
    print(f"   Recall: {best_prec_row['recall']:.4f}")
    
    # Best IoU
    best_iou_row = df.loc[df['iou'].idxmax()]
    print(f"\n📐 Best IoU: {best_iou_row['iou']:.4f} at threshold={best_iou_row['threshold']:.3f}")
    print(f"   F1: {best_iou_row['f1']:.4f}")
    print(f"   Precision: {best_iou_row['precision']:.4f}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Post-training threshold sweep")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml file")
    parser.add_argument("--threshold-min", type=float, default=0.5, help="Minimum threshold")
    parser.add_argument("--threshold-max", type=float, default=0.75, help="Maximum threshold")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Threshold step size")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: auto)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: from config)")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Initialize W&B if requested
    wandb_run = None
    if args.wandb:
        checkpoint_name = Path(args.checkpoint).stem
        wandb_run = wandb.init(
            project=cfg['wandb'].get('project', 'emberformer'),
            entity=cfg['wandb'].get('entity', None),
            name=f"threshold-sweep-{checkpoint_name}",
            tags=["threshold-sweep", "post-training"],
            config={
                'checkpoint': args.checkpoint,
                'threshold_min': args.threshold_min,
                'threshold_max': args.threshold_max,
                'threshold_step': args.threshold_step,
            }
        )
        print("📊 W&B logging enabled")
    
    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Generate thresholds
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step/2, args.threshold_step)
    print(f"🎚️  Thresholds: {len(thresholds)} values from {args.threshold_min} to {args.threshold_max}")
    
    # Build model
    print("\n🏗️  Building model...")
    model = EmberFormerDINO(cfg).to(device)
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    checkpoint_path = os.path.expanduser(args.checkpoint)
    model = load_checkpoint(checkpoint_path, model, device)
    
    # Load validation dataset
    print("\n📂 Loading validation dataset...")
    data_dir = os.path.expanduser(cfg['data']['data_dir'])
    sequence_length = cfg['data']['sequence_length']
    resize_to = cfg['data'].get('resize_to', None)
    
    dataset = RawFireDataset(
        root_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        use_pixel_data=True,
        resize_to=resize_to,
        fire_channel=cfg['encoding']['fire_channel'],
        fire_value=cfg['encoding']['fire_value'],
        isochrone_channel=cfg['encoding']['isochrone_channel'],
        isochrone_value=cfg['encoding']['isochrone_value'],
    )
    
    batch_size = args.batch_size if args.batch_size else cfg['data']['batch_size']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_raw_dino,
    )
    
    print(f"  ✓ Validation samples: {len(dataset)}")
    print(f"  ✓ Batch size: {batch_size}")
    print(f"  ✓ Batches: {len(dataloader)}")
    
    # Collect predictions
    target_thresh = cfg['model']['metrics'].get('target_thresh', 0.05)
    all_probs, all_targets = collect_predictions(model, dataloader, device, target_thresh)
    
    # Sweep thresholds
    results = sweep_thresholds(all_probs, all_targets, thresholds, device)
    
    # Log to W&B
    if wandb_run:
        for result in results:
            wandb.log({
                'threshold': result['threshold'],
                'val/accuracy': result['accuracy'],
                'val/precision': result['precision'],
                'val/recall': result['recall'],
                'val/f1': result['f1'],
                'val/iou': result['iou'],
            })
        
        # Create summary table
        wandb.log({"threshold_results": wandb.Table(dataframe=pd.DataFrame(results))})
    
    # Save results
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        output_path = f"results/threshold_sweep_{checkpoint_name}.csv"
    else:
        output_path = args.output
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df = save_results(results, output_path)
    
    # Print summary
    print_summary(df)
    
    # Finish W&B
    if wandb_run:
        wandb.finish()
        print("✓ W&B run finished")


if __name__ == "__main__":
    main()
