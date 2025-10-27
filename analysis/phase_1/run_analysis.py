#!/usr/bin/env python
"""
Phase 1 Interpretability Analysis for EmberFormer-DINO

Runs all interpretability analyses on the Phase 1 checkpoint (frozen DINO).

Usage:
    cd ~/Code/emberformer
    python analysis/phase_1/run_analysis.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import argparse
from datetime import datetime

from models.emberformer import EmberFormerDINO
from data import RawFireDataset
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF

# Import analysis modules
from .spatial_importance import analyze_spatial_importance
from .temporal_importance import analyze_temporal_importance
from .feature_ablation import analyze_feature_importance
from .wind_analysis import analyze_wind_direction
from .extreme_events import analyze_extreme_events

def load_model_and_data(checkpoint_path, config_path, device='cuda'):
    """Load trained model and validation dataset"""
    
    print("=" * 70)
    print("Loading Phase 1 Model and Data")
    print("=" * 70)
    
    # Expand paths
    checkpoint_path = os.path.expanduser(checkpoint_path)
    config_path = os.path.expanduser(config_path)
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"  ✓ Epoch: {checkpoint['epoch']}")
    print(f"  ✓ Val F1: {checkpoint['val_f1']:.4f}")
    print(f"  ✓ Val IoU: {checkpoint['val_iou']:.4f}")
    print(f"  ✓ Phase: {checkpoint['phase']}")
    
    # Create resize transform (matching training)
    target_size = cfg['data'].get('resize_to', 406)
    
    class ResizeTransform:
        def __init__(self, size):
            self.size = size
        
        def __call__(self, img):
            return TF.resize(img, [self.size, self.size],
                           interpolation=TF.InterpolationMode.BILINEAR,
                           antialias=True)
    
    transform = ResizeTransform(target_size)
    
    # Load dataset
    print(f"\n📊 Loading validation dataset...")
    data_dir = os.path.expanduser(cfg['data']['data_dir'])
    
    full_dataset = RawFireDataset(data_dir,
                                   sequence_length=cfg['data']['sequence_length'],
                                   transform=transform)
    
    # Get validation split
    total_samples = len(full_dataset.samples)
    train_size = int(cfg['split']['train'] * total_samples)
    val_indices = list(range(train_size, total_samples))
    
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"  ✓ Validation samples: {len(val_dataset)}")
    
    # Get static channels from first sample
    first_sample = val_dataset[0]
    static_channels = first_sample[1].shape[0]
    
    # Create model
    print(f"\n🔧 Creating model...")
    model = EmberFormerDINO(
        dino_model=cfg['model']['dino']['model_name'],
        freeze_dino=False,  # False to inspect all layers
        d_model=cfg['model']['temporal']['d_model'],
        nhead=cfg['model']['temporal']['nhead'],
        num_layers=cfg['model']['temporal']['num_layers'],
        dim_feedforward=cfg['model']['temporal']['dim_feedforward'],
        dropout=float(cfg['model']['temporal']['dropout']),
        spatial_hidden=cfg['model']['spatial']['hidden_channels'],
        patch_size=cfg['model']['refinement']['patch_size'],
        static_channels=static_channels,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    
    return model, val_dataset, cfg, device


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Interpretability Analysis")
    parser.add_argument('--checkpoint', type=str, 
                       default='~/data/emberformer/checkpoints/dino_phase1_best.pt',
                       help='Path to Phase 1 checkpoint')
    parser.add_argument('--config', type=str,
                       default='configs/emberformer_dino.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str,
                       default='analysis/phase_1',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples for statistical analyses')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--skip', type=str, nargs='*',
                       choices=['spatial', 'temporal', 'features', 'wind', 'extreme'],
                       help='Skip specific analyses')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save analysis metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'config': args.config,
        'num_samples': args.num_samples,
        'device': args.device,
    }
    
    with open(output_dir / 'analysis_metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)
    
    print("\n" + "=" * 70)
    print("EmberFormer-DINO Phase 1 Interpretability Analysis")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")
    
    # Load model and data
    model, val_dataset, cfg, device = load_model_and_data(
        args.checkpoint, args.config, args.device
    )
    
    skip = args.skip or []
    
    # Analysis 1: Spatial Importance
    if 'spatial' not in skip:
        print("\n" + "=" * 70)
        print("Analysis 1: Spatial Importance Maps")
        print("=" * 70)
        analyze_spatial_importance(
            model, val_dataset, device,
            output_dir=output_dir / 'spatial',
            num_samples=10
        )
    
    # Analysis 2: Temporal Importance
    if 'temporal' not in skip:
        print("\n" + "=" * 70)
        print("Analysis 2: Temporal Importance Patterns")
        print("=" * 70)
        analyze_temporal_importance(
            model, val_dataset, device,
            output_dir=output_dir / 'temporal',
            num_samples=args.num_samples
        )
    
    # Analysis 3: Feature Ablation
    if 'features' not in skip:
        print("\n" + "=" * 70)
        print("Analysis 3: Static Feature Importance (Ablation)")
        print("=" * 70)
        analyze_feature_importance(
            model, val_dataset, device,
            output_dir=output_dir / 'features',
            num_samples=1000  # Use subset for speed
        )
    
    # Analysis 4: Wind Direction
    if 'wind' not in skip:
        print("\n" + "=" * 70)
        print("Analysis 4: Wind-Fire Alignment")
        print("=" * 70)
        analyze_wind_direction(
            model, val_dataset, device,
            output_dir=output_dir / 'wind',
            num_samples=args.num_samples
        )
    
    # Analysis 5: Extreme Events
    if 'extreme' not in skip:
        print("\n" + "=" * 70)
        print("Analysis 5: Extreme Event Analysis")
        print("=" * 70)
        analyze_extreme_events(
            model, val_dataset, device,
            output_dir=output_dir / 'extreme_events',
            percentile=95,
            num_visualize=10
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - spatial/*.png (importance maps)")
    print("  - temporal/temporal_importance.png")
    print("  - features/feature_importance.png")
    print("  - wind/wind_alignment.png")
    print("  - extreme_events/*.png (top 10 events)")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
