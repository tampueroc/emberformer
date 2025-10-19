#!/usr/bin/env python
"""
Phase 1 Advanced Interpretability Analysis for EmberFormer-DINO

Runs the new thesis-ready analyses with timestamped outputs.

Usage:
    cd ~/Code/emberformer
    python analysis/phase_1/run_advanced_analysis.py
    
    # Or with custom run name
    python analysis/phase_1/run_advanced_analysis.py --run-name frozen_dino_baseline
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

# Import advanced analysis modules
from advanced.attention_entropy import run_attention_entropy_analysis
from advanced.thesis_utils import ThesisOutputManager


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
    parser = argparse.ArgumentParser(description="Phase 1 Advanced Analysis (Thesis-Ready)")
    parser.add_argument('--checkpoint', type=str, 
                       default='~/data/emberformer/checkpoints/dino_phase1_best.pt',
                       help='Path to Phase 1 checkpoint')
    parser.add_argument('--config', type=str,
                       default='configs/emberformer_dino.yaml',
                       help='Path to config file')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom run name (default: timestamp)')
    parser.add_argument('--phase', type=int, default=1,
                       help='Phase number for output directory')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--analyses', type=str, nargs='+',
                       choices=['entropy', 'trajectory', 'heads', 'uncertainty', 
                                'history', 'embeddings', 'gradients'],
                       default=['entropy'],
                       help='Which analyses to run')
    
    args = parser.parse_args()
    
    # Create output manager
    output_manager = ThesisOutputManager(
        phase=args.phase,
        use_timestamp=(args.run_name is None),
        custom_run_name=args.run_name
    )
    
    print("\n" + "=" * 70)
    print("EmberFormer-DINO Advanced Interpretability Analysis")
    print("=" * 70)
    print(f"Phase: {args.phase}")
    print(f"Output: {output_manager.output_dir}")
    print(f"Analyses: {', '.join(args.analyses)}")
    print("=" * 70 + "\n")
    
    # Load model and data
    model, val_dataset, cfg, device = load_model_and_data(
        args.checkpoint, args.config, args.device
    )
    
    # Create dataloader
    dataloader = DataLoader(
        val_dataset, 
        batch_size=1,  # Process one at a time for analysis
        shuffle=False,
        num_workers=0
    )
    
    # Dataset name for metadata
    dataset_name = f"validation_set_{len(val_dataset)}_samples"
    
    # Run requested analyses
    results = {}
    
    if 'entropy' in args.analyses:
        print("\n" + "=" * 70)
        print("TIER 1.1: Attention Entropy Analysis")
        print("=" * 70)
        results['entropy'] = run_attention_entropy_analysis(
            model,
            dataloader,
            output_manager,
            args.checkpoint,
            dataset_name,
            device=device,
            max_samples=args.num_samples
        )
    
    # TODO: Add other analyses as they're implemented
    # if 'trajectory' in args.analyses:
    #     results['trajectory'] = run_temporal_trajectory_analysis(...)
    
    # if 'heads' in args.analyses:
    #     results['heads'] = run_head_specialization_analysis(...)
    
    # if 'uncertainty' in args.analyses:
    #     results['uncertainty'] = run_attention_uncertainty_analysis(...)
    
    # if 'history' in args.analyses:
    #     results['history'] = run_history_window_ablation(...)
    
    # if 'embeddings' in args.analyses:
    #     results['embeddings'] = run_embedding_space_analysis(...)
    
    # if 'gradients' in args.analyses:
    #     results['gradients'] = run_gradient_flow_analysis(...)
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ Advanced Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_manager.output_dir}")
    print("\nDirectory structure:")
    print(f"  - {output_manager.figures_dir}/")
    print(f"  - {output_manager.tables_dir}/")
    print(f"  - {output_manager.metrics_dir}/")
    print(f"  - {output_manager.reports_dir}/")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
