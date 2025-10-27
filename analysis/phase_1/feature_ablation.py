"""
Static Feature Ablation Analysis

Tests the importance of each terrain feature by removing it and measuring
the impact on model performance.
"""

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchmetrics
from torch.utils.data import DataLoader, Subset


def collate_raw_dino(batch):
    """Collate function for DINO training (copied from train_dino.py)"""
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


def evaluate_model_f1(model, loader, device):
    """Compute F1 score on validation set"""
    f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            fire_hist, static, wind, targets, valid_t = batch
            fire_hist = fire_hist.to(device)
            static = static.to(device)
            wind = wind.to(device)
            targets = targets.to(device)
            valid_t = valid_t.to(device)
            
            logits = model(fire_hist, static, wind, valid_t)
            preds = (torch.sigmoid(logits) > 0.5).int()
            
            mask = torch.ones_like(targets, dtype=torch.bool)
            f1_metric.update(preds[mask].flatten(), targets[mask].flatten().int())
    
    return f1_metric.compute().item()


def analyze_feature_importance(model, dataset, device, output_dir, num_samples=1000):
    """
    Measure impact of each static terrain channel via ablation
    
    Args:
        model: trained model
        dataset: validation dataset  
        device: torch device
        output_dir: where to save results
        num_samples: number of samples to use (subset for speed)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nRunning feature ablation analysis...")
    print(f"Using {num_samples} samples for speed")
    print(f"Output directory: {output_dir}")
    
    # Use subset
    subset_indices = list(range(min(num_samples, len(dataset))))
    val_subset = Subset(dataset, subset_indices)
    
    # Static channel names (landscape features)
    static_channels = [
        'fuels',
        'arqueo',
        'canopy_bulk_density',
        'canopy_base_height',
        'elevation',
        'flora',
        'paleo',
        'urban'
    ]
    
    # Get actual number of channels from first sample
    first_sample = dataset[0]
    num_channels = first_sample[1].shape[0]
    static_channels = static_channels[:num_channels]
    
    print(f"Found {num_channels} static channels")
    
    # Baseline performance
    print("\n1. Computing baseline F1...")
    baseline_loader = DataLoader(val_subset, batch_size=16, collate_fn=collate_raw_dino)
    baseline_f1 = evaluate_model_f1(model, baseline_loader, device)
    print(f"   Baseline F1: {baseline_f1:.4f}")
    
    importance = {}
    
    # Test each channel
    for i, channel_name in enumerate(static_channels):
        print(f"\n2. Ablating channel {i}: {channel_name}...")
        
        # Create wrapper dataset that zeros channel i
        class AblatedDataset(Dataset):
            def __init__(self, dataset, channel_idx):
                self.dataset = dataset
                self.channel_idx = channel_idx
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                fire_hist, static, wind, target = self.dataset[idx]
                # Zero out channel
                static = static.clone()
                static[self.channel_idx] = 0.0
                return fire_hist, static, wind, target
        
        ablated_dataset = AblatedDataset(val_subset, i)
        ablated_loader = DataLoader(ablated_dataset, batch_size=16,
                                    collate_fn=collate_raw_dino)
        
        # Evaluate
        f1_without = evaluate_model_f1(model, ablated_loader, device)
        drop = baseline_f1 - f1_without
        importance[channel_name] = drop
        
        print(f"   F1 without: {f1_without:.4f} (drop: {drop:+.4f})")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = list(importance.keys())
    drops = list(importance.values())
    colors = ['red' if d > 0 else 'green' for d in drops]
    
    bars = ax.barh(channels, drops, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('F1 Score Drop When Feature Removed', fontsize=12)
    ax.set_title('Static Feature Importance (Ablation Study)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, drops):
        label_x = val + (0.001 if val >= 0 else -0.001)
        ha = 'left' if val >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:+.4f}',
               ha=ha, va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print ranking
    print("\n" + "="*60)
    print("Feature Importance Ranking")
    print("="*60)
    
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (feature, drop) in enumerate(sorted_features, 1):
        impact = "🔴 Critical" if drop > 0.02 else ("🟡 Important" if drop > 0.01 else "🟢 Minor")
        print(f"{rank}. {feature:20s}: {drop:+.4f}  {impact}")
    
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # Save raw data
    import json
    with open(output_dir / 'feature_importance.json', 'w') as f:
        json.dump({
            'baseline_f1': baseline_f1,
            'importance': importance,
            'sorted_ranking': [(f, v) for f, v in sorted_features]
        }, f, indent=2)
    print(f"Raw data saved to: feature_importance.json")
