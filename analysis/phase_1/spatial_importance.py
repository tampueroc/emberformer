"""
Spatial Importance Analysis

Computes gradient-based spatial importance maps showing which input pixels
influence predictions most strongly.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_spatial_importance_map(model, sample, device):
    """
    Compute spatial importance using gradient-based attribution
    
    Args:
        model: EmberFormerDINO model
        sample: (fire_hist, static, wind, target) tuple
        device: torch device
    
    Returns:
        importance: [T, H, W] normalized importance maps
    """
    fire_hist, static, wind, target = sample
    
    # fire_hist is [1, H, W, T] from dataset, need [B, T, 1, H, W] for model
    fire_hist = fire_hist.permute(3, 0, 1, 2)  # [T, 1, H, W]
    fire_hist = fire_hist.unsqueeze(0).to(device).requires_grad_(True)  # [B, T, 1, H, W]
    static = static.unsqueeze(0).to(device)
    wind = wind.unsqueeze(0).to(device)
    
    B, T, C, H, W = fire_hist.shape
    valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
    
    model.eval()
    
    # Forward pass
    logits = model(fire_hist, static, wind, valid_t)
    
    # Backward on output sum
    loss = logits.sum()
    loss.backward()
    
    # Gradient magnitude as importance
    importance = fire_hist.grad[0].abs()  # [T, 1, H, W]
    
    # Normalize per timestep
    importance_norm = []
    for t in range(T):
        imp_t = importance[t, 0]
        if imp_t.max() > 0:
            imp_t = (imp_t - imp_t.min()) / (imp_t.max() - imp_t.min() + 1e-8)
        importance_norm.append(imp_t)
    
    importance_norm = torch.stack(importance_norm)
    
    return importance_norm.cpu()


def visualize_spatial_importance_single(fire_hist, importance, sample_idx, output_path):
    """
    Visualize spatial importance for a single sample
    
    Args:
        fire_hist: [1, H, W, T] fire history
        importance: [T, H, W] importance maps
        sample_idx: sample index
        output_path: where to save figure
    """
    T = fire_hist.shape[-1]
    
    fig, axes = plt.subplots(2, T, figsize=(4*T, 8))
    
    # Handle single timestep case
    if T == 1:
        axes = axes.reshape(2, 1)
    
    for t in range(T):
        # Top row: Original fire frame
        axes[0, t].imshow(fire_hist[0, :, :, t].numpy(), cmap='hot', interpolation='nearest')
        axes[0, t].set_title(f'Fire t-{T-t-1}', fontsize=12)
        axes[0, t].axis('off')
        
        # Bottom row: Importance map
        im = axes[1, t].imshow(importance[t].numpy(), cmap='viridis', interpolation='bilinear')
        axes[1, t].set_title(f'Importance t-{T-t-1}', fontsize=12)
        axes[1, t].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Sample {sample_idx}: Spatial Importance Over Time', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_spatial_importance(model, dataset, device, output_dir, num_samples=10):
    """
    Main function to analyze spatial importance across multiple samples
    
    Args:
        model: trained model
        dataset: validation dataset
        device: torch device
        output_dir: where to save results
        num_samples: number of samples to analyze
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nAnalyzing spatial importance for {num_samples} samples...")
    print(f"Output directory: {output_dir}")
    
    # Statistics
    importance_stats = {
        'max_importance': [],
        'mean_importance': [],
        'concentration': [],  # Std dev / mean
    }
    
    for sample_idx in tqdm(range(min(num_samples, len(dataset))), desc="Computing importance"):
        try:
            fire_hist, static, wind, target = dataset[sample_idx]
            
            # Get importance maps
            importance = compute_spatial_importance_map(
                model, (fire_hist, static, wind, target), device
            )
            
            # Visualize
            output_path = output_dir / f'spatial_importance_sample_{sample_idx:03d}.png'
            visualize_spatial_importance_single(fire_hist, importance, sample_idx, output_path)
            
            # Collect statistics
            T = importance.shape[0]
            for t in range(T):
                imp = importance[t]
                importance_stats['max_importance'].append(imp.max().item())
                importance_stats['mean_importance'].append(imp.mean().item())
                if imp.mean() > 0:
                    importance_stats['concentration'].append((imp.std() / imp.mean()).item())
        
        except Exception as e:
            print(f"  ⚠️  Error processing sample {sample_idx}: {e}")
            continue
    
    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Max importance distribution
    axes[0].hist(importance_stats['max_importance'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Max Importance')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Maximum Importance')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Mean importance distribution
    axes[1].hist(importance_stats['mean_importance'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Mean Importance')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Mean Importance')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Concentration (spatial focus)
    if importance_stats['concentration']:
        axes[2].hist(importance_stats['concentration'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[2].set_xlabel('Concentration (Std/Mean)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Spatial Focus Distribution')
        axes[2].axvline(np.mean(importance_stats['concentration']), 
                       color='red', linestyle='--', label=f"Mean: {np.mean(importance_stats['concentration']):.2f}")
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_importance_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n✓ Spatial importance analysis complete!")
    print(f"  Analyzed: {len(importance_stats['max_importance'])} timesteps across {num_samples} samples")
    print(f"  Max importance: {np.mean(importance_stats['max_importance']):.4f} ± {np.std(importance_stats['max_importance']):.4f}")
    print(f"  Mean importance: {np.mean(importance_stats['mean_importance']):.4f} ± {np.std(importance_stats['mean_importance']):.4f}")
    if importance_stats['concentration']:
        print(f"  Concentration: {np.mean(importance_stats['concentration']):.4f} ± {np.std(importance_stats['concentration']):.4f}")
        print(f"    (Higher = more focused on specific regions)")
    print(f"\nResults saved to: {output_dir}")
