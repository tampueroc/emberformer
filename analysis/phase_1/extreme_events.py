"""
Extreme Event Analysis

Identifies and analyzes extreme fire spread events to understand
model performance on challenging scenarios.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm


def identify_extreme_events(dataset, percentile=95):
    """
    Find samples with extreme fire spread
    
    Args:
        dataset: validation dataset
        percentile: percentile threshold for "extreme" (default 95th)
    
    Returns:
        extreme_indices: list of sample indices
        spread_areas: list of (index, spread_area) tuples
    """
    print(f"\nScanning dataset for extreme events (>{percentile}th percentile)...")
    
    spread_areas = []
    
    for i in tqdm(range(len(dataset)), desc="Computing spread areas"):
        try:
            fire_hist, static, wind, target = dataset[i]
            T = fire_hist.shape[-1]
            
            fire_t = fire_hist[0, :, :, T-1].sum().item()
            fire_t1 = target.sum().item()
            spread = fire_t1 - fire_t
            
            spread_areas.append((i, spread))
        except:
            continue
    
    spread_areas.sort(key=lambda x: x[1], reverse=True)
    
    threshold_idx = int(len(spread_areas) * (1 - percentile/100))
    extreme_indices = [idx for idx, _ in spread_areas[:threshold_idx]]
    
    print(f"  ✓ Identified {len(extreme_indices)} extreme events")
    print(f"  Spread range: {spread_areas[0][1]:.0f} to {spread_areas[threshold_idx][1]:.0f} pixels")
    
    return extreme_indices, spread_areas


def visualize_extreme_event(model, sample, sample_idx, output_path, device):
    """
    Create detailed visualization of an extreme event
    
    Shows:
        - Fire history progression
        - Terrain features
        - Model prediction vs ground truth
        - Error analysis
    """
    fire_hist, static, wind, target = sample
    T = fire_hist.shape[-1]
    
    # Get prediction
    # fire_hist is [1, H, W, T] from dataset, need [B, T, 1, H, W] for model
    fire_hist_permuted = fire_hist.permute(3, 0, 1, 2)  # [T, 1, H, W]
    fire_hist_batch = fire_hist_permuted.unsqueeze(0).to(device)  # [B, T, 1, H, W]
    static_batch = static.unsqueeze(0).to(device)
    wind_batch = wind.unsqueeze(0).to(device)
    valid_t = torch.ones(1, T, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        logits = model(fire_hist_batch, static_batch, wind_batch, valid_t)
        pred = torch.sigmoid(logits)[0, 0].cpu()
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Row 1: Fire history
    for i in range(min(T, 4)):
        ax = fig.add_subplot(gs[0, i])
        if i < T:
            fire_frame = fire_hist[0, :, :, i].numpy()
            ax.imshow(fire_frame, cmap='hot', interpolation='nearest')
            ax.set_title(f'Fire t-{T-i-1}', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, color='gray')
            ax.set_title(f'Fire t-{T-i-1}', fontsize=11)
        ax.axis('off')
    
    # Row 2: Terrain features
    terrain_names = ['Elevation', 'Slope', 'Aspect', 'Fuel Load']
    cmaps = ['terrain', 'YlOrRd', 'twilight', 'YlGn']
    for i in range(min(4, static.shape[0])):
        ax = fig.add_subplot(gs[1, i])
        cmap = cmaps[i] if i < len(cmaps) else 'viridis'
        im = ax.imshow(static[i].numpy(), cmap=cmap, interpolation='bilinear')
        name = terrain_names[i] if i < len(terrain_names) else f'Static {i}'
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 3: Predictions and errors
    ax_last = fig.add_subplot(gs[2, 0])
    ax_last.imshow(fire_hist[0, :, :, -1].numpy(), cmap='hot', interpolation='nearest')
    ax_last.set_title('Fire at t', fontsize=11, fontweight='bold')
    ax_last.axis('off')
    
    # Add wind arrow
    wind_t = wind[-1].numpy()
    arrow_scale = 50
    ax_last.arrow(30, 30, wind_t[0]*arrow_scale, wind_t[1]*arrow_scale, 
                 color='cyan', width=3, head_width=10, head_length=8,
                 length_includes_head=True, linewidth=2)
    ax_last.text(30, 50, f'Wind', color='cyan', fontsize=9, fontweight='bold')
    
    ax_pred = fig.add_subplot(gs[2, 1])
    im_pred = ax_pred.imshow(pred.numpy(), cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
    ax_pred.set_title('Predicted t+1', fontsize=11, fontweight='bold')
    ax_pred.axis('off')
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    
    ax_true = fig.add_subplot(gs[2, 2])
    ax_true.imshow(target[0].numpy(), cmap='hot', interpolation='nearest')
    ax_true.set_title('Actual t+1', fontsize=11, fontweight='bold')
    ax_true.axis('off')
    
    ax_error = fig.add_subplot(gs[2, 3])
    pred_bin = (pred > 0.5).float()
    target_bin = target[0]
    error_map = pred_bin - target_bin  # +1: FP, -1: FN, 0: correct
    
    im_error = ax_error.imshow(error_map.numpy(), cmap='RdYlGn_r', vmin=-1, vmax=1, interpolation='nearest')
    ax_error.set_title('Error Map', fontsize=11, fontweight='bold')
    ax_error.axis('off')
    
    # Add error legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', label='False Negative (missed fire)'),
        Patch(facecolor='lightgreen', label='Correct'),
        Patch(facecolor='darkgreen', label='False Positive (false alarm)')
    ]
    ax_error.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Compute metrics
    tp = ((pred_bin == 1) & (target_bin == 1)).sum().item()
    fp = ((pred_bin == 1) & (target_bin == 0)).sum().item()
    fn = ((pred_bin == 0) & (target_bin == 1)).sum().item()
    tn = ((pred_bin == 0) & (target_bin == 0)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Title with metrics
    spread_actual = target.sum().item() - fire_hist[0, :, :, -1].sum().item()
    spread_pred = pred_bin.sum().item() - fire_hist[0, :, :, -1].sum().item()
    
    title = f'Extreme Event #{sample_idx} | T={T} | Wind=({wind_t[0]:.2f}, {wind_t[1]:.2f})\n'
    title += f'Spread: Actual={spread_actual:.0f}, Predicted={spread_pred:.0f} pixels | '
    title += f'F1={f1:.3f}, IoU={iou:.3f}, Precision={precision:.3f}, Recall={recall:.3f}'
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_extreme_events(model, dataset, device, output_dir, percentile=95, num_visualize=10):
    """
    Main function to analyze extreme events
    
    Args:
        model: trained model
        dataset: validation dataset
        device: torch device
        output_dir: where to save results
        percentile: percentile threshold for extreme events
        num_visualize: number of extreme events to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nAnalyzing extreme events (top {100-percentile}%)...")
    print(f"Output directory: {output_dir}")
    
    # Identify extreme events
    extreme_indices, spread_areas = identify_extreme_events(dataset, percentile)
    
    # Visualize top N extreme events
    print(f"\nVisualizing top {num_visualize} extreme events...")
    
    for i, idx in enumerate(tqdm(extreme_indices[:num_visualize], desc="Creating visualizations")):
        try:
            sample = dataset[idx]
            output_path = output_dir / f'extreme_event_{i:02d}_idx{idx:05d}.png'
            visualize_extreme_event(model, sample, idx, output_path, device)
        except Exception as e:
            print(f"  ⚠️  Error visualizing sample {idx}: {e}")
            continue
    
    # Plot spread distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # All spreads
    all_spreads = [s for _, s in spread_areas]
    axes[0].hist(all_spreads, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Spread Area (pixels)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Fire Spread Areas', fontsize=13, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Mark percentile threshold
    threshold_spread = spread_areas[int(len(spread_areas) * (1 - percentile/100))][1]
    axes[0].axvline(threshold_spread, color='red', linestyle='--', linewidth=2,
                   label=f'{percentile}th percentile: {threshold_spread:.0f}')
    axes[0].legend(fontsize=10)
    
    # Extreme spreads only
    extreme_spreads = [s for _, s in spread_areas[:len(extreme_indices)]]
    axes[1].hist(extreme_spreads, bins=30, edgecolor='black', alpha=0.7, color='darkred')
    axes[1].set_xlabel('Spread Area (pixels)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'Extreme Events Only (Top {100-percentile}%)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spread_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print("\n" + "="*60)
    print("Extreme Event Statistics")
    print("="*60)
    print(f"\nTotal samples: {len(spread_areas)}")
    print(f"Extreme events (>{percentile}th percentile): {len(extreme_indices)}")
    print(f"\nSpread Statistics:")
    print(f"  Mean: {np.mean(all_spreads):.1f} pixels")
    print(f"  Median: {np.median(all_spreads):.1f} pixels")
    print(f"  {percentile}th percentile: {threshold_spread:.1f} pixels")
    print(f"  Max: {spread_areas[0][1]:.1f} pixels")
    print(f"\nExtreme Event Spread:")
    print(f"  Mean: {np.mean(extreme_spreads):.1f} pixels")
    print(f"  Min: {min(extreme_spreads):.1f} pixels")
    print(f"  Max: {max(extreme_spreads):.1f} pixels")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # Save data
    import json
    with open(output_dir / 'extreme_events_summary.json', 'w') as f:
        json.dump({
            'total_samples': len(spread_areas),
            'num_extreme': len(extreme_indices),
            'percentile': percentile,
            'threshold': threshold_spread,
            'extreme_indices': extreme_indices[:100],  # Save top 100
            'statistics': {
                'all_mean': float(np.mean(all_spreads)),
                'all_median': float(np.median(all_spreads)),
                'extreme_mean': float(np.mean(extreme_spreads)),
                'max_spread': float(spread_areas[0][1]),
            }
        }, f, indent=2)
    print(f"Summary saved to: extreme_events_summary.json")
