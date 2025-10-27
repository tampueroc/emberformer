"""
Temporal Importance Analysis

Measures which timesteps in the fire history are most important for predictions.
"""

import torch
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sns  # type: ignore


def compute_temporal_importance(model, sample, device):
    """
    Measure importance of each timestep using gradient magnitude
    
    Args:
        model: EmberFormerDINO model
        sample: (fire_hist, static, wind, target) tuple
        device: torch device
    
    Returns:
        temporal_importance: [T] array of importance values
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
    logits = model(fire_hist, static, wind, valid_t)
    
    # Backward
    loss = logits.sum()
    loss.backward()
    
    # Average gradient magnitude per timestep
    temporal_importance = fire_hist.grad.abs().mean(dim=(0, 2, 3, 4))  # [T]
    
    return temporal_importance.cpu().numpy()


def analyze_temporal_importance(model, dataset, device, output_dir, num_samples=100):
    """
    Analyze temporal importance patterns across multiple samples
    
    Question: Do recent frames dominate, or is full history used?
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nAnalyzing temporal importance for {num_samples} samples...")
    print(f"Output directory: {output_dir}")
    
    # Group results by sequence length
    results_by_length = {2: [], 3: [], 4: []}
    
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Computing temporal importance"):
        try:
            fire_hist, static, wind, target = dataset[i]
            T = fire_hist.shape[-1]
            
            if T not in results_by_length:
                continue
            
            importance = compute_temporal_importance(
                model, (fire_hist, static, wind, target), device
            )
            
            results_by_length[T].append(importance)
        
        except Exception as e:
            print(f"  ⚠️  Error processing sample {i}: {e}")
            continue
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (T, importance_list) in zip(axes, results_by_length.items()):
        if not importance_list:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'T={T} (n=0)')
            continue
        
        # Average importance across samples
        avg_importance = np.mean(importance_list, axis=0)
        std_importance = np.std(importance_list, axis=0)
        
        timesteps = [f't-{T-i-1}' for i in range(T)]
        
        bars = ax.bar(timesteps, avg_importance, yerr=std_importance, 
                     alpha=0.7, capsize=5, color='steelblue')
        ax.set_ylabel('Importance (Gradient Magnitude)', fontsize=11)
        ax.set_title(f'T={T} (n={len(importance_list)})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight most recent
        ax.axhline(avg_importance.mean(), color='r', 
                  linestyle='--', alpha=0.5, linewidth=1.5, label='Mean')
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, avg_importance)):
            percentage = (val / avg_importance.sum()) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_importance[i],
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
    
    plt.suptitle('Temporal Importance by Sequence Length', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\n✓ Temporal importance analysis complete!")
    print("\n" + "="*60)
    
    for T, importance_list in results_by_length.items():
        if importance_list:
            avg = np.mean(importance_list, axis=0)
            print(f"\nT={T} (n={len(importance_list)} samples):")
            for t in range(T):
                percentage = (avg[t] / avg.sum()) * 100
                print(f"  t-{T-t-1}: {avg[t]:.4f} ({percentage:.1f}%)")
            
            # Check recency bias
            recent_ratio = avg[-1] / avg.mean()
            print(f"  Recency bias: {recent_ratio:.2f}x (t-0 vs mean)")
            
            if recent_ratio > 2.0:
                print(f"  → Strong recency bias: Recent frame dominates")
            elif recent_ratio > 1.5:
                print(f"  → Moderate recency bias")
            else:
                print(f"  → Balanced temporal attention")
    
    print("\n" + "="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # Save raw data
    import pickle
    with open(output_dir / 'temporal_importance_data.pkl', 'wb') as f:
        pickle.dump(results_by_length, f)
    print(f"Raw data saved to: temporal_importance_data.pkl")
