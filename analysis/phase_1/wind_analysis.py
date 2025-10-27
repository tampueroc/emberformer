"""
Wind-Fire Alignment Analysis

Tests whether the model learned wind-driven spread correctly by measuring
directional alignment of predictions with wind direction.
"""

import torch
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def analyze_wind_direction(model, dataset, device, output_dir, num_samples=100):
    """
    Analyze wind-fire alignment in model predictions
    
    Tests: Does predicted spread align with wind direction?
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nAnalyzing wind-fire alignment for {num_samples} samples...")
    print(f"Output directory: {output_dir}")
    
    results = []
    
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Analyzing wind alignment"):
        try:
            fire_hist, static, wind, target = dataset[i]
            
            # Prepare inputs
            # fire_hist is [1, H, W, T] from dataset, need [B, T, 1, H, W] for model
            fire_hist = fire_hist.permute(3, 0, 1, 2)  # [T, 1, H, W]
            fire_hist_batch = fire_hist.unsqueeze(0).to(device)  # [B, T, 1, H, W]
            static_batch = static.unsqueeze(0).to(device)
            wind_batch = wind.unsqueeze(0).to(device)
            
            B, T = fire_hist_batch.shape[:2]
            valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
            
            # Get prediction
            with torch.no_grad():
                pred = model(fire_hist_batch, static_batch, wind_batch, valid_t)
                pred = torch.sigmoid(pred)[0, 0].cpu()  # [H, W]
            
            # Wind at last timestep
            wind_speed = wind[-1, 0].item()
            wind_dir = wind[-1, 1].item()  # Radians
            
            # Analyze spread direction
            fire_current = fire_hist[0, :, :, -1] > 0.5
            fire_pred = pred > 0.5
            new_fire = fire_pred & ~fire_current
            
            if new_fire.sum() > 10 and fire_current.sum() > 0:
                # Compute fire center
                y_fire, x_fire = torch.where(fire_current)
                cy, cx = y_fire.float().mean(), x_fire.float().mean()
                
                # Vector from center to new fire
                y_new, x_new = torch.where(new_fire)
                dy = y_new.float() - cy
                dx = x_new.float() - cx
                
                # Angle of each spread pixel
                spread_angles = torch.atan2(dy, dx)
                
                # Alignment with wind (lower = better)
                angle_diff = torch.abs((spread_angles - wind_dir + np.pi) % (2*np.pi) - np.pi)
                
                results.append({
                    'wind_speed': wind_speed,
                    'wind_dir': wind_dir,
                    'mean_alignment': angle_diff.mean().item(),
                    'spread_area': new_fire.sum().item(),
                })
        
        except Exception as e:
            continue
    
    if not results:
        print("⚠️  No valid results - all samples skipped")
        return
    
    df = pd.DataFrame(results)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Wind speed vs alignment
    axes[0].scatter(df['wind_speed'], df['mean_alignment'], alpha=0.5, s=30)
    axes[0].set_xlabel('Wind Speed', fontsize=12)
    axes[0].set_ylabel('Spread-Wind Alignment (rad)', fontsize=12)
    axes[0].set_title('Wind Speed vs Directional Alignment', fontsize=13, fontweight='bold')
    axes[0].axhline(np.pi/2, color='red', linestyle='--', linewidth=1.5, label='Perpendicular (π/2)')
    axes[0].axhline(np.pi/4, color='orange', linestyle='--', linewidth=1.5, label='Well-aligned (π/4)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # 2. Alignment distribution
    axes[1].hist(df['mean_alignment'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(df['mean_alignment'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {df['mean_alignment'].mean():.3f} rad")
    axes[1].axvline(np.pi/4, color='orange', linestyle='--', linewidth=1.5, label='π/4 (45°)')
    axes[1].axvline(np.pi/2, color='darkred', linestyle='--', linewidth=1.5, label='π/2 (90°)')
    axes[1].set_xlabel('Alignment (radians)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of Spread-Wind Alignment', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    # 3. Polar histogram
    ax_polar = plt.subplot(133, projection='polar')
    wind_bins = np.linspace(-np.pi, np.pi, 9)
    alignment_by_dir = []
    
    for i in range(len(wind_bins)-1):
        mask = (df['wind_dir'] >= wind_bins[i]) & (df['wind_dir'] < wind_bins[i+1])
        if mask.sum() > 0:
            alignment_by_dir.append(df[mask]['mean_alignment'].mean())
        else:
            alignment_by_dir.append(0)
    
    theta = (wind_bins[:-1] + wind_bins[1:]) / 2
    bars = ax_polar.bar(theta, alignment_by_dir, width=2*np.pi/8, alpha=0.7, 
                       edgecolor='black', color='steelblue')
    ax_polar.set_title('Mean Alignment by Wind Direction\n(Lower = Better)', 
                      fontsize=12, fontweight='bold', pad=20)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wind_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print("\n" + "="*60)
    print("Wind-Fire Alignment Analysis")
    print("="*60)
    print(f"\nSamples analyzed: {len(df)}")
    print(f"\nAlignment Statistics:")
    print(f"  Mean alignment: {df['mean_alignment'].mean():.3f} rad ({np.degrees(df['mean_alignment'].mean()):.1f}°)")
    print(f"  Median alignment: {df['mean_alignment'].median():.3f} rad ({np.degrees(df['mean_alignment'].median()):.1f}°)")
    print(f"  Std dev: {df['mean_alignment'].std():.3f} rad")
    
    print(f"\nAlignment Quality:")
    well_aligned = (df['mean_alignment'] < np.pi/4).mean() * 100
    moderately_aligned = ((df['mean_alignment'] >= np.pi/4) & (df['mean_alignment'] < np.pi/2)).mean() * 100
    perpendicular = (df['mean_alignment'] >= np.pi/2).mean() * 100
    
    print(f"  Well-aligned (< 45°):     {well_aligned:.1f}%  {'✓' if well_aligned > 50 else '⚠️'}")
    print(f"  Moderate (45° - 90°):     {moderately_aligned:.1f}%")
    print(f"  Perpendicular (> 90°):    {perpendicular:.1f}%  {'⚠️' if perpendicular > 30 else '✓'}")
    
    print(f"\nWind Speed Correlation:")
    corr = df['wind_speed'].corr(df['mean_alignment'])
    print(f"  Correlation: {corr:.3f}")
    if abs(corr) < 0.1:
        print(f"  → Weak correlation (model uses wind but not speed-dependent)")
    elif corr < -0.1:
        print(f"  → Negative correlation (higher wind → better alignment)")
    else:
        print(f"  → Positive correlation (higher wind → worse alignment)")
    
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # Save data
    df.to_csv(output_dir / 'wind_alignment_data.csv', index=False)
    print(f"Raw data saved to: wind_alignment_data.csv")
