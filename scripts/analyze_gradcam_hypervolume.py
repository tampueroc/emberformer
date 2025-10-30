"""
Grad-CAM Environmental Hypervolume Analysis

Following Pais et al. (2020) methodology:
1. Extract important pixels from Grad-CAM (top 1% importance)
2. Extract environmental variables at those pixels
3. Build n-dimensional hypervolume of extreme fire conditions
4. Compare extreme vs normal fire environmental envelopes

Usage:
    python scripts/analyze_gradcam_hypervolume.py --checkpoint checkpoints/dino_phase2_best.pt --num_samples 100
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml
import sys
import json
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import RawFireDataset
from models.emberformer import EmberFormerDINO
from scripts.analyze_gradcam import GradCAM, load_model
import torchvision.transforms.functional as TF


class GradCAMHypervolume:
    """
    Extract environmental hypervolume from Grad-CAM important pixels
    """
    def __init__(self, model, device='cuda', importance_threshold=99):
        """
        Args:
            model: Trained EmberFormerDINO
            device: cuda or cpu
            importance_threshold: Percentile threshold for important pixels (99 = top 1%)
        """
        self.model = model
        self.device = device
        self.importance_threshold = importance_threshold
        self.gradcam = GradCAM(model, model.refinement_decoder.output_conv)
        
        # Static channel names (from dataset)
        self.static_names = [
            'elevation', 'slope', 'aspect', 'fuel_load', 
            'vegetation', 'canopy_height', 'canopy_density', 'other'
        ]
        
        # Storage for extracted data
        self.all_data = []
        
    def extract_important_pixels(self, sample, fire_intensity=None):
        """
        Extract environmental conditions at Grad-CAM important pixels
        
        Args:
            sample: (fire_hist, static, wind, target) tuple
            fire_intensity: Scalar metric for fire severity (optional)
        
        Returns:
            dict with environmental data at important pixels
        """
        fire_hist, static, wind, target = sample
        
        # Prepare for model
        fire_hist = fire_hist.permute(3, 0, 1, 2)  # [T, 1, H, W]
        fire_hist_batch = fire_hist.unsqueeze(0).to(self.device)
        static_batch = static.unsqueeze(0).to(self.device)
        wind_batch = wind.unsqueeze(0).to(self.device)
        
        B, T, C, H, W = fire_hist_batch.shape
        valid_t = torch.ones(B, T, dtype=torch.bool, device=self.device)
        
        # Compute Grad-CAM
        with torch.enable_grad():
            cam = self.gradcam(fire_hist_batch, static_batch, wind_batch, valid_t)
        
        cam_np = cam[0].cpu().numpy()  # [H, W]
        
        # Get important pixels (top 1%)
        threshold = np.percentile(cam_np, self.importance_threshold)
        important_mask = cam_np > threshold
        
        # Get coordinates of important pixels
        y_coords, x_coords = np.where(important_mask)
        
        if len(y_coords) == 0:
            return None
        
        # Extract environmental data at important pixels
        static_np = static.cpu().numpy()  # [C, H, W]
        wind_np = wind.cpu().numpy()  # [T, 2]
        
        pixel_data = []
        for y, x in zip(y_coords, x_coords):
            # Static features at this pixel
            env_dict = {
                'y': int(y),
                'x': int(x),
                'importance': float(cam_np[y, x]),
            }
            
            # Add static features
            for i, name in enumerate(self.static_names[:static_np.shape[0]]):
                env_dict[name] = float(static_np[i, y, x])
            
            # Add wind (use most recent timestep)
            env_dict['wind_speed'] = float(wind_np[-1, 0])
            env_dict['wind_direction'] = float(wind_np[-1, 1])
            
            # Add fire intensity metric if provided
            if fire_intensity is not None:
                env_dict['fire_intensity'] = float(fire_intensity)
            
            pixel_data.append(env_dict)
        
        return {
            'pixels': pixel_data,
            'num_important': len(y_coords),
            'importance_ratio': len(y_coords) / (H * W),
            'threshold': float(threshold),
        }
    
    def collect_samples(self, dataset, num_samples=100, compute_intensity=True):
        """
        Collect environmental data from multiple samples
        
        Args:
            dataset: RawFireDataset
            num_samples: Number of samples to analyze
            compute_intensity: Whether to compute fire intensity metric
        """
        print(f"Collecting environmental data from {num_samples} samples...")
        
        for sample_idx in range(min(num_samples, len(dataset))):
            fire_hist, static, wind, target = dataset[sample_idx]
            
            # Compute fire intensity (area burned as proxy)
            if compute_intensity:
                fire_intensity = target.sum().item() / target.numel()
            else:
                fire_intensity = None
            
            # Extract important pixels
            result = self.extract_important_pixels(
                (fire_hist, static, wind, target),
                fire_intensity=fire_intensity
            )
            
            if result is not None:
                self.all_data.extend(result['pixels'])
            
            if (sample_idx + 1) % 10 == 0:
                print(f"  Processed {sample_idx + 1}/{num_samples} samples, {len(self.all_data)} pixels collected")
        
        print(f"\n✓ Collected {len(self.all_data)} important pixels from {num_samples} samples")
    
    def build_hypervolume(self, extreme_threshold=99, feature_subset=None):
        """
        Build convex hull hypervolume of environmental conditions
        
        Args:
            extreme_threshold: Percentile for extreme fires (99 = top 1% intensity)
            feature_subset: List of features to use (default: all static + wind)
        
        Returns:
            dict with hypervolume statistics
        """
        if len(self.all_data) == 0:
            raise ValueError("No data collected. Run collect_samples() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_data)
        
        # Default features: static + wind
        if feature_subset is None:
            feature_subset = self.static_names[:7] + ['wind_speed', 'wind_direction']
        
        # Filter for features that exist
        feature_subset = [f for f in feature_subset if f in df.columns]
        
        print(f"\nBuilding hypervolume with features: {feature_subset}")
        
        # Extract feature matrix
        X = df[feature_subset].values  # [N, n_features]
        
        # Separate extreme vs normal fires
        if 'fire_intensity' in df.columns:
            extreme_threshold_val = np.percentile(df['fire_intensity'], extreme_threshold)
            extreme_mask = df['fire_intensity'] > extreme_threshold_val
            
            X_extreme = X[extreme_mask]
            X_normal = X[~extreme_mask]
            
            print(f"  Extreme fires (>{extreme_threshold}th percentile): {X_extreme.shape[0]} pixels")
            print(f"  Normal fires: {X_normal.shape[0]} pixels")
        else:
            X_extreme = X
            X_normal = None
            print(f"  Total pixels: {X.shape[0]}")
        
        # Compute convex hull for extreme fires
        try:
            hull_extreme = ConvexHull(X_extreme)
            volume_extreme = hull_extreme.volume
            print(f"  Extreme fire hypervolume: {volume_extreme:.4e}")
        except Exception as e:
            print(f"  ⚠️  Could not compute convex hull: {e}")
            volume_extreme = None
            hull_extreme = None
        
        # Compute convex hull for normal fires
        volume_normal = None
        hull_normal = None
        if X_normal is not None and len(X_normal) > len(feature_subset):
            try:
                hull_normal = ConvexHull(X_normal)
                volume_normal = hull_normal.volume
                print(f"  Normal fire hypervolume: {volume_normal:.4e}")
                
                if volume_extreme is not None and volume_normal is not None:
                    ratio = volume_extreme / volume_normal
                    print(f"  Volume ratio (extreme/normal): {ratio:.3f}")
            except Exception as e:
                print(f"  ⚠️  Could not compute normal fire hull: {e}")
        
        return {
            'features': feature_subset,
            'n_extreme': X_extreme.shape[0] if X_extreme is not None else 0,
            'n_normal': X_normal.shape[0] if X_normal is not None else 0,
            'volume_extreme': float(volume_extreme) if volume_extreme is not None else None,
            'volume_normal': float(volume_normal) if volume_normal is not None else None,
            'hull_extreme': hull_extreme,
            'hull_normal': hull_normal,
            'X_extreme': X_extreme,
            'X_normal': X_normal,
        }
    
    def visualize_hypervolume_2d(self, feature_x, feature_y, output_dir='results/hypervolume'):
        """
        Visualize 2D projection of hypervolume
        
        Args:
            feature_x: First feature name (e.g., 'slope')
            feature_y: Second feature name (e.g., 'wind_speed')
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.all_data)
        
        # Check features exist
        if feature_x not in df.columns or feature_y not in df.columns:
            print(f"⚠️  Features {feature_x} or {feature_y} not found in data")
            return
        
        # Separate extreme vs normal
        if 'fire_intensity' in df.columns:
            threshold = np.percentile(df['fire_intensity'], 99)
            extreme_mask = df['fire_intensity'] > threshold
            
            df_extreme = df[extreme_mask]
            df_normal = df[~extreme_mask]
        else:
            df_extreme = df
            df_normal = None
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if df_normal is not None:
            ax.scatter(df_normal[feature_x], df_normal[feature_y], 
                      c='lightblue', alpha=0.3, s=10, label='Normal fires')
        
        ax.scatter(df_extreme[feature_x], df_extreme[feature_y],
                  c='red', alpha=0.6, s=20, label='Extreme fires (top 1%)')
        
        ax.set_xlabel(feature_x, fontsize=12)
        ax.set_ylabel(feature_y, fontsize=12)
        ax.set_title(f'Environmental Hypervolume: {feature_x} vs {feature_y}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hypervolume_2d_{feature_x}_{feature_y}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved 2D projection: {feature_x} vs {feature_y}")
    
    def visualize_feature_distributions(self, output_dir='results/hypervolume'):
        """
        Visualize distributions of environmental features for extreme vs normal fires
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.all_data)
        
        # Get features to plot
        features = [f for f in self.static_names[:7] + ['wind_speed', 'wind_direction'] 
                   if f in df.columns]
        
        # Check if we have intensity data
        if 'fire_intensity' not in df.columns:
            print("⚠️  No fire_intensity data, skipping extreme/normal comparison")
            return
        
        # Separate extreme vs normal
        threshold = np.percentile(df['fire_intensity'], 99)
        df_extreme = df[df['fire_intensity'] > threshold]
        df_normal = df[df['fire_intensity'] <= threshold]
        
        # Plot distributions
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Plot histograms
            ax.hist(df_normal[feature], bins=30, alpha=0.5, label='Normal', color='blue', density=True)
            ax.hist(df_extreme[feature], bins=30, alpha=0.5, label='Extreme (top 1%)', color='red', density=True)
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{feature} Distribution', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Environmental Feature Distributions: Extreme vs Normal Fires', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved feature distributions")
    
    def save_results(self, output_dir='results/hypervolume'):
        """Save collected data and statistics to JSON"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        df = pd.DataFrame(self.all_data)
        df.to_csv(f'{output_dir}/important_pixels_data.csv', index=False)
        print(f"✓ Saved raw data: {output_dir}/important_pixels_data.csv")
        
        # Compute summary statistics
        stats = {
            'total_pixels': len(self.all_data),
            'importance_threshold_percentile': self.importance_threshold,
        }
        
        # Feature statistics
        features = [f for f in self.static_names[:7] + ['wind_speed', 'wind_direction'] 
                   if f in df.columns]
        
        for feature in features:
            stats[f'{feature}_mean'] = float(df[feature].mean())
            stats[f'{feature}_std'] = float(df[feature].std())
            stats[f'{feature}_min'] = float(df[feature].min())
            stats[f'{feature}_max'] = float(df[feature].max())
        
        # Extreme fire statistics
        if 'fire_intensity' in df.columns:
            threshold = np.percentile(df['fire_intensity'], 99)
            df_extreme = df[df['fire_intensity'] > threshold]
            
            stats['extreme_threshold'] = float(threshold)
            stats['n_extreme_pixels'] = len(df_extreme)
            
            for feature in features:
                stats[f'{feature}_extreme_mean'] = float(df_extreme[feature].mean())
                stats[f'{feature}_extreme_std'] = float(df_extreme[feature].std())
        
        # Save statistics
        with open(f'{output_dir}/hypervolume_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved statistics: {output_dir}/hypervolume_statistics.json")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Hypervolume Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, 
                       default='~/data/deep_crown_dataset/organized_spreads',
                       help='Path to dataset root')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to analyze')
    parser.add_argument('--importance_threshold', type=int, default=99,
                       help='Percentile threshold for important pixels (99 = top 1%)')
    parser.add_argument('--extreme_threshold', type=int, default=99,
                       help='Percentile threshold for extreme fires (99 = top 1%)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    
    # Load dataset with resize transform
    print(f"\nLoading dataset from {args.data_root}...")
    with open('configs/emberformer_dino.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    resize_to = cfg['data'].get('resize_to', 406)
    
    class ResizeTransform:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return TF.resize(img, [self.size, self.size],
                           interpolation=TF.InterpolationMode.BILINEAR,
                           antialias=True)
    
    transform = ResizeTransform(resize_to)
    dataset = RawFireDataset(args.data_root, sequence_length=4, transform=transform)
    print(f"Dataset size: {len(dataset)} samples (resized to {resize_to}×{resize_to})\n")
    
    # Initialize analyzer
    analyzer = GradCAMHypervolume(
        model, 
        device=device,
        importance_threshold=args.importance_threshold
    )
    
    # Collect data
    analyzer.collect_samples(dataset, num_samples=args.num_samples)
    
    # Build hypervolume
    print("\n" + "="*60)
    print("Building Environmental Hypervolume")
    print("="*60)
    hypervolume_stats = analyzer.build_hypervolume(extreme_threshold=args.extreme_threshold)
    
    # Visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # 2D projections
    feature_pairs = [
        ('slope', 'wind_speed'),
        ('elevation', 'fuel_load'),
        ('canopy_density', 'wind_speed'),
    ]
    
    for feat_x, feat_y in feature_pairs:
        analyzer.visualize_hypervolume_2d(feat_x, feat_y)
    
    # Feature distributions
    analyzer.visualize_feature_distributions()
    
    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    analyzer.save_results()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to results/hypervolume/")


if __name__ == '__main__':
    main()
