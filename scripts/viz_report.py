"""
Visualize U-Space Importance Surface and CHE Envelope

Stage 4 of hypervolume pipeline:
- Load U-space data and CHE envelope
- Build 2D importance heatmap (z-scored Grad-CAM)
- Overlay CHE envelope contour
- Generate publication-quality figures

Usage:
    python scripts/viz_report.py \
        --u_space data/u_space \
        --che data/che \
        --output results/hypervolume
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from scipy.stats import binned_statistic_2d


class USpaceVisualizer:
    """Visualize importance surface in U-space with CHE envelope"""
    
    def __init__(self, grid_resolution=100):
        self.grid_resolution = grid_resolution
        
    def build_importance_surface(self, U, gradcam, bounds=None):
        """Build z-scored importance heatmap from Grad-CAM values"""
        print(f"\n{'='*60}")
        print(f"Building Importance Surface")
        print(f"{'='*60}")
        
        U_2d = U[:, :2]
        
        # Determine bounds
        if bounds is None:
            mins = U_2d.min(axis=0)
            maxs = U_2d.max(axis=0)
            padding = (maxs - mins) * 0.1
            bounds = (mins - padding, maxs + padding)
        
        # Bin Grad-CAM values
        stat, x_edges, y_edges, _ = binned_statistic_2d(
            U_2d[:, 0], U_2d[:, 1], gradcam,
            statistic='mean',
            bins=self.grid_resolution,
            range=[[bounds[0][0], bounds[1][0]], 
                   [bounds[0][1], bounds[1][1]]]
        )
        
        # Z-score the grid (symmetric around 0)
        valid_mask = ~np.isnan(stat)
        if valid_mask.sum() > 0:
            mean_val = np.nanmean(stat)
            std_val = np.nanstd(stat)
            stat_zscore = (stat - mean_val) / (std_val + 1e-8)
        else:
            stat_zscore = stat
        
        print(f"  ✓ Grid resolution: {self.grid_resolution}×{self.grid_resolution}")
        print(f"  ✓ Valid bins: {valid_mask.sum():,}/{self.grid_resolution**2}")
        print(f"  ✓ Z-score range: [{np.nanmin(stat_zscore):.2f}, {np.nanmax(stat_zscore):.2f}]")
        print(f"{'='*60}\n")
        
        return stat_zscore, x_edges, y_edges
    
    def plot_importance_with_envelope(self, U, gradcam, envelope_data, 
                                     transform_meta, output_dir):
        """Plot U1×U2 importance heatmap with CHE contour"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating Visualization")
        print(f"{'='*60}")
        
        # Build importance surface
        bounds = None
        if envelope_data is not None and 'bounds' in envelope_data:
            bounds = (envelope_data['bounds'][0], envelope_data['bounds'][1])
        
        importance, x_edges, y_edges = self.build_importance_surface(U, gradcam, bounds)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot importance heatmap
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        im = ax.imshow(
            importance.T,
            origin='lower',
            extent=extent,
            cmap='RdYlBu_r',  # Red = high importance, Blue = low
            aspect='auto',
            interpolation='bilinear',
            vmin=-3, vmax=3  # Symmetric z-score range
        )
        
        # Add CHE envelope contour if available
        if envelope_data is not None and 'envelope_mask' in envelope_data:
            mask = envelope_data['envelope_mask']
            
            # Create contour coordinates
            x_grid = np.linspace(extent[0], extent[1], mask.shape[1])
            y_grid = np.linspace(extent[2], extent[3], mask.shape[0])
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Plot contour
            ax.contour(X, Y, mask, levels=[0.5], colors='black', 
                      linewidths=2.5, linestyles='solid')
            ax.contour(X, Y, mask, levels=[0.5], colors='lime', 
                      linewidths=1.5, linestyles='solid', 
                      label='Environmental Envelope (CHE)')
        
        # Formatting
        variance_u1 = transform_meta['variance_explained'][0] * 100
        variance_u2 = transform_meta['variance_explained'][1] * 100
        
        ax.set_xlabel(f'U1 ({variance_u1:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.set_title('Extreme Fire Environmental Hypervolume\nImportance Surface (z-scored Grad-CAM)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance (z-score)', fontsize=12, fontweight='bold')
        
        # Legend if envelope exists
        if envelope_data is not None and 'envelope_mask' in envelope_data:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / 'importance_surface_u1_u2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        print(f"{'='*60}\n")
    
    def plot_scatter_overlay(self, U, gradcam, envelope_data, 
                            transform_meta, output_dir):
        """Plot scatter of U-space points with envelope overlay"""
        output_dir = Path(output_dir)
        
        print(f"Generating scatter plot...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Subsample for visualization (plot max 10k points)
        if U.shape[0] > 10000:
            idx = np.random.choice(U.shape[0], 10000, replace=False)
            U_plot = U[idx, :2]
            gradcam_plot = gradcam[idx]
        else:
            U_plot = U[:, :2]
            gradcam_plot = gradcam
        
        # Scatter with Grad-CAM coloring
        scatter = ax.scatter(
            U_plot[:, 0], U_plot[:, 1],
            c=gradcam_plot,
            cmap='hot',
            s=5,
            alpha=0.3,
            edgecolors='none'
        )
        
        # Add CHE envelope contour
        if envelope_data is not None and 'envelope_mask' in envelope_data:
            mask = envelope_data['envelope_mask']
            bounds = envelope_data['bounds']
            
            x_grid = np.linspace(bounds[0][0], bounds[1][0], mask.shape[1])
            y_grid = np.linspace(bounds[0][1], bounds[1][1], mask.shape[0])
            X, Y = np.meshgrid(x_grid, y_grid)
            
            ax.contour(X, Y, mask, levels=[0.5], colors='cyan', 
                      linewidths=3, linestyles='solid',
                      label='Environmental Envelope')
        
        # Formatting
        variance_u1 = transform_meta['variance_explained'][0] * 100
        variance_u2 = transform_meta['variance_explained'][1] * 100
        
        ax.set_xlabel(f'U1 ({variance_u1:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.set_title('U-Space: Extreme Fire Pixels with Environmental Envelope', 
                    fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Grad-CAM Importance', fontsize=12, fontweight='bold')
        
        if envelope_data is not None:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_path = output_dir / 'u_space_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize U-Space Importance Surface')
    parser.add_argument('--u_space', type=str, required=True,
                       help='Directory with U-space data')
    parser.add_argument('--che', type=str, default=None,
                       help='Directory with CHE envelope (optional)')
    parser.add_argument('--output', type=str, default='results/hypervolume',
                       help='Output directory for visualizations')
    parser.add_argument('--grid_resolution', type=int, default=100,
                       help='Resolution for importance surface grid')
    
    args = parser.parse_args()
    
    # Load U-space data
    print(f"\n{'='*60}")
    print(f"Loading Data")
    print(f"{'='*60}")
    
    u_space_dir = Path(args.u_space)
    data = np.load(u_space_dir / 'extreme.npz')
    U = data['U']
    gradcam = data['gradcam']
    
    print(f"U-space: {U.shape[0]:,} points × {U.shape[1]} dimensions")
    
    # Load transform metadata
    with open(u_space_dir / 'transform.json', 'r') as f:
        transform_meta = json.load(f)
    
    print(f"PCA variance: {transform_meta['total_variance']*100:.2f}%")
    
    # Load CHE envelope if available
    envelope_data = None
    if args.che:
        che_dir = Path(args.che)
        if (che_dir / 'envelope.npz').exists():
            envelope_npz = np.load(che_dir / 'envelope.npz')
            envelope_data = {
                'envelope_mask': envelope_npz['envelope_mask'],
                'occupancy_grid': envelope_npz['occupancy_grid'],
                'bounds': envelope_npz['bounds'],
            }
            print(f"CHE envelope: loaded")
        
        if (che_dir / 'summary.json').exists():
            with open(che_dir / 'summary.json', 'r') as f:
                che_stats = json.load(f)
            print(f"CHE method: {che_stats['method']}")
            if 'area' in che_stats:
                print(f"CHE area: {che_stats['area']:.6e}")
    
    print(f"{'='*60}\n")
    
    # Visualize
    viz = USpaceVisualizer(grid_resolution=args.grid_resolution)
    
    # Importance surface
    viz.plot_importance_with_envelope(U, gradcam, envelope_data, 
                                     transform_meta, args.output)
    
    # Scatter overlay
    viz.plot_scatter_overlay(U, gradcam, envelope_data, 
                            transform_meta, args.output)
    
    print(f"\n{'='*60}")
    print(f"✓ Visualization Complete")
    print(f"{'='*60}")
    print(f"Output: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
