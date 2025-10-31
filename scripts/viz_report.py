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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import argparse
import json
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull


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
    
    def plot_3d_scatter(self, U, gradcam, transform_meta, output_dir):
        """Plot 3D scatter of U1, U2, U3 with convex hull surface"""
        output_dir = Path(output_dir)
        
        if U.shape[1] < 3:
            print(f"⚠️  Skipping 3D plot: only {U.shape[1]} components available")
            return
        
        print(f"Generating 3D scatter plot with convex hull...")
        
        # Subsample for visualization (plot max 50k points for 3D)
        max_points = 50000
        if U.shape[0] > max_points:
            idx = np.random.choice(U.shape[0], max_points, replace=False)
            U_plot = U[idx, :3]
            gradcam_plot = gradcam[idx]
        else:
            U_plot = U[:, :3]
            gradcam_plot = gradcam
        
        # Compute convex hull on subsampled data for surface
        # Further subsample for hull computation if needed (max 10k points)
        if U_plot.shape[0] > 10000:
            idx_hull = np.random.choice(U_plot.shape[0], 10000, replace=False)
            U_hull = U_plot[idx_hull]
        else:
            U_hull = U_plot
        
        try:
            hull = ConvexHull(U_hull)
            print(f"  ✓ Computed convex hull: {len(hull.simplices)} faces, {len(hull.vertices)} vertices")
        except Exception as e:
            print(f"  ⚠️  Failed to compute convex hull: {e}")
            hull = None
        
        # Create 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot convex hull surface (swap axes: U3=x, U2=y, U1=z)
        if hull is not None:
            # Get the vertices for each simplex with swapped axes
            verts = []
            for simplex in hull.simplices:
                # Original: [U1, U2, U3], swap to [U3, U2, U1]
                v0 = [U_hull[simplex[0], 2], U_hull[simplex[0], 1], U_hull[simplex[0], 0]]
                v1 = [U_hull[simplex[1], 2], U_hull[simplex[1], 1], U_hull[simplex[1], 0]]
                v2 = [U_hull[simplex[2], 2], U_hull[simplex[2], 1], U_hull[simplex[2], 0]]
                verts.append([v0, v1, v2])
            
            # Create Poly3DCollection
            poly = Poly3DCollection(verts, alpha=0.15, facecolor='cyan', 
                                   edgecolor='darkblue', linewidths=0.3)
            ax.add_collection3d(poly)
        
        # Scatter with Grad-CAM coloring (swap axes: U3=x, U2=y, U1=z)
        scatter = ax.scatter(
            U_plot[:, 2], U_plot[:, 1], U_plot[:, 0],
            c=gradcam_plot,
            cmap='hot',
            s=2,
            alpha=0.6,
            edgecolors='none'
        )
        
        # Formatting
        variance_u1 = transform_meta['variance_explained'][0] * 100
        variance_u2 = transform_meta['variance_explained'][1] * 100
        variance_u3 = transform_meta['variance_explained'][2] * 100
        
        ax.set_xlabel(f'U3 ({variance_u3:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_zlabel(f'U1 ({variance_u1:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_title('3D U-Space with Convex Hull Envelope', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Grad-CAM Importance', fontsize=11, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Grid
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / 'u_space_3d_hull.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        
        # Save another angle
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot hull again (swap axes: U3=x, U2=y, U1=z)
        if hull is not None:
            verts = []
            for simplex in hull.simplices:
                v0 = [U_hull[simplex[0], 2], U_hull[simplex[0], 1], U_hull[simplex[0], 0]]
                v1 = [U_hull[simplex[1], 2], U_hull[simplex[1], 1], U_hull[simplex[1], 0]]
                v2 = [U_hull[simplex[2], 2], U_hull[simplex[2], 1], U_hull[simplex[2], 0]]
                verts.append([v0, v1, v2])
            poly = Poly3DCollection(verts, alpha=0.15, facecolor='cyan', 
                                   edgecolor='darkblue', linewidths=0.3)
            ax.add_collection3d(poly)
        
        scatter = ax.scatter(
            U_plot[:, 2], U_plot[:, 1], U_plot[:, 0],
            c=gradcam_plot,
            cmap='hot',
            s=2,
            alpha=0.6,
            edgecolors='none'
        )
        
        ax.set_xlabel(f'U3 ({variance_u3:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_zlabel(f'U1 ({variance_u1:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_title('3D U-Space with Convex Hull Envelope (Top View)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Grad-CAM Importance', fontsize=11, fontweight='bold')
        
        ax.view_init(elev=70, azim=45)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        output_path = output_dir / 'u_space_3d_hull_top.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        print(f"{'='*60}\n")
    
    def plot_3d_importance_surface(self, U, gradcam, transform_meta, output_dir):
        """Plot 3D surface: U1 (x), U2 (y), Grad-CAM importance (z)"""
        output_dir = Path(output_dir)
        
        print(f"Generating 3D importance surface plot...")
        
        # Subsample for visualization
        max_points = 50000
        if U.shape[0] > max_points:
            idx = np.random.choice(U.shape[0], max_points, replace=False)
            U_plot = U[idx, :2]
            gradcam_plot = gradcam[idx]
        else:
            U_plot = U[:, :2]
            gradcam_plot = gradcam
        
        # Create 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter with height = importance
        scatter = ax.scatter(
            U_plot[:, 0], U_plot[:, 1], gradcam_plot,
            c=gradcam_plot,
            cmap='hot',
            s=3,
            alpha=0.5,
            edgecolors='none'
        )
        
        # Formatting
        variance_u1 = transform_meta['variance_explained'][0] * 100
        variance_u2 = transform_meta['variance_explained'][1] * 100
        
        ax.set_xlabel(f'U1 ({variance_u1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Grad-CAM Importance', fontsize=12, fontweight='bold')
        ax.set_title('3D Importance Surface: U1 × U2 × Importance', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Grad-CAM Value', fontsize=11, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=25, azim=135)
        
        # Grid
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / 'importance_surface_3d.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        
        # Save another angle (side view)
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            U_plot[:, 0], U_plot[:, 1], gradcam_plot,
            c=gradcam_plot,
            cmap='hot',
            s=3,
            alpha=0.5,
            edgecolors='none'
        )
        
        ax.set_xlabel(f'U1 ({variance_u1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Grad-CAM Importance', fontsize=12, fontweight='bold')
        ax.set_title('3D Importance Surface: U1 × U2 × Importance (Side View)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Grad-CAM Value', fontsize=11, fontweight='bold')
        
        ax.view_init(elev=10, azim=0)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        output_path = output_dir / 'importance_surface_3d_side.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        print(f"{'='*60}\n")
    
    def plot_3d_surface_mesh(self, U, gradcam, transform_meta, output_dir):
        """Plot 3D surface mesh: interpolated surface over U1-U2 grid"""
        output_dir = Path(output_dir)
        
        print(f"Generating 3D surface mesh plot...")
        
        # Subsample for interpolation
        max_points = 100000
        if U.shape[0] > max_points:
            idx = np.random.choice(U.shape[0], max_points, replace=False)
            U_plot = U[idx, :2]
            gradcam_plot = gradcam[idx]
        else:
            U_plot = U[:, :2]
            gradcam_plot = gradcam
        
        # Create regular grid
        grid_res = 50
        u1_min, u1_max = U_plot[:, 0].min(), U_plot[:, 0].max()
        u2_min, u2_max = U_plot[:, 1].min(), U_plot[:, 1].max()
        
        u1_grid = np.linspace(u1_min, u1_max, grid_res)
        u2_grid = np.linspace(u2_min, u2_max, grid_res)
        U1, U2 = np.meshgrid(u1_grid, u2_grid)
        
        # Interpolate Grad-CAM values onto grid
        points = U_plot
        values = gradcam_plot
        Z = griddata(points, values, (U1, U2), method='cubic', fill_value=np.nan)
        
        # Create 3D figure with surface
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(
            U1, U2, Z,
            cmap='hot',
            alpha=0.7,
            edgecolor='none',
            antialiased=True,
            shade=True
        )
        
        # Overlay scattered points (subsampled for clarity)
        sample_size = min(5000, len(U_plot))
        idx_scatter = np.random.choice(len(U_plot), sample_size, replace=False)
        ax.scatter(
            U_plot[idx_scatter, 0], 
            U_plot[idx_scatter, 1], 
            gradcam_plot[idx_scatter],
            c='black',
            s=1,
            alpha=0.2
        )
        
        # Formatting
        variance_u1 = transform_meta['variance_explained'][0] * 100
        variance_u2 = transform_meta['variance_explained'][1] * 100
        
        ax.set_xlabel(f'U1 ({variance_u1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Grad-CAM Importance', fontsize=12, fontweight='bold')
        ax.set_title('3D Importance Surface Mesh: U1 × U2 × Importance', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(surf, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Grad-CAM Value', fontsize=11, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=135)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / 'importance_surface_mesh.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {output_path}")
        
        # Save another angle
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(
            U1, U2, Z,
            cmap='hot',
            alpha=0.7,
            edgecolor='none',
            antialiased=True,
            shade=True
        )
        
        ax.scatter(
            U_plot[idx_scatter, 0], 
            U_plot[idx_scatter, 1], 
            gradcam_plot[idx_scatter],
            c='black',
            s=1,
            alpha=0.2
        )
        
        ax.set_xlabel(f'U1 ({variance_u1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'U2 ({variance_u2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Grad-CAM Importance', fontsize=12, fontweight='bold')
        ax.set_title('3D Importance Surface Mesh (Top View)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(surf, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Grad-CAM Value', fontsize=11, fontweight='bold')
        
        ax.view_init(elev=60, azim=45)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        output_path = output_dir / 'importance_surface_mesh_top.png'
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
    
    # 3D scatter plot with convex hull (U1, U2, U3)
    viz.plot_3d_scatter(U, gradcam, transform_meta, args.output)
    
    print(f"\n{'='*60}")
    print(f"✓ Visualization Complete")
    print(f"{'='*60}")
    print(f"Output: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
