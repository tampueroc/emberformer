"""
Visualize 3D Environmental Hypervolume

Creates 3D scatter plot of U-space with CHE envelope boundary.

Usage:
    python scripts/visualize_hypervolume_3d.py \
        --u_space data/u_space \
        --che data/che \
        --output results/hypervolume_3d.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse


def visualize_3d(u_space_dir, che_dir, output_path, max_points=10000):
    """Create 3D visualization of hypervolume"""
    
    # Load data
    print(f"Loading U-space from {u_space_dir}...")
    uspace = np.load(Path(u_space_dir) / 'extreme.npz')
    U = uspace['U']
    
    print(f"Loading CHE from {che_dir}...")
    che = np.load(Path(che_dir) / 'envelope.npz')
    envelope_mask = che['envelope_mask']
    bounds = che['bounds']
    
    n_components = U.shape[1]
    print(f"U-space dimensions: {n_components}")
    
    if n_components < 3:
        print(f"ERROR: Need at least 3 components for 3D plot, got {n_components}")
        print("Falling back to 2D plot...")
        visualize_2d(u_space_dir, che_dir, output_path.replace('3d', '2d'))
        return
    
    # Subsample for plotting
    if U.shape[0] > max_points:
        print(f"Subsampling {max_points:,} / {U.shape[0]:,} points for visualization...")
        idx = np.random.choice(U.shape[0], max_points, replace=False)
        U_plot = U[idx]
    else:
        U_plot = U
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot extreme fire points
    ax.scatter(U_plot[:, 0], U_plot[:, 1], U_plot[:, 2], 
              c='blue', alpha=0.3, s=1, label='Extreme fire pixels')
    
    # Plot 3D convex hull surface
    if envelope_mask.ndim == 3:
        print("3D envelope detected, extracting isosurface...")
        
        # Option 1: Plot hull vertices if available
        if 'hull_vertices' in che:
            hull_verts = che['hull_vertices']
            # Subsample vertices for cleaner visualization
            if len(hull_verts) > 5000:
                idx = np.random.choice(len(hull_verts), 5000, replace=False)
                hull_verts = hull_verts[idx]
            
            print(f"  Plotting {len(hull_verts)} hull boundary vertices...")
            ax.scatter(hull_verts[:, 0], hull_verts[:, 1], hull_verts[:, 2],
                      c='red', alpha=0.6, s=20, marker='o', 
                      label='CHE envelope surface', edgecolors='darkred', linewidths=0.5)
        
        # Option 2: Extract isosurface from occupancy grid
        else:
            print("  Extracting envelope boundary from occupancy grid...")
            from skimage import measure
            
            # Create isosurface at occupancy threshold
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    occupancy_grid, 
                    level=0.5,  # Binary mask
                    spacing=(
                        (bounds[1][0]-bounds[0][0])/occupancy_grid.shape[0],
                        (bounds[1][1]-bounds[0][1])/occupancy_grid.shape[1],
                        (bounds[1][2]-bounds[0][2])/occupancy_grid.shape[2]
                    )
                )
                
                # Transform to U-space coordinates
                verts[:, 0] += bounds[0][0]
                verts[:, 1] += bounds[0][1]
                verts[:, 2] += bounds[0][2]
                
                # Plot as mesh
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor='red', 
                                       edgecolor='darkred', linewidths=0.2)
                ax.add_collection3d(mesh)
                print(f"  Plotted {len(faces)} surface triangles")
                
            except ImportError:
                print("  scikit-image not available, using hull vertices instead")
    
    elif envelope_mask.ndim == 2:
        print("2D envelope detected, extending to 3D visualization...")
        # For 2D envelope in U1-U2, show as vertical extrusion in U1-U2-U3 space
        grid_res = envelope_mask.shape[0]
        u1_edges = np.linspace(bounds[0][0], bounds[1][0], grid_res)
        u2_edges = np.linspace(bounds[0][1], bounds[1][1], grid_res)
        
        # Get U3 range from data
        u3_min, u3_max = U[:, 2].min(), U[:, 2].max()
        
        envelope_points = []
        for i in range(grid_res):
            for j in range(grid_res):
                if envelope_mask[j, i]:
                    # Add points at min and max U3 for this U1-U2 cell
                    envelope_points.append([u1_edges[i], u2_edges[j], u3_min])
                    envelope_points.append([u1_edges[i], u2_edges[j], u3_max])
        
        if len(envelope_points) > 0:
            envelope_points = np.array(envelope_points)
            ax.scatter(envelope_points[:, 0], envelope_points[:, 1], envelope_points[:, 2],
                      c='red', alpha=0.5, s=10, marker='s', label='CHE envelope boundary')
    
    # Labels
    ax.set_xlabel('U1 (PC1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('U2 (PC2)', fontsize=12, fontweight='bold')
    ax.set_zlabel('U3 (PC3)', fontsize=12, fontweight='bold')
    ax.set_title('3D Environmental Hypervolume\nExtreme Fire Conditions', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 3D visualization to {output_path}")
    
    # Also save rotating view
    angles = [(30, i) for i in range(0, 360, 45)]
    for idx, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)
        angle_path = output_path.parent / f"{output_path.stem}_view{idx}{output_path.suffix}"
        plt.savefig(angle_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {len(angles)} viewing angles")


def visualize_2d(u_space_dir, che_dir, output_path):
    """Fallback 2D visualization"""
    uspace = np.load(Path(u_space_dir) / 'extreme.npz')
    che = np.load(Path(che_dir) / 'envelope.npz')
    
    U = uspace['U']
    envelope_mask = che['envelope_mask']
    bounds = che['bounds']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(U[:, 0], U[:, 1], alpha=0.1, s=1, c='blue', label='Extreme fires')
    
    # Plot envelope
    grid_res = envelope_mask.shape[0]
    u1_edges = np.linspace(bounds[0][0], bounds[1][0], grid_res+1)
    u2_edges = np.linspace(bounds[0][1], bounds[1][1], grid_res+1)
    
    for i in range(grid_res):
        for j in range(grid_res):
            if envelope_mask[j, i]:
                rect = plt.Rectangle((u1_edges[i], u2_edges[j]), 
                                    u1_edges[i+1]-u1_edges[i],
                                    u2_edges[j+1]-u2_edges[j],
                                    fill=False, edgecolor='red', linewidth=0.5)
                ax.add_patch(rect)
    
    ax.set_xlabel('U1 (PC1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('U2 (PC2)', fontsize=12, fontweight='bold')
    ax.set_title('2D Environmental Hypervolume', fontsize=14, fontweight='bold')
    ax.legend()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 2D visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D Environmental Hypervolume')
    parser.add_argument('--u_space', type=str, default='data/u_space',
                       help='U-space directory')
    parser.add_argument('--che', type=str, default='data/che',
                       help='CHE directory')
    parser.add_argument('--output', type=str, default='results/hypervolume_3d.png',
                       help='Output path for visualization')
    parser.add_argument('--max_points', type=int, default=10000,
                       help='Maximum points to plot (subsample if exceeded)')
    
    args = parser.parse_args()
    
    visualize_3d(args.u_space, args.che, args.output, args.max_points)


if __name__ == '__main__':
    main()
