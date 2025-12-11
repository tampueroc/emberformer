"""
Convex Hull Ensemble (CHE) for Environmental Envelope

Stage 3 of hypervolume pipeline:
- Load U-space data
- Subsample boundary points if needed (>20k guard)
- Bootstrap convex hulls in 2D-3D U-subspaces
- Build occupancy grid and threshold for envelope
- Fallback to MVE (Minimum Volume Ellipsoid) if hull fails

Usage:
    python scripts/envelope_che.py \
        --input data/u_space \
        --output data/che \
        --n_bootstraps 200 \
        --occupancy_threshold 0.3 \
        --max_hull_points 10000
"""

import numpy as np
from pathlib import Path
import argparse
import json
from scipy.spatial import ConvexHull
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm
import time


class CHEEnvelope:
    """Convex Hull Ensemble with MVE fallback"""
    
    def __init__(self, n_bootstraps=200, occupancy_threshold=0.3, 
                 max_hull_points=10000, grid_resolution=100):
        self.n_bootstraps = n_bootstraps
        self.occupancy_threshold = occupancy_threshold
        self.max_hull_points = max_hull_points
        self.grid_resolution = grid_resolution
        
        self.hulls = []
        self.occupancy_grid = None
        self.envelope_mask = None
        self.bounds = None
        
    def subsample_boundary(self, U, target_size=10000):
        """Subsample to boundary representatives using MiniBatchKMeans"""
        if U.shape[0] <= target_size:
            return U
        
        print(f"\n{'='*60}")
        print(f"Subsampling for CHE")
        print(f"{'='*60}")
        print(f"Original points: {U.shape[0]:,}")
        print(f"Target points: {target_size:,}")
        
        # Use MiniBatchKMeans for scalability
        kmeans = MiniBatchKMeans(
            n_clusters=target_size,
            batch_size=10000,
            random_state=42,
            verbose=0
        )
        kmeans.fit(U)
        
        U_sampled = kmeans.cluster_centers_
        print(f"  ✓ Subsampled to {U_sampled.shape[0]:,} cluster centroids")
        print(f"{'='*60}\n")
        
        return U_sampled
    
    def bootstrap_hulls(self, U, n_bootstraps, sample_fraction=0.8, n_dims=3):
        """Bootstrap convex hulls in n-dimensional U-space"""
        n_dims = min(n_dims, U.shape[1])  # Don't exceed available dimensions
        
        print(f"\n{'='*60}")
        print(f"Bootstrap Convex Hulls ({n_dims}D)")
        print(f"{'='*60}")
        print(f"Bootstraps: {n_bootstraps}")
        print(f"Sample fraction: {sample_fraction}")
        print(f"Dimensions: {n_dims} (U1-U{n_dims})")
        
        hulls = []
        n_samples = U.shape[0]
        subsample_size = int(n_samples * sample_fraction)
        
        successful = 0
        for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            # Random sample
            idx = np.random.choice(n_samples, subsample_size, replace=True)
            U_boot = U[idx, :n_dims]  # Use first n_dims components
            
            try:
                hull = ConvexHull(U_boot, qhull_options='QJ')
                hulls.append(hull)
                successful += 1
            except Exception:
                continue
        
        print(f"\n  ✓ Successful hulls: {successful}/{n_bootstraps}")
        print(f"{'='*60}\n")
        
        return hulls
    
    def build_occupancy_grid(self, U, hulls, n_dims=3):
        """Build n-D occupancy grid from hull ensemble"""
        n_dims = min(n_dims, U.shape[1])
        
        print(f"\n{'='*60}")
        print(f"Building Occupancy Grid ({n_dims}D)")
        print(f"{'='*60}")
        
        # Determine bounds with padding
        U_subset = U[:, :n_dims]
        mins = U_subset.min(axis=0)
        maxs = U_subset.max(axis=0)
        padding = (maxs - mins) * 0.1
        self.bounds = (mins - padding, maxs + padding)
        self.n_dims = n_dims
        
        # Print bounds
        for i in range(n_dims):
            print(f"U{i+1} range: [{self.bounds[0][i]:.3f}, {self.bounds[1][i]:.3f}]")
        
        # Create grid
        if n_dims == 2:
            print(f"Grid resolution: {self.grid_resolution}×{self.grid_resolution}")
            u1_grid = np.linspace(self.bounds[0][0], self.bounds[1][0], self.grid_resolution)
            u2_grid = np.linspace(self.bounds[0][1], self.bounds[1][1], self.grid_resolution)
            U1, U2 = np.meshgrid(u1_grid, u2_grid)
            grid_points = np.column_stack([U1.ravel(), U2.ravel()])
            grid_shape = (self.grid_resolution, self.grid_resolution)
        elif n_dims == 3:
            print(f"Grid resolution: {self.grid_resolution}×{self.grid_resolution}×{self.grid_resolution}")
            u1_grid = np.linspace(self.bounds[0][0], self.bounds[1][0], self.grid_resolution)
            u2_grid = np.linspace(self.bounds[0][1], self.bounds[1][1], self.grid_resolution)
            u3_grid = np.linspace(self.bounds[0][2], self.bounds[1][2], self.grid_resolution)
            U1, U2, U3 = np.meshgrid(u1_grid, u2_grid, u3_grid)
            grid_points = np.column_stack([U1.ravel(), U2.ravel(), U3.ravel()])
            grid_shape = (self.grid_resolution, self.grid_resolution, self.grid_resolution)
        else:
            raise ValueError(f"Only 2D and 3D grids supported, got {n_dims}D")
        
        # Count occupancy (how many hulls contain each point)
        occupancy = np.zeros(len(grid_points))
        
        for hull in tqdm(hulls, desc="Computing occupancy"):
            # Check which grid points are inside this hull
            # Use Delaunay-based containment check (faster than point-in-hull)
            from scipy.spatial import Delaunay
            delaunay = Delaunay(hull.points[hull.vertices])
            inside = delaunay.find_simplex(grid_points) >= 0
            occupancy[inside] += 1
        
        # Normalize to fraction
        occupancy = occupancy / len(hulls)
        self.occupancy_grid = occupancy.reshape(grid_shape)
        
        # Threshold for envelope
        self.envelope_mask = self.occupancy_grid >= self.occupancy_threshold
        
        # Compute volume/area
        voxel_size = np.prod([(self.bounds[1][i] - self.bounds[0][i]) / self.grid_resolution 
                             for i in range(n_dims)])
        envelope_volume = np.sum(self.envelope_mask) * voxel_size
        
        print(f"\n  ✓ Occupancy grid complete")
        print(f"    Max occupancy: {self.occupancy_grid.max():.3f}")
        print(f"    Envelope {'volume' if n_dims==3 else 'area'} (τ={self.occupancy_threshold}): {envelope_volume:.6e}")
        print(f"    Envelope coverage: {100*np.mean(self.envelope_mask):.2f}% of grid")
        print(f"{'='*60}\n")
        
        return envelope_volume
    
    def fallback_mve(self, U):
        """Fallback to Minimum Volume Ellipsoid (MVE) using EllipticEnvelope"""
        print(f"\n{'='*60}")
        print(f"Fallback: Minimum Volume Ellipsoid (MVE)")
        print(f"{'='*60}")
        
        # Use first 2-3 components
        U_sub = U[:, :min(3, U.shape[1])]
        
        # Fit elliptic envelope
        mve = EllipticEnvelope(
            contamination=0.01,  # Assume 1% outliers
            random_state=42
        )
        mve.fit(U_sub)
        
        # Compute volume (use determinant of covariance)
        # For 2D: area = π * sqrt(det(Σ))
        # For 3D: volume = (4/3)π * det(Σ)^(1/3)
        cov = mve.covariance_
        det = np.linalg.det(cov)
        
        if U_sub.shape[1] == 2:
            volume = np.pi * np.sqrt(det)
            metric_name = "area"
        else:
            volume = (4/3) * np.pi * (det ** (1/3))
            metric_name = "volume"
        
        print(f"  ✓ MVE {metric_name}: {volume:.6e}")
        print(f"  Center: {mve.location_}")
        print(f"{'='*60}\n")
        
        return {
            'method': 'MVE',
            'volume': float(volume),
            'center': mve.location_.tolist(),
            'covariance': cov.tolist(),
        }
    
    def fit(self, U, n_dims=3):
        """Fit CHE to U-space data"""
        n_dims = min(n_dims, U.shape[1])
        
        print(f"\n{'='*60}")
        print(f"CHE Envelope Fitting ({n_dims}D)")
        print(f"{'='*60}")
        print(f"Input: {U.shape[0]:,} points × {U.shape[1]} dimensions")
        print(f"Using: {n_dims} dimensions for CHE")
        
        # Guard: subsample if too many points
        if U.shape[0] > self.max_hull_points:
            U_sub = self.subsample_boundary(U, self.max_hull_points)
        else:
            U_sub = U
        
        # Try CHE
        try:
            # Bootstrap hulls
            hulls = self.bootstrap_hulls(U_sub, self.n_bootstraps, n_dims=n_dims)
            
            if len(hulls) < 10:
                raise ValueError("Too few successful hulls, falling back to MVE")
            
            # Store hulls for later saving
            self.hulls = hulls
            
            # Build occupancy grid
            volume = self.build_occupancy_grid(U, hulls, n_dims=n_dims)
            
            return {
                'method': 'CHE',
                'volume': float(volume),
                'area': float(volume),  # For backwards compat
                'n_hulls': len(hulls),
                'occupancy_threshold': self.occupancy_threshold,
                'grid_resolution': self.grid_resolution,
                'n_dims': n_dims,
            }
        
        except Exception as e:
            print(f"\n⚠️  CHE failed: {e}")
            print(f"   Falling back to MVE...")
            return self.fallback_mve(U)
    
    def save(self, output_dir, stats):
        """Save envelope data and statistics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save grid data if CHE succeeded
        if self.occupancy_grid is not None:
            save_dict = {
                'occupancy_grid': self.occupancy_grid.astype(np.float32),
                'envelope_mask': self.envelope_mask,
                'bounds': np.array(self.bounds),
            }
            
            # Save hull vertices if available (for 3D surface plotting)
            if hasattr(self, 'hulls') and len(self.hulls) > 0:
                # Get combined hull vertices from all bootstrap hulls
                all_vertices = []
                for hull in self.hulls:
                    all_vertices.append(hull.points[hull.vertices])
                save_dict['hull_vertices'] = np.vstack(all_vertices).astype(np.float32)
            
            np.savez_compressed(output_dir / 'envelope.npz', **save_dict)
            print(f"  ✓ Saved {output_dir}/envelope.npz")
        
        # Save statistics
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved {output_dir}/summary.json")


def main():
    parser = argparse.ArgumentParser(description='CHE Environmental Envelope')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with U-space data')
    parser.add_argument('--output', type=str, default='data/che',
                       help='Output directory for CHE results')
    parser.add_argument('--n_bootstraps', type=int, default=200,
                       help='Number of bootstrap iterations')
    parser.add_argument('--occupancy_threshold', type=float, default=0.3,
                       help='Occupancy threshold for envelope (0.3 = 30%% of hulls)')
    parser.add_argument('--max_hull_points', type=int, default=10000,
                       help='Maximum points for hull computation (subsample if exceeded)')
    parser.add_argument('--grid_resolution', type=int, default=100,
                       help='Resolution of occupancy grid')
    parser.add_argument('--n_dims', type=int, default=3,
                       help='Number of dimensions to use (2=2D, 3=3D hull)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Load U-space data
    print(f"\n{'='*60}")
    print(f"Loading U-Space Data")
    print(f"{'='*60}")
    
    input_dir = Path(args.input)
    data = np.load(input_dir / 'extreme.npz')
    U = data['U']
    
    print(f"Loaded: {U.shape[0]:,} points × {U.shape[1]} dimensions")
    print(f"{'='*60}\n")
    
    # Fit CHE
    che = CHEEnvelope(
        n_bootstraps=args.n_bootstraps,
        occupancy_threshold=args.occupancy_threshold,
        max_hull_points=args.max_hull_points,
        grid_resolution=args.grid_resolution
    )
    
    start_time = time.time()
    stats = che.fit(U, n_dims=args.n_dims)
    elapsed = time.time() - start_time
    
    stats['elapsed_seconds'] = elapsed
    stats['n_samples'] = U.shape[0]
    stats['n_dimensions'] = U.shape[1]
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Saving Results")
    print(f"{'='*60}")
    che.save(args.output, stats)
    
    print(f"\n{'='*60}")
    print(f"✓ CHE Complete")
    print(f"{'='*60}")
    print(f"Method: {stats['method']}")
    if stats['method'] == 'CHE':
        print(f"Envelope area: {stats['area']:.6e}")
        print(f"Hulls: {stats['n_hulls']}")
    else:
        print(f"Envelope volume: {stats['volume']:.6e}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
