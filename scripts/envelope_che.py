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
    
    def bootstrap_hulls_2d(self, U, n_bootstraps, sample_fraction=0.8):
        """Bootstrap 2D convex hulls in U1-U2 plane"""
        print(f"\n{'='*60}")
        print(f"Bootstrap Convex Hulls (2D)")
        print(f"{'='*60}")
        print(f"Bootstraps: {n_bootstraps}")
        print(f"Sample fraction: {sample_fraction}")
        
        hulls = []
        n_samples = U.shape[0]
        subsample_size = int(n_samples * sample_fraction)
        
        successful = 0
        for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            # Random sample
            idx = np.random.choice(n_samples, subsample_size, replace=True)
            U_boot = U[idx, :2]  # Use only U1, U2
            
            try:
                hull = ConvexHull(U_boot, qhull_options='QJ')
                hulls.append(hull)
                successful += 1
            except Exception:
                continue
        
        print(f"\n  ✓ Successful hulls: {successful}/{n_bootstraps}")
        print(f"{'='*60}\n")
        
        return hulls
    
    def build_occupancy_grid(self, U, hulls):
        """Build 2D occupancy grid from hull ensemble"""
        print(f"\n{'='*60}")
        print(f"Building Occupancy Grid")
        print(f"{'='*60}")
        
        # Determine bounds with padding
        U_2d = U[:, :2]
        mins = U_2d.min(axis=0)
        maxs = U_2d.max(axis=0)
        padding = (maxs - mins) * 0.1
        self.bounds = (mins - padding, maxs + padding)
        
        print(f"Grid resolution: {self.grid_resolution}×{self.grid_resolution}")
        print(f"U1 range: [{self.bounds[0][0]:.3f}, {self.bounds[1][0]:.3f}]")
        print(f"U2 range: [{self.bounds[0][1]:.3f}, {self.bounds[1][1]:.3f}]")
        
        # Create grid
        u1_grid = np.linspace(self.bounds[0][0], self.bounds[1][0], self.grid_resolution)
        u2_grid = np.linspace(self.bounds[0][1], self.bounds[1][1], self.grid_resolution)
        U1, U2 = np.meshgrid(u1_grid, u2_grid)
        grid_points = np.column_stack([U1.ravel(), U2.ravel()])
        
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
        self.occupancy_grid = occupancy.reshape(self.grid_resolution, self.grid_resolution)
        
        # Threshold for envelope
        self.envelope_mask = self.occupancy_grid >= self.occupancy_threshold
        
        # Compute area (count pixels above threshold)
        pixel_area = ((self.bounds[1][0] - self.bounds[0][0]) / self.grid_resolution) * \
                     ((self.bounds[1][1] - self.bounds[0][1]) / self.grid_resolution)
        envelope_area = np.sum(self.envelope_mask) * pixel_area
        
        print(f"\n  ✓ Occupancy grid complete")
        print(f"    Max occupancy: {self.occupancy_grid.max():.3f}")
        print(f"    Envelope area (τ={self.occupancy_threshold}): {envelope_area:.6e}")
        print(f"    Envelope coverage: {100*np.mean(self.envelope_mask):.2f}% of grid")
        print(f"{'='*60}\n")
        
        return envelope_area
    
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
    
    def fit(self, U):
        """Fit CHE to U-space data"""
        print(f"\n{'='*60}")
        print(f"CHE Envelope Fitting")
        print(f"{'='*60}")
        print(f"Input: {U.shape[0]:,} points × {U.shape[1]} dimensions")
        
        # Guard: subsample if too many points
        if U.shape[0] > self.max_hull_points:
            U_sub = self.subsample_boundary(U, self.max_hull_points)
        else:
            U_sub = U
        
        # Try CHE
        try:
            # Bootstrap hulls
            hulls = self.bootstrap_hulls_2d(U_sub, self.n_bootstraps)
            
            if len(hulls) < 10:
                raise ValueError("Too few successful hulls, falling back to MVE")
            
            # Build occupancy grid
            area = self.build_occupancy_grid(U, hulls)
            
            return {
                'method': 'CHE',
                'area': float(area),
                'n_hulls': len(hulls),
                'occupancy_threshold': self.occupancy_threshold,
                'grid_resolution': self.grid_resolution,
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
            np.savez_compressed(
                output_dir / 'envelope.npz',
                occupancy_grid=self.occupancy_grid.astype(np.float32),
                envelope_mask=self.envelope_mask,
                bounds=np.array(self.bounds),
            )
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
    stats = che.fit(U)
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
