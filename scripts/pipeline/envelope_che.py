"""
Convex Hull Envelope for Environmental Hypervolume

Stage 3 of hypervolume pipeline:
- Load U-space data from 99th percentile extreme fires
- Build single convex hull enclosing all points
- Save hull for binary classification in danger map

Usage:
    python scripts/pipeline/envelope_che.py \
        --input data/u_space \
        --output data/che
"""

import numpy as np
from pathlib import Path
import argparse
import json
import subprocess
from scipy.spatial import ConvexHull, Delaunay
import time


def get_git_commit_hash():
    """Get short git commit hash for output directory naming"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


class ConvexHullEnvelope:
    """Single convex hull envelope for binary classification"""
    
    def __init__(self, n_dims=None):
        self.n_dims = n_dims
        self.hull = None
        self.delaunay = None
        self.bounds = None
    
    def fit(self, U):
        """Build convex hull on U-space points"""
        # Use specified dims or all available
        n_dims = self.n_dims if self.n_dims else U.shape[1]
        n_dims = min(n_dims, U.shape[1])
        self.n_dims = n_dims
        
        U_subset = U[:, :n_dims]
        
        print(f"\n{'='*60}")
        print(f"Building Convex Hull ({n_dims}D)")
        print(f"{'='*60}")
        print(f"Points: {U_subset.shape[0]:,}")
        print(f"Dimensions: {n_dims}")
        
        # Build convex hull
        self.hull = ConvexHull(U_subset, qhull_options='QJ')
        
        # Build Delaunay triangulation for fast point-in-hull queries
        self.delaunay = Delaunay(U_subset[self.hull.vertices])
        
        # Store bounds
        self.bounds = (U_subset.min(axis=0), U_subset.max(axis=0))
        
        print(f"\n  ✓ Convex hull complete")
        print(f"    Vertices: {len(self.hull.vertices):,}")
        print(f"    Volume: {self.hull.volume:.6e}")
        for i in range(n_dims):
            print(f"    U{i+1} range: [{self.bounds[0][i]:.3f}, {self.bounds[1][i]:.3f}]")
        print(f"{'='*60}\n")
        
        return {
            'method': 'ConvexHull',
            'n_vertices': len(self.hull.vertices),
            'volume': float(self.hull.volume),
            'n_dims': n_dims,
        }
    
    def contains(self, points):
        """Check if points are inside the convex hull
        
        Returns:
            Boolean array: True if inside hull (dangerous)
        """
        points = np.atleast_2d(points)
        points_subset = points[:, :self.n_dims]
        return self.delaunay.find_simplex(points_subset) >= 0
    
    def save(self, output_dir, stats):
        """Save hull data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save hull vertices and Delaunay for point-in-hull queries
        np.savez_compressed(
            output_dir / 'envelope.npz',
            hull_points=self.hull.points.astype(np.float32),
            hull_vertices=self.hull.vertices,
            bounds=np.array(self.bounds),
            n_dims=self.n_dims,
        )
        print(f"  ✓ Saved {output_dir}/envelope.npz")
        
        # Save statistics
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved {output_dir}/summary.json")


def main():
    parser = argparse.ArgumentParser(description='Convex Hull Envelope')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with U-space data')
    parser.add_argument('--output', type=str, default='data/che',
                       help='Output directory for envelope')
    parser.add_argument('--n_dims', type=int, default=None,
                       help='Number of dimensions to use (default: all)')
    
    args = parser.parse_args()
    
    commit_hash = get_git_commit_hash()
    output_dir = Path(args.output) / commit_hash
    print(f"Git commit: {commit_hash}")
    print(f"Output directory: {output_dir}\n")
    
    # Load U-space data
    print(f"\n{'='*60}")
    print(f"Loading U-Space Data")
    print(f"{'='*60}")
    
    input_dir = Path(args.input)
    data = np.load(input_dir / 'extreme.npz')
    U = data['U']
    
    print(f"Loaded: {U.shape[0]:,} points × {U.shape[1]} dimensions")
    print(f"{'='*60}\n")
    
    # Fit convex hull
    envelope = ConvexHullEnvelope(n_dims=args.n_dims)
    
    start_time = time.time()
    stats = envelope.fit(U)
    elapsed = time.time() - start_time
    
    stats['elapsed_seconds'] = elapsed
    stats['n_samples'] = U.shape[0]
    
    # Save
    print(f"\n{'='*60}")
    print(f"Saving Results")
    print(f"{'='*60}")
    envelope.save(output_dir, stats)
    
    print(f"\n{'='*60}")
    print(f"✓ Convex Hull Complete")
    print(f"{'='*60}")
    print(f"Vertices: {stats['n_vertices']:,}")
    print(f"Volume: {stats['volume']:.6e}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
