"""
Convex Hull Ensemble (CHE) for Environmental Hypervolume

Stage 2 of hypervolume pipeline:
- Load U-space data from 99th percentile extreme fires
- Build bootstrap ensemble of convex hulls
- Save ensemble for occupancy-based classification in danger map

The CHE approach uses multiple bootstrap-sampled hulls. A point is classified
by its "occupancy" - the fraction of hulls it falls within.

Usage:
    python scripts/pipeline/envelope_che.py \
        --input data/u_space \
        --output data/che \
        --n_bootstraps 100 \
        --occupancy_threshold 0.5
"""

import numpy as np
from pathlib import Path
import argparse
import json
import subprocess
from scipy.spatial import ConvexHull
from tqdm import tqdm
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


class ConvexHullEnsemble:
    """Convex Hull Ensemble for occupancy-based classification"""
    
    def __init__(self, n_dims=None, n_bootstraps=100, occupancy_threshold=0.5,
                 max_hull_points=10000, seed=42):
        self.n_dims = n_dims
        self.n_bootstraps = n_bootstraps
        self.occupancy_threshold = occupancy_threshold
        self.max_hull_points = max_hull_points
        self.seed = seed
        self.hull_vertices = []  # List of vertex coordinate arrays
        self.hull_stats = []  # Per-hull statistics
    
    def fit(self, U):
        """Build bootstrap ensemble of convex hulls on U-space points"""
        N = U.shape[0]
        n_dims = self.n_dims if self.n_dims else U.shape[1]
        n_dims = min(n_dims, U.shape[1])
        self.n_dims = n_dims
        
        U_subset = U[:, :n_dims].astype(np.float32)
        
        print(f"\n{'='*60}")
        print(f"Building Convex Hull Ensemble ({n_dims}D)")
        print(f"{'='*60}")
        print(f"Total points: {N:,}")
        print(f"Dimensions: {n_dims}")
        print(f"Bootstraps: {self.n_bootstraps}")
        print(f"Max hull points: {self.max_hull_points:,}")
        print(f"Occupancy threshold: {self.occupancy_threshold}")
        print(f"Random seed: {self.seed}")
        
        rng = np.random.default_rng(self.seed)
        
        self.hull_vertices = []
        self.hull_stats = []
        
        for b in tqdm(range(self.n_bootstraps), desc="Building hulls"):
            if N > self.max_hull_points:
                # Sample without replacement when we have more points than needed
                indices = rng.choice(N, size=self.max_hull_points, replace=False)
            else:
                # Bootstrap with replacement when we have fewer points
                indices = rng.choice(N, size=N, replace=True)
            
            U_boot = U_subset[indices]
            
            try:
                hull = ConvexHull(U_boot, qhull_options='QJ')
                V = U_boot[hull.vertices]
                
                self.hull_vertices.append(V)
                self.hull_stats.append({
                    'n_vertices': len(hull.vertices),
                    'volume': float(hull.volume),
                })
            except Exception as e:
                print(f"\n  Warning: Hull {b} failed: {e}")
                continue
        
        n_successful = len(self.hull_vertices)
        vertices_counts = [s['n_vertices'] for s in self.hull_stats]
        volumes = [s['volume'] for s in self.hull_stats]
        
        print(f"\n  ✓ CHE ensemble complete")
        print(f"    Successful hulls: {n_successful}/{self.n_bootstraps}")
        print(f"    Vertices: mean={np.mean(vertices_counts):.1f}, "
              f"median={np.median(vertices_counts):.1f}, "
              f"range=[{np.min(vertices_counts)}, {np.max(vertices_counts)}]")
        print(f"    Volume: mean={np.mean(volumes):.2e}, "
              f"median={np.median(volumes):.2e}")
        print(f"{'='*60}\n")
        
        return {
            'method': 'CHE',
            'n_bootstraps': self.n_bootstraps,
            'n_successful': n_successful,
            'occupancy_threshold': self.occupancy_threshold,
            'n_dims': n_dims,
            'seed': self.seed,
            'max_hull_points': self.max_hull_points,
            'n_samples': N,
            'vertices_mean': float(np.mean(vertices_counts)),
            'vertices_median': float(np.median(vertices_counts)),
            'volume_mean': float(np.mean(volumes)),
            'volume_median': float(np.median(volumes)),
        }
    
    def save(self, output_dir, stats):
        """Save CHE ensemble data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Concatenate all vertex arrays with offsets for indexing
        vertices_concat = np.concatenate(self.hull_vertices, axis=0)
        
        # Build offsets array: hull i vertices are vertices_concat[offsets[i]:offsets[i+1]]
        offsets = np.zeros(len(self.hull_vertices) + 1, dtype=np.int64)
        for i, V in enumerate(self.hull_vertices):
            offsets[i + 1] = offsets[i] + len(V)
        
        np.savez_compressed(
            output_dir / 'che_ensemble.npz',
            n_dims=self.n_dims,
            n_bootstraps=len(self.hull_vertices),
            vertices_concat=vertices_concat.astype(np.float32),
            offsets=offsets,
        )
        print(f"  ✓ Saved {output_dir}/che_ensemble.npz")
        print(f"    Total vertices: {len(vertices_concat):,}")
        print(f"    Hulls: {len(self.hull_vertices)}")
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved {output_dir}/summary.json")


def main():
    parser = argparse.ArgumentParser(description='Convex Hull Ensemble (CHE)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with U-space data')
    parser.add_argument('--output', type=str, default='data/che',
                       help='Output directory for CHE ensemble')
    parser.add_argument('--n_dims', type=int, default=None,
                       help='Number of dimensions to use (default: all)')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap samples (default: 100)')
    parser.add_argument('--occupancy_threshold', type=float, default=0.5,
                       help='Fraction of hulls for inside classification (default: 0.5)')
    parser.add_argument('--max_hull_points', type=int, default=10000,
                       help='Max points per hull (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
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
    
    # Fit CHE ensemble
    che = ConvexHullEnsemble(
        n_dims=args.n_dims,
        n_bootstraps=args.n_bootstraps,
        occupancy_threshold=args.occupancy_threshold,
        max_hull_points=args.max_hull_points,
        seed=args.seed
    )
    
    start_time = time.time()
    stats = che.fit(U)
    elapsed = time.time() - start_time
    
    stats['elapsed_seconds'] = elapsed
    
    # Save
    print(f"\n{'='*60}")
    print(f"Saving Results")
    print(f"{'='*60}")
    che.save(output_dir, stats)
    
    print(f"\n{'='*60}")
    print(f"✓ CHE Ensemble Complete")
    print(f"{'='*60}")
    print(f"Bootstraps: {stats['n_successful']}/{stats['n_bootstraps']}")
    print(f"Mean vertices: {stats['vertices_mean']:.1f}")
    print(f"Mean volume: {stats['volume_mean']:.2e}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
