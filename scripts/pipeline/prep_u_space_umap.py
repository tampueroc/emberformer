"""
Prepare U-Space from Salience Data (UMAP variant)

Stage 2 of hypervolume pipeline:
- Load Parquet salience data
- Filter for extreme fires (top percentile)
- Feature engineering: circular encodings for aspect/wind_direction
- Z-score normalization
- UMAP projection to U-space (≤5 components)

Usage:
    python scripts/pipeline/prep_u_space_umap.py \
        --input data/salience \
        --output data/u_space \
        --extreme_threshold 99 \
        --max_components 5
"""

import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import argparse
import json
import subprocess
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Try GPU UMAP first, fall back to CPU
try:
    from cuml.manifold import UMAP
    UMAP_BACKEND = "gpu"
    print("Using GPU UMAP (cuML)")
except ImportError:
    from umap import UMAP
    UMAP_BACKEND = "cpu"
    print("Using CPU UMAP (umap-learn)")


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


class USpacePrepUMAP:
    """Prepare U-space from salience data using UMAP"""
    
    def __init__(self, max_components=5, n_neighbors=15, min_dist=0.1, 
                 metric='euclidean', random_state=42, min_feature_std=0.01):
        self.max_components = max_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.min_feature_std = min_feature_std
        self.scaler = None
        self.umap = None
        self.feature_names = None
        
    def load_extreme_fires(self, salience_dir, extreme_threshold=99):
        """Load only extreme fire pixels from Parquet files"""
        salience_dir = Path(salience_dir)
        
        with open(salience_dir / 'quantiles.json', 'r') as f:
            quantiles = json.load(f)
        
        threshold_val = quantiles[f'p{extreme_threshold}']
        print(f"\n{'='*60}")
        print(f"Loading Extreme Fires (>{extreme_threshold}th percentile)")
        print(f"{'='*60}")
        print(f"Fire intensity threshold: {threshold_val:.4f}")
        
        parquet_files = sorted(salience_dir.glob('part-*.parquet'))
        print(f"Found {len(parquet_files)} parquet files")
        
        data_chunks = []
        total_rows = 0
        extreme_rows = 0
        
        for pfile in tqdm(parquet_files, desc="Loading"):
            table = pq.read_table(pfile, filters=[
                ('fire_intensity', '>', threshold_val)
            ])
            
            total_rows += pq.read_table(pfile).num_rows
            extreme_rows += table.num_rows
            
            if table.num_rows > 0:
                data_chunks.append(table.to_pydict())
        
        print(f"\n{'='*60}")
        print(f"Total pixels: {total_rows:,}")
        print(f"Extreme pixels: {extreme_rows:,} ({100*extreme_rows/total_rows:.2f}%)")
        print(f"{'='*60}\n")
        
        if len(data_chunks) == 0:
            raise ValueError("No extreme fire pixels found!")
        
        combined = {k: np.concatenate([chunk[k] for chunk in data_chunks]) 
                   for k in data_chunks[0].keys()}
        
        return combined
    
    def engineer_features(self, data):
        """Apply feature engineering transformations"""
        print(f"\n{'='*60}")
        print(f"Feature Engineering")
        print(f"{'='*60}")
        
        features = {}
        feature_list = []
        
        valid_features = ['forest', 'cbd', 'cbh', 'elevation']
        excluded_features = ['arqueo', 'flora', 'paleo', 'urbana']
        
        for feat in valid_features:
            if feat in data:
                features[feat] = np.array(data[feat], dtype=np.float32)
                feature_list.append(feat)
                print(f"  ✓ {feat}: {features[feat].shape[0]:,} values")
        
        for feat in excluded_features:
            if feat in data:
                print(f"  ⊗ {feat}: excluded (nodata-dominated, constant in landscape)")
        
        if 'wind_speed' in data and 'wind_direction' in data:
            print(f"  ⊗ wind: excluded (temporal, not landscape-based)")
        
        print(f"\nTotal features: {len(feature_list)}")
        print(f"{'='*60}\n")
        
        X = np.column_stack([features[f] for f in feature_list])
        
        metadata = {
            'gradcam': np.array(data['gradcam'], dtype=np.float32),
            'fire_intensity': np.array(data['fire_intensity'], dtype=np.float32),
        }
        
        if 'y_landscape' in data:
            metadata['y'] = np.array(data['y_landscape'], dtype=np.int32)
            metadata['x'] = np.array(data['x_landscape'], dtype=np.int32)
        else:
            metadata['y'] = np.array(data['y'], dtype=np.int32)
            metadata['x'] = np.array(data['x'], dtype=np.int32)
        
        self.feature_names = feature_list
        return X, metadata
    
    def fit_transform(self, X):
        """Z-score normalization + UMAP projection"""
        print(f"\n{'='*60}")
        print(f"UMAP Transformation")
        print(f"{'='*60}")
        print(f"Input shape: {X.shape}")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        print(f"  ✓ Z-score normalized (mean=0, std=1)")
        
        n_components = min(self.max_components, X_scaled.shape[1])
        
        print(f"\n  UMAP parameters:")
        print(f"    backend: {UMAP_BACKEND}")
        print(f"    n_components: {n_components}")
        print(f"    n_neighbors: {self.n_neighbors}")
        print(f"    min_dist: {self.min_dist}")
        print(f"    metric: {self.metric}")
        
        if UMAP_BACKEND == "gpu":
            self.umap = UMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                verbose=True
            )
            U = self.umap.fit_transform(X_scaled)
        else:
            self.umap = UMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                verbose=True
            )
            U = self.umap.fit_transform(X_scaled)
        
        print(f"\n  ✓ UMAP complete:")
        print(f"    Output shape: {U.shape}")
        print(f"{'='*60}\n")
        
        return U
    
    def save(self, U, metadata, output_dir):
        """Save U-space data and transformation metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_dir / 'extreme.npz',
            U=U.astype(np.float32),
            gradcam=metadata['gradcam'],
            fire_intensity=metadata['fire_intensity'],
            y=metadata['y'],
            x=metadata['x'],
        )
        
        transform_meta = {
            'method': 'umap',
            'feature_names': self.feature_names,
            'n_components': U.shape[1],
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist(),
            'n_samples': U.shape[0],
        }
        
        with open(output_dir / 'transform.json', 'w') as f:
            json.dump(transform_meta, f, indent=2)
        
        # Save fitted UMAP model for transform() on new data
        with open(output_dir / 'umap_model.pkl', 'wb') as f:
            pickle.dump(self.umap, f)
        
        # Save fitted scaler
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"{'='*60}")
        print(f"✓ Saved U-space data (UMAP)")
        print(f"{'='*60}")
        print(f"  {output_dir}/extreme.npz ({U.shape[0]:,} points × {U.shape[1]} dims)")
        print(f"  {output_dir}/transform.json (UMAP metadata)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare U-Space from Salience Data (UMAP)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with Parquet salience files')
    parser.add_argument('--output', type=str, default='data/u_space',
                       help='Output directory for U-space data')
    parser.add_argument('--extreme_threshold', type=int, default=99,
                       help='Percentile threshold for extreme fires (99 = top 1%%)')
    parser.add_argument('--max_components', type=int, default=5,
                       help='Maximum number of UMAP components to keep')
    parser.add_argument('--variance_threshold', type=float, default=0.80,
                       help='Unused (kept for CLI compatibility with PCA version)')
    parser.add_argument('--min_feature_std', type=float, default=0.10,
                       help='Minimum std for feature to be kept (default=0.10)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='Number of neighbors for UMAP (default: 15)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='Minimum distance for UMAP (default: 0.1)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples for UMAP (subsample if exceeded)')
    
    args = parser.parse_args()
    
    commit_hash = get_git_commit_hash()
    output_dir = Path(args.output) / commit_hash
    print(f"Git commit: {commit_hash}")
    print(f"Output directory: {output_dir}\n")
    
    prep = USpacePrepUMAP(
        max_components=args.max_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_feature_std=args.min_feature_std
    )
    
    data = prep.load_extreme_fires(args.input, args.extreme_threshold)
    X, metadata = prep.engineer_features(data)
    U = prep.fit_transform(X)
    prep.save(U, metadata, output_dir)


if __name__ == '__main__':
    main()
