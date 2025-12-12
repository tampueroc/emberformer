"""
Prepare U-Space from Salience Data

Stage 2 of hypervolume pipeline:
- Load Parquet salience data
- Filter for extreme fires (top percentile)
- Feature engineering: circular encodings for aspect/wind_direction
- Z-score normalization
- PCA projection to U-space (≤5 components, ≥80% variance)

Usage:
    python scripts/prep_u_space.py \
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


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


class USpacePrep:
    """Prepare U-space from salience data with feature engineering"""
    
    def __init__(self, max_components=5, variance_threshold=0.80, min_feature_std=0.01):
        self.max_components = max_components
        self.variance_threshold = variance_threshold
        self.min_feature_std = min_feature_std
        self.scaler = None
        self.pca = None
        self.feature_names = None
        
    def load_extreme_fires(self, salience_dir, extreme_threshold=99):
        """Load only extreme fire pixels from Parquet files"""
        salience_dir = Path(salience_dir)
        
        # Load quantiles
        with open(salience_dir / 'quantiles.json', 'r') as f:
            quantiles = json.load(f)
        
        # Determine threshold
        threshold_val = quantiles[f'p{extreme_threshold}']
        print(f"\n{'='*60}")
        print(f"Loading Extreme Fires (>{extreme_threshold}th percentile)")
        print(f"{'='*60}")
        print(f"Fire intensity threshold: {threshold_val:.4f}")
        
        # Find all parquet files
        parquet_files = sorted(salience_dir.glob('part-*.parquet'))
        print(f"Found {len(parquet_files)} parquet files")
        
        # Load and filter
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
        
        # Combine chunks
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
        
        # Only use features with meaningful variance (not nodata-dominated)
        # Excluded: arqueo, flora, paleo, urbana (constant ~1 or dominated by nodata=-1)
        valid_features = ['forest', 'cbd', 'cbh', 'elevation']
        excluded_features = ['arqueo', 'flora', 'paleo', 'urbana']
        
        for feat in valid_features:
            if feat in data:
                features[feat] = np.array(data[feat], dtype=np.float32)
                feature_list.append(feat)
                print(f"  ✓ {feat}: {features[feat].shape[0]:,} values")
        
        # Log excluded features
        for feat in excluded_features:
            if feat in data:
                print(f"  ⊗ {feat}: excluded (nodata-dominated, constant in landscape)")
        
        # SKIP WIND: It's constant across extreme fires (~0.6° at 9.7 m/s)
        # Wind is temporal, not a landscape determinant
        if 'wind_speed' in data and 'wind_direction' in data:
            print(f"  ⊗ wind: excluded (temporal, not landscape-based)")
        
        print(f"\nTotal features: {len(feature_list)}")
        print(f"{'='*60}\n")
        
        # Stack into matrix
        X = np.column_stack([features[f] for f in feature_list])
        
        # Keep metadata - support both old (y, x) and new (y_landscape, x_landscape) schemas
        metadata = {
            'gradcam': np.array(data['gradcam'], dtype=np.float32),
            'fire_intensity': np.array(data['fire_intensity'], dtype=np.float32),
        }
        
        # Use landscape coordinates if available, else fall back to cam coordinates
        if 'y_landscape' in data:
            metadata['y'] = np.array(data['y_landscape'], dtype=np.int32)
            metadata['x'] = np.array(data['x_landscape'], dtype=np.int32)
        else:
            metadata['y'] = np.array(data['y'], dtype=np.int32)
            metadata['x'] = np.array(data['x'], dtype=np.int32)
        
        self.feature_names = feature_list
        return X, metadata
    
    def fit_transform(self, X):
        """Simple Z-score normalization + PCA (matching 13a08e2 methodology)"""
        print(f"\n{'='*60}")
        print(f"PCA Transformation")
        print(f"{'='*60}")
        print(f"Input shape: {X.shape}")

        # Simple Z-score normalization for ALL features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        print(f"  ✓ Z-score normalized (mean=0, std=1)")
        
        # PCA
        n_components = min(self.max_components, X_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        U = self.pca.fit_transform(X_scaled)
        
        # Find components explaining variance_threshold
        cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_keep = np.searchsorted(cumsum_var, self.variance_threshold) + 1
        n_keep = max(2, min(n_keep, n_components))  # Keep at least 2, at most max_components
        
        U = U[:, :n_keep]
        
        print(f"\n  ✓ PCA complete:")
        print(f"    Components: {n_keep} (variance ≥ {self.variance_threshold*100:.0f}%)")
        print(f"    Total variance explained: {cumsum_var[n_keep-1]*100:.2f}%")
        print(f"    Output shape: {U.shape}")
        
        # Print variance per component
        print(f"\n  Variance by component:")
        for i in range(n_keep):
            print(f"    U{i+1}: {self.pca.explained_variance_ratio_[i]*100:.2f}% "
                  f"(cumulative: {cumsum_var[i]*100:.2f}%)")
        
        print(f"{'='*60}\n")
        
        return U[:, :n_keep]
    
    def save(self, U, metadata, output_dir):
        """Save U-space data and transformation metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save U-space data
        np.savez_compressed(
            output_dir / 'extreme.npz',
            U=U.astype(np.float32),
            gradcam=metadata['gradcam'],
            fire_intensity=metadata['fire_intensity'],
            y=metadata['y'],
            x=metadata['x'],
        )
        
        # Save transformation metadata
        transform_meta = {
            'feature_names': self.feature_names,
            'n_components': U.shape[1],
            'variance_explained': self.pca.explained_variance_ratio_[:U.shape[1]].tolist(),
            'total_variance': float(np.sum(self.pca.explained_variance_ratio_[:U.shape[1]])),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist(),
            'pca_components': self.pca.components_[:U.shape[1]].tolist(),
            'n_samples': U.shape[0],
        }
        
        with open(output_dir / 'transform.json', 'w') as f:
            json.dump(transform_meta, f, indent=2)
        
        print(f"{'='*60}")
        print(f"✓ Saved U-space data")
        print(f"{'='*60}")
        print(f"  {output_dir}/extreme.npz ({U.shape[0]:,} points × {U.shape[1]} dims)")
        print(f"  {output_dir}/transform.json (PCA metadata)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare U-Space from Salience Data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with Parquet salience files')
    parser.add_argument('--output', type=str, default='data/u_space',
                       help='Output directory for U-space data')
    parser.add_argument('--extreme_threshold', type=int, default=99,
                       help='Percentile threshold for extreme fires (99 = top 1%)')
    parser.add_argument('--max_components', type=int, default=5,
                       help='Maximum number of PCA components to keep')
    parser.add_argument('--variance_threshold', type=float, default=0.80,
                       help='Minimum cumulative variance to retain (0.80 = 80%%)')
    parser.add_argument('--min_feature_std', type=float, default=0.10,
                       help='Minimum std for feature to be kept (default=0.10, filters sparse/constant features)')
    
    args = parser.parse_args()
    
    # Append git commit hash to output directory
    commit_hash = get_git_commit_hash()
    output_dir = Path(args.output) / commit_hash
    print(f"Git commit: {commit_hash}")
    print(f"Output directory: {output_dir}\n")
    
    # Initialize
    prep = USpacePrep(
        max_components=args.max_components,
        variance_threshold=args.variance_threshold,
        min_feature_std=args.min_feature_std
    )
    
    # Load extreme fires
    data = prep.load_extreme_fires(args.input, args.extreme_threshold)
    
    # Feature engineering
    X, metadata = prep.engineer_features(data)
    
    # Transform to U-space
    U = prep.fit_transform(X)
    
    # Save
    prep.save(U, metadata, output_dir)


if __name__ == '__main__':
    main()
