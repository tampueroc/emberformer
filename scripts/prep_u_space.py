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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


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
        
        # Static features (keep as-is)
        for feat in ['forest', 'arqueo', 'cbd', 'cbh', 'elevation', 'flora', 'paleo', 'urbana']:
            if feat in data:
                features[feat] = np.array(data[feat], dtype=np.float32)
                feature_list.append(feat)
                print(f"  ✓ {feat}: {features[feat].shape[0]:,} values")
        
        # Wind: polar coordinates with circular encoding
        if 'wind_speed' in data and 'wind_direction' in data:
            wind_speed = np.array(data['wind_speed'], dtype=np.float32)
            wind_dir_rad = np.array(data['wind_direction'], dtype=np.float32) * np.pi / 180
            
            features['wind_speed'] = wind_speed
            features['wind_direction_cos'] = np.cos(wind_dir_rad)
            features['wind_direction_sin'] = np.sin(wind_dir_rad)
            feature_list.extend(['wind_speed', 'wind_direction_cos', 'wind_direction_sin'])
            print(f"  ✓ wind_speed, wind_direction → wind_speed, wind_direction_cos, wind_direction_sin")
        
        print(f"\nTotal features: {len(feature_list)}")
        print(f"{'='*60}\n")
        
        # Stack into matrix
        X = np.column_stack([features[f] for f in feature_list])
        
        # Keep metadata
        metadata = {
            'gradcam': np.array(data['gradcam'], dtype=np.float32),
            'fire_intensity': np.array(data['fire_intensity'], dtype=np.float32),
            'y': np.array(data['y'], dtype=np.int32),
            'x': np.array(data['x'], dtype=np.int32),
        }
        
        self.feature_names = feature_list
        return X, metadata
    
    def fit_transform(self, X):
        """Smart normalization + PCA: handle binary/sparse features separately"""
        print(f"\n{'='*60}")
        print(f"PCA Transformation with Smart Normalization")
        print(f"{'='*60}")
        print(f"Input shape: {X.shape}")
        
        # Fit scaler to get statistics
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        # Identify sparse/binary features (low variance or mostly nodata)
        feature_stds = self.scaler.scale_
        feature_means = self.scaler.mean_
        
        # Features with std < threshold are sparse (mostly nodata or constant)
        sparse_mask = feature_stds < self.min_feature_std
        
        # Also check for binary-like: mean near -1 or 1 AND low std
        binary_like_mask = (np.abs(feature_means) > 0.95) & (feature_stds < 0.2)
        
        # Combine: features that are either sparse or binary-like
        special_features_mask = sparse_mask | binary_like_mask
        continuous_mask = ~special_features_mask
        
        print(f"\n  Feature categorization:")
        print(f"    Continuous features: {continuous_mask.sum()}")
        print(f"    Sparse/binary features: {special_features_mask.sum()}")
        
        # Print special features that will be handled differently
        if special_features_mask.any():
            print(f"\n  Special handling (no z-score):")
            for name, mean, std in zip(np.array(self.feature_names)[special_features_mask],
                                      feature_means[special_features_mask],
                                      feature_stds[special_features_mask]):
                print(f"    • {name}: mean={mean:.3f}, std={std:.4f}")
        
        # Apply normalization
        X_scaled = np.zeros_like(X, dtype=np.float32)
        
        # Z-score for continuous features
        if continuous_mask.any():
            X_scaled[:, continuous_mask] = (X[:, continuous_mask] - feature_means[continuous_mask]) / \
                                           (feature_stds[continuous_mask] + 1e-8)
            print(f"\n  ✓ Z-score normalized {continuous_mask.sum()} continuous features")
        
        # Keep sparse/binary features as-is (already normalized to [-1, 1])
        if special_features_mask.any():
            X_scaled[:, special_features_mask] = X[:, special_features_mask]
            print(f"  ✓ Preserved {special_features_mask.sum()} sparse/binary features (no z-score)")
        
        print(f"\n  All features retained: {X_scaled.shape[1]}")
        
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
    prep.save(U, metadata, args.output)


if __name__ == '__main__':
    main()
