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
from tqdm import tqdm

# Forest fuel type codes from spain_lookup_table.csv (38 categories)
# These are the raw integer codes before normalization
FOREST_CODES = np.array([
    0, 91, 92, 93, 98, 99,  # Non-fuel (6)
    101, 102, 103, 104, 105, 106, 107, 108,  # GR1-GR8 (8)
    121, 122, 123, 124,  # GS1-GS4 (4)
    142, 143, 144, 145, 146, 147, 148, 149,  # SH2-SH9 (8)
    161, 162, 163, 164, 165,  # TU1-TU5 (5)
    181, 182, 183, 185, 186, 188, 189,  # TL1-TL9 (7, some missing)
], dtype=np.int32)
N_FOREST_CATEGORIES = len(FOREST_CODES)  # 38

# Raw feature ranges for denormalization (from landscape GeoTIFF)
# Normalization was: (value - min) / (max - min)
# Denormalization is: value * (max - min) + min
FEATURE_RANGES = {
    'forest': (0.0, 189.0),
    'cbd': (0.0, 0.4467),
    'cbh': (0.0, 13.8692),
    'elevation': (345.9422, 3012.5251),
}

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


def denormalize(normalized_values, feat_name):
    """
    Convert normalized [0,1] values back to raw scale.
    
    The normalization was: (value - min) / (max - min)
    So inverse is: value * (max - min) + min
    """
    fmin, fmax = FEATURE_RANGES[feat_name]
    return normalized_values * (fmax - fmin) + fmin


def normalized_to_forest_code(normalized_values, forest_min=0, forest_max=189):
    """
    Convert normalized [0,1] forest values back to integer fuel type codes.
    """
    raw = normalized_values * (forest_max - forest_min) + forest_min
    return np.round(raw).astype(np.int32)


def one_hot_encode_forest(forest_codes):
    """
    One-hot encode forest fuel type codes.
    
    Args:
        forest_codes: array of integer fuel type codes (e.g., 0, 91, 101, ...)
    
    Returns:
        one_hot: (N, 39) array with binary indicators
        feature_names: list of feature names like 'forest_0', 'forest_91', ...
    """
    n_samples = len(forest_codes)
    one_hot = np.zeros((n_samples, N_FOREST_CATEGORIES), dtype=np.float32)
    
    # Build code to index mapping
    code_to_idx = {code: idx for idx, code in enumerate(FOREST_CODES)}
    
    for i, code in enumerate(forest_codes):
        if code in code_to_idx:
            one_hot[i, code_to_idx[code]] = 1.0
        else:
            # Unknown code - leave as all zeros (could also use nearest)
            pass
    
    feature_names = [f'forest_{code}' for code in FOREST_CODES]
    return one_hot, feature_names


class USpacePrepUMAP:
    """Prepare U-space from salience data using UMAP"""

    def __init__(self, max_components=3, n_neighbors=15, min_dist=0.1,
                 metric='euclidean', random_state=42, min_feature_std=0.01,
                 batch_size=10000):
        self.max_components = max_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.min_feature_std = min_feature_std
        self.batch_size = batch_size
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

    def engineer_features(self, data, include_wind=True):
        """Apply feature engineering transformations with one-hot encoded forest"""
        print(f"\n{'='*60}")
        print(f"Feature Engineering (One-Hot Forest)")
        print(f"{'='*60}")

        feature_arrays = []
        feature_list = []

        # One-hot encode forest (categorical with 39 classes)
        if 'forest' in data:
            forest_normalized = np.array(data['forest'], dtype=np.float32)
            forest_codes = normalized_to_forest_code(forest_normalized)
            forest_onehot, forest_names = one_hot_encode_forest(forest_codes)
            feature_arrays.append(forest_onehot)
            feature_list.extend(forest_names)
            
            # Show distribution of categories
            unique_codes, counts = np.unique(forest_codes, return_counts=True)
            print(f"  ✓ forest: one-hot encoded ({N_FOREST_CATEGORIES} categories)")
            print(f"    Active categories: {len(unique_codes)}")
            top_5 = sorted(zip(unique_codes, counts), key=lambda x: -x[1])[:5]
            print(f"    Top 5: {[(c, n) for c, n in top_5]}")

        # Continuous features (cbd, cbh, elevation) - DENORMALIZED to raw scale
        continuous_features = ['cbd', 'cbh', 'elevation']
        excluded_features = ['arqueo', 'flora', 'paleo', 'urbana']

        for feat in continuous_features:
            if feat in data:
                feat_normalized = np.array(data[feat], dtype=np.float32)
                feat_raw = denormalize(feat_normalized, feat)
                feature_arrays.append(feat_raw.reshape(-1, 1))
                feature_list.append(feat)
                fmin, fmax = FEATURE_RANGES[feat]
                print(f"  ✓ {feat}: {feat_raw.shape[0]:,} values (raw: {fmin:.2f}-{fmax:.2f})")

        for feat in excluded_features:
            if feat in data:
                print(f"  ⊗ {feat}: excluded (nodata-dominated, constant in landscape)")

        # Wind features - optional, circular encoding for direction
        if include_wind and 'wind_speed' in data and 'wind_direction' in data:
            wind_speed = np.array(data['wind_speed'], dtype=np.float32).reshape(-1, 1)
            wind_dir = np.array(data['wind_direction'], dtype=np.float32)

            # Circular encoding for wind direction (already normalized 0-1, treat as fraction of 2π)
            wind_dir_rad = wind_dir * 2 * np.pi
            wind_dir_sin = np.sin(wind_dir_rad).astype(np.float32).reshape(-1, 1)
            wind_dir_cos = np.cos(wind_dir_rad).astype(np.float32).reshape(-1, 1)
            
            feature_arrays.extend([wind_speed, wind_dir_sin, wind_dir_cos])
            feature_list.extend(['wind_speed', 'wind_dir_sin', 'wind_dir_cos'])
            print(f"  ✓ wind_speed: {wind_speed.shape[0]:,} values")
            print(f"  ✓ wind_dir_sin/cos: circular encoding")
        elif 'wind_speed' in data:
            print(f"  ⊗ wind: excluded (use --include_wind to enable)")

        print(f"\nTotal features: {len(feature_list)}")
        print(f"  (39 forest one-hot + 3 continuous = 42 base features)")
        print(f"{'='*60}\n")

        X = np.hstack(feature_arrays)

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
        """UMAP projection with batch transform for memory efficiency"""
        print(f"\n{'='*60}")
        print(f"UMAP Transformation (batch processing)")
        print(f"{'='*60}")
        print(f"Input shape: {X.shape}")
        print(f"Batch size: {self.batch_size:,}")
        print(f"Feature ranges:")
        for i, name in enumerate(self.feature_names):
            print(f"    {name}: [{X[:,i].min():.3f}, {X[:,i].max():.3f}]")

        n_components = min(self.max_components, X.shape[1])
        n_samples = X.shape[0]

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
        else:
            self.umap = UMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                verbose=True
            )

        if UMAP_BACKEND == "gpu":
            # GPU: use fit_transform directly (more stable with cuML)
            print(f"\n  GPU fit_transform on {n_samples:,} samples...")
            U = self.umap.fit_transform(X)
        else:
            # CPU: fit then batch transform for memory efficiency
            print(f"\n  Fitting UMAP on {n_samples:,} samples...")
            self.umap.fit(X)

            print(f"  Transforming in batches of {self.batch_size:,}...")
            U = np.zeros((n_samples, n_components), dtype=np.float32)
            
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            for i in tqdm(range(n_batches), desc="Batch transform"):
                start = i * self.batch_size
                end = min(start + self.batch_size, n_samples)
                U[start:end] = self.umap.transform(X[start:end])

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
            'n_samples': U.shape[0],
        }

        with open(output_dir / 'transform.json', 'w') as f:
            json.dump(transform_meta, f, indent=2)

        # Save fitted UMAP model for transform() on new data
        with open(output_dir / 'umap_model.pkl', 'wb') as f:
            pickle.dump(self.umap, f)

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
    parser.add_argument('--metric', type=str, default='euclidean',
                       help='Distance metric for UMAP (default: euclidean)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for UMAP transform (default: 10000)')
    parser.add_argument('--include_wind', action='store_true',
                       help='Include wind speed and direction (circular encoding)')

    args = parser.parse_args()

    commit_hash = get_git_commit_hash()
    output_dir = Path(args.output) / commit_hash
    print(f"Git commit: {commit_hash}")
    print(f"Output directory: {output_dir}\n")

    prep = USpacePrepUMAP(
        max_components=args.max_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        min_feature_std=args.min_feature_std,
        batch_size=args.batch_size
    )

    data = prep.load_extreme_fires(args.input, args.extreme_threshold)
    X, metadata = prep.engineer_features(data, include_wind=args.include_wind)
    U = prep.fit_transform(X)
    prep.save(U, metadata, output_dir)


if __name__ == '__main__':
    main()
