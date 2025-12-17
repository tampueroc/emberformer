"""
Prepare U-Space from Salience Data (Direct Z-score, no dimensionality reduction)

Stage 2 of hypervolume pipeline:
- Load Parquet salience data
- Filter for extreme fires (top percentile)
- Feature engineering
- Z-score normalization only (no PCA/UMAP)

This avoids the out-of-sample projection issues with UMAP.

Usage:
    python scripts/pipeline/prep_u_space_direct.py \
        --input data/salience \
        --output data/u_space_direct \
        --extreme_threshold 99
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


class USpacePrepDirect:
    """Prepare U-space from salience data using only z-score normalization"""
    
    def __init__(self, min_feature_std=0.01):
        self.min_feature_std = min_feature_std
        self.scaler = None
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
    
    def engineer_features(self, data, include_wind=False):
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
        
        # Wind features - optional, circular encoding for direction
        if include_wind and 'wind_speed' in data and 'wind_direction' in data:
            wind_speed = np.array(data['wind_speed'], dtype=np.float32)
            wind_dir = np.array(data['wind_direction'], dtype=np.float32)
            
            wind_dir_rad = wind_dir * 2 * np.pi
            features['wind_speed'] = wind_speed
            features['wind_dir_sin'] = np.sin(wind_dir_rad).astype(np.float32)
            features['wind_dir_cos'] = np.cos(wind_dir_rad).astype(np.float32)
            feature_list.extend(['wind_speed', 'wind_dir_sin', 'wind_dir_cos'])
            print(f"  ✓ wind_speed: {wind_speed.shape[0]:,} values")
            print(f"  ✓ wind_dir_sin/cos: circular encoding")
        elif 'wind_speed' in data:
            print(f"  ⊗ wind: excluded (use --include_wind to enable)")
        
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
        """Z-score normalization only (no dimensionality reduction)"""
        print(f"\n{'='*60}")
        print(f"Z-Score Normalization (Direct, no DR)")
        print(f"{'='*60}")
        print(f"Input shape: {X.shape}")

        self.scaler = StandardScaler()
        U = self.scaler.fit_transform(X)
        print(f"  ✓ Z-score normalized (mean=0, std=1)")
        print(f"  Output shape: {U.shape}")
        print(f"{'='*60}\n")
        
        return U.astype(np.float32)
    
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
            'method': 'direct',
            'feature_names': self.feature_names,
            'n_components': U.shape[1],
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist(),
            'n_samples': U.shape[0],
        }
        
        with open(output_dir / 'transform.json', 'w') as f:
            json.dump(transform_meta, f, indent=2)
        
        # Save fitted scaler
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"{'='*60}")
        print(f"✓ Saved U-space data (Direct Z-score)")
        print(f"{'='*60}")
        print(f"  {output_dir}/extreme.npz ({U.shape[0]:,} points × {U.shape[1]} dims)")
        print(f"  {output_dir}/transform.json (metadata)")
        print(f"  {output_dir}/scaler.pkl (fitted scaler)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare U-Space (Direct Z-score)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with Parquet salience files')
    parser.add_argument('--output', type=str, default='data/u_space_direct',
                       help='Output directory for U-space data')
    parser.add_argument('--extreme_threshold', type=int, default=99,
                       help='Percentile threshold for extreme fires (99 = top 1%%)')
    parser.add_argument('--max_components', type=int, default=5,
                       help='Unused (kept for CLI compatibility)')
    parser.add_argument('--variance_threshold', type=float, default=0.80,
                       help='Unused (kept for CLI compatibility)')
    parser.add_argument('--min_feature_std', type=float, default=0.10,
                       help='Minimum std for feature to be kept')
    parser.add_argument('--include_wind', action='store_true',
                       help='Include wind speed and direction (circular encoding)')
    
    args = parser.parse_args()
    
    commit_hash = get_git_commit_hash()
    output_dir = Path(args.output) / commit_hash
    print(f"Git commit: {commit_hash}")
    print(f"Output directory: {output_dir}\n")
    
    prep = USpacePrepDirect(
        min_feature_std=args.min_feature_std
    )
    
    data = prep.load_extreme_fires(args.input, args.extreme_threshold)
    X, metadata = prep.engineer_features(data, include_wind=args.include_wind)
    U = prep.fit_transform(X)
    prep.save(U, metadata, output_dir)


if __name__ == '__main__':
    main()
