"""
Create Spatial Danger Map from Environmental Hypervolume

Stage 5 of hypervolume pipeline - CORRECT METHODOLOGY:
- Load full landscape GeoTIFF (all environmental features)
- For each landscape pixel: extract features → engineer → z-score → PCA
- Check if environmental conditions fall inside CHE envelope
- Assign danger based on envelope membership/distance

This identifies areas with environmental conditions matching extreme fires,
regardless of whether fires occurred there in training data.

Usage:
    python scripts/create_danger_map.py \
        --u_space data/u_space \
        --che data/che \
        --data_root ~/data/deep_crown_dataset/organized_spreads \
        --output data/danger_map
"""

import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.transforms import LandscapeNormalize


class DangerMapper:
    """Create danger map by projecting landscape to U-space and checking envelope membership"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        
        # Load and normalize landscape using SAME transform as training
        landscape_path = self.data_root / 'landscape' / 'Input_Geotiff.tif'
        print(f"Loading landscape from {landscape_path}...")
        
        normalizer = LandscapeNormalize()
        landscape_xr = normalizer(str(landscape_path))
        
        # Convert to numpy [C, H, W]
        self.landscape = landscape_xr.values.astype(np.float32)
        self.landscape_shape = (landscape_xr.shape[1], landscape_xr.shape[2])  # (H, W)
        self.nodata = -1.0  # After normalization, -9999 becomes -1
        
        # Get raw landscape for visualization
        with rasterio.open(landscape_path) as src:
            self.landscape_raw = src.read().astype(np.float32)
            self.landscape_transform = src.transform
            self.landscape_crs = src.crs
        
        print(f"  Shape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
        print(f"  Bands: {self.landscape.shape[0]}")
        print(f"  Normalized: min={self.landscape.min():.2f}, max={self.landscape.max():.2f}")
        
        # Create valid data mask
        self.valid_mask = self.landscape[0] != self.nodata
        print(f"  Valid pixels: {self.valid_mask.sum():,} / {self.valid_mask.size:,} "
              f"({100*self.valid_mask.mean():.1f}%)")
    
    def load_transformation(self, u_space_dir):
        """Load PCA transformation metadata from prep_u_space stage"""
        u_space_dir = Path(u_space_dir)
        
        print(f"\nLoading PCA transformation from {u_space_dir}...")
        
        with open(u_space_dir / 'transform.json', 'r') as f:
            self.transform_meta = json.load(f)
        
        self.feature_names = self.transform_meta['feature_names']
        self.scaler_mean = np.array(self.transform_meta['scaler_mean'])
        self.scaler_std = np.array(self.transform_meta['scaler_std'])
        self.pca_components = np.array(self.transform_meta['pca_components'])
        self.n_components = self.transform_meta['n_components']
        
        print(f"  Features: {self.feature_names}")
        print(f"  PCA components: {self.n_components}")
        print(f"  Total variance: {self.transform_meta['total_variance']*100:.2f}%")
    
    def load_envelope(self, che_dir):
        """Load CHE envelope from stage 3"""
        che_dir = Path(che_dir)
        
        print(f"\nLoading CHE envelope from {che_dir}...")
        
        # Load envelope data
        envelope_data = np.load(che_dir / 'envelope.npz')
        self.occupancy_grid = envelope_data['occupancy_grid']
        self.envelope_mask = envelope_data['envelope_mask']
        self.envelope_bounds = envelope_data['bounds']
        
        # Load hull vertices for distance computation if available
        if 'hull_vertices' in envelope_data:
            self.hull_vertices = envelope_data['hull_vertices']
            print(f"  Loaded {len(self.hull_vertices):,} hull vertices for distance computation")
        else:
            self.hull_vertices = None
        
        # Load summary
        with open(che_dir / 'summary.json', 'r') as f:
            che_summary = json.load(f)
        
        print(f"  Method: {che_summary['method']}")
        if 'area' in che_summary:
            print(f"  Envelope area: {che_summary['area']:.2e}")
        print(f"  Grid resolution: {self.occupancy_grid.shape}")
    
    def engineer_pixel_features(self, pixel_features):
        """
        Apply same feature engineering as prep_u_space
        
        Args:
            pixel_features: dict with raw landscape bands
        
        Returns:
            feature vector matching self.feature_names order
        """
        features = []
        
        for feat_name in self.feature_names:
            if feat_name == 'forest':
                features.append(pixel_features['forest'])
            elif feat_name == 'arqueo':
                features.append(pixel_features['arqueo'])
            elif feat_name == 'cbd':
                features.append(pixel_features['cbd'])
            elif feat_name == 'cbh':
                features.append(pixel_features['cbh'])
            elif feat_name == 'elevation':
                features.append(pixel_features['elevation'])
            elif feat_name == 'flora':
                features.append(pixel_features['flora'])
            elif feat_name == 'paleo':
                features.append(pixel_features['paleo'])
            elif feat_name == 'urbana':
                features.append(pixel_features['urbana'])
            elif feat_name == 'wind_speed':
                features.append(pixel_features['wind_speed'])
            elif feat_name == 'wind_direction_cos':
                features.append(np.cos(pixel_features['wind_direction'] * np.pi / 180))
            elif feat_name == 'wind_direction_sin':
                features.append(np.sin(pixel_features['wind_direction'] * np.pi / 180))
        
        return np.array(features, dtype=np.float32)
    
    def project_to_uspace(self, X):
        """Project features to U-space using saved PCA with smart normalization"""
        X_scaled = np.zeros_like(X, dtype=np.float32)
        
        # Apply same normalization logic as training
        feature_stds = self.scaler_std
        feature_means = self.scaler_mean
        
        # Identify special features (same logic as prep_u_space)
        sparse_mask = feature_stds < 0.10
        binary_like_mask = (np.abs(feature_means) > 0.95) & (feature_stds < 0.2)
        special_features_mask = sparse_mask | binary_like_mask
        continuous_mask = ~special_features_mask
        
        # Z-score for continuous features
        if continuous_mask.any():
            X_scaled[continuous_mask] = (X[continuous_mask] - feature_means[continuous_mask]) / \
                                        (feature_stds[continuous_mask] + 1e-8)
        
        # Keep sparse/binary as-is
        if special_features_mask.any():
            X_scaled[special_features_mask] = X[special_features_mask]
        
        # PCA projection
        U = X_scaled @ self.pca_components.T
        
        return U
    
    def compute_distance_to_envelope(self, U):
        """
        Compute distance from U-space point to CHE envelope boundary
        
        Returns:
            distance: Normalized distance (0 = inside envelope, 1+ = far outside)
        """
        n_dims = self.envelope_bounds[0].shape[0]
        U_check = U[:n_dims]
        
        # Check if we have hull vertices for accurate distance
        if self.hull_vertices is not None and len(self.hull_vertices) > 0:
            from scipy.spatial.distance import cdist
            # Distance to nearest hull vertex
            distances = cdist([U_check], self.hull_vertices[:, :n_dims])
            min_dist = distances.min()
            
            # Check if inside envelope (using grid)
            inside = self._check_inside_grid(U_check, n_dims)
            
            if inside:
                # Inside envelope = 0 distance (most dangerous)
                return 0.0
            else:
                # Outside envelope = distance to boundary
                # Normalize by typical envelope radius for [0, 1+] range
                envelope_radius = np.linalg.norm(self.envelope_bounds[1] - self.envelope_bounds[0]) / 2
                normalized_dist = min_dist / envelope_radius
                return normalized_dist
        
        else:
            # Fallback: use grid occupancy as proxy
            inside, occupancy_dist = self._check_inside_grid(U_check, n_dims, return_occupancy=True)
            if inside:
                return 0.0
            else:
                # Use inverse occupancy as distance proxy
                return 1.0 + occupancy_dist
    
    def _check_inside_grid(self, U_check, n_dims, return_occupancy=False):
        """Helper to check if point is inside envelope using grid"""
        # Check if outside bounds
        for i in range(n_dims):
            if U_check[i] < self.envelope_bounds[0][i] or U_check[i] > self.envelope_bounds[1][i]:
                return (False, float('inf')) if return_occupancy else False
        
        # Map to grid indices
        if n_dims == 2:
            u1_min, u2_min = self.envelope_bounds[0]
            u1_max, u2_max = self.envelope_bounds[1]
            i = int((U_check[0] - u1_min) / (u1_max - u1_min) * (self.occupancy_grid.shape[1] - 1))
            j = int((U_check[1] - u2_min) / (u2_max - u2_min) * (self.occupancy_grid.shape[0] - 1))
            i = np.clip(i, 0, self.occupancy_grid.shape[1] - 1)
            j = np.clip(j, 0, self.occupancy_grid.shape[0] - 1)
            
            inside = bool(self.envelope_mask[j, i])
            occupancy = float(self.occupancy_grid[j, i])
            
        elif n_dims == 3:
            # 3D grid
            indices = []
            for dim in range(3):
                idx = int((U_check[dim] - self.envelope_bounds[0][dim]) / 
                         (self.envelope_bounds[1][dim] - self.envelope_bounds[0][dim]) * 
                         (self.occupancy_grid.shape[dim] - 1))
                idx = np.clip(idx, 0, self.occupancy_grid.shape[dim] - 1)
                indices.append(idx)
            
            inside = bool(self.envelope_mask[indices[1], indices[0], indices[2]])
            occupancy = float(self.occupancy_grid[indices[1], indices[0], indices[2]])
        else:
            raise ValueError(f"Only 2D and 3D envelopes supported, got {n_dims}D")
        
        if return_occupancy:
            return inside, 1.0 - occupancy
        else:
            return inside
    
    def create_danger_map(self, typical_wind_speed=5.0, typical_wind_dir=180.0, chunk_size=1000):
        """
        Create danger map over full landscape
        
        Args:
            typical_wind_speed: Typical summer wind speed (m/s, RAW) to use for all pixels
            typical_wind_dir: Typical wind direction (degrees) for circular encoding
            chunk_size: Process landscape in chunks (rows at a time)
        
        Returns:
            danger_grid: [H, W] array with danger scores
        """
        # Normalize wind speed using training ranges [0, 51] m/s → [0, 1]
        # (from WeatherNormalize.fit_transform)
        wind_speed_normalized = (typical_wind_speed - 0.0) / (51.0 - 0.0)
        
        print(f"\n{'='*60}")
        print(f"Creating Danger Map via U-Space Projection")
        print(f"{'='*60}")
        print(f"Landscape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
        print(f"Using typical wind: {typical_wind_speed} m/s (normalized: {wind_speed_normalized:.4f}) @ {typical_wind_dir}°")
        print(f"Processing in chunks of {chunk_size} rows...")
        
        H, W = self.landscape_shape
        danger_grid = np.full((H, W), np.nan, dtype=np.float32)
        
        # Band mapping (actual GeoTIFF structure)
        band_idx = {
            'forest': 0,
            'arqueo': 1,
            'cbd': 2,
            'cbh': 3,
            'elevation': 4,
            'flora': 5,
            'paleo': 6,
            'urbana': 7,
        }
        
        # Debug: track statistics
        u_values = []
        inside_count = 0
        sample_features = []
        
        # Store raw features for pixels inside danger zone
        danger_pixel_data = []
        
        # First pass: compute all distances
        distance_grid = np.full((H, W), np.nan, dtype=np.float32)
        
        print("  Pass 1: Computing distances to envelope...")
        for start_row in tqdm(range(0, H, chunk_size), desc="Computing distances"):
            end_row = min(start_row + chunk_size, H)
            
            for y in range(start_row, end_row):
                for x in range(W):
                    # Skip invalid pixels
                    if not self.valid_mask[y, x]:
                        continue
                    
                    # Extract environmental features
                    pixel_features = {
                        'forest': self.landscape[band_idx['forest'], y, x],
                        'arqueo': self.landscape[band_idx['arqueo'], y, x],
                        'cbd': self.landscape[band_idx['cbd'], y, x],
                        'cbh': self.landscape[band_idx['cbh'], y, x],
                        'elevation': self.landscape[band_idx['elevation'], y, x],
                        'flora': self.landscape[band_idx['flora'], y, x],
                        'paleo': self.landscape[band_idx['paleo'], y, x],
                        'urbana': self.landscape[band_idx['urbana'], y, x],
                        'wind_speed': wind_speed_normalized,
                        'wind_direction': typical_wind_dir,
                    }
                    
                    # Engineer features
                    X = self.engineer_pixel_features(pixel_features)
                    
                    # Project to U-space
                    U = self.project_to_uspace(X)
                    
                    # Debug: collect U values
                    if len(u_values) < 1000:
                        u_values.append(U[:2])
                    
                    # Compute distance to envelope
                    distance = self.compute_distance_to_envelope(U)
                    distance_grid[y, x] = distance
        
        # Normalize distances to [0, 1] based on actual range
        valid_distances = distance_grid[~np.isnan(distance_grid)]
        min_dist = valid_distances.min()
        max_dist = valid_distances.max()
        
        print(f"\n  Distance range: [{min_dist:.4f}, {max_dist:.4f}]")
        print(f"  Pass 2: Normalizing to danger scores...")
        
        # Second pass: normalize and collect danger pixels
        for y in range(H):
            for x in range(W):
                # Skip invalid pixels
                if not self.valid_mask[y, x] or np.isnan(distance_grid[y, x]):
                    continue
                
                distance = distance_grid[y, x]
                
                # Normalize to [0, 1]
                if max_dist > min_dist:
                    danger_score = (distance - min_dist) / (max_dist - min_dist)
                else:
                    danger_score = 0.0
                
                danger_grid[y, x] = danger_score
                
                # Track inside envelope pixels (distance == 0)
                if distance == 0.0:
                    inside_count += 1
                    
                    # Get pixel features again
                    pixel_features = {
                        'forest': self.landscape[band_idx['forest'], y, x],
                        'arqueo': self.landscape[band_idx['arqueo'], y, x],
                        'cbd': self.landscape[band_idx['cbd'], y, x],
                        'cbh': self.landscape[band_idx['cbh'], y, x],
                        'elevation': self.landscape[band_idx['elevation'], y, x],
                        'flora': self.landscape[band_idx['flora'], y, x],
                        'paleo': self.landscape[band_idx['paleo'], y, x],
                        'urbana': self.landscape[band_idx['urbana'], y, x],
                        'wind_speed': wind_speed_normalized,
                        'wind_direction': typical_wind_dir,
                    }
                    
                    # Capture for analysis
                    danger_record = {
                        'y': y,
                        'x': x,
                        'forest': pixel_features['forest'],
                        'arqueo': pixel_features['arqueo'],
                        'cbd': pixel_features['cbd'],
                        'cbh': pixel_features['cbh'],
                        'elevation': pixel_features['elevation'],
                        'flora': pixel_features['flora'],
                        'paleo': pixel_features['paleo'],
                        'urbana': pixel_features['urbana'],
                        'wind_speed': pixel_features['wind_speed'],
                        'wind_direction': pixel_features['wind_direction'],
                        'distance_to_envelope': distance,
                        'danger_score': danger_score,
                    }
                    danger_pixel_data.append(danger_record)
                
                # Debug: sample first 10 pixels
                if len(sample_features) < 10:
                    sample_features.append(pixel_features)
        
        # Debug output
        if len(u_values) > 0:
            u_values = np.array(u_values)
            print(f"\n  DEBUG: Sample U-space projections:")
            print(f"    U1 range: [{u_values[:, 0].min():.2f}, {u_values[:, 0].max():.2f}]")
            print(f"    U2 range: [{u_values[:, 1].min():.2f}, {u_values[:, 1].max():.2f}]")
            print(f"    Envelope U1 bounds: [{self.envelope_bounds[0][0]:.2f}, {self.envelope_bounds[1][0]:.2f}]")
            print(f"    Envelope U2 bounds: [{self.envelope_bounds[0][1]:.2f}, {self.envelope_bounds[1][1]:.2f}]")
        
        if len(sample_features) > 0:
            print(f"\n  DEBUG: Sample landscape features:")
            for i, pf in enumerate(sample_features[:3]):
                print(f"    Pixel {i}: elev={pf['elevation']:.1f}, forest={pf['forest']:.1f}, "
                      f"cbd={pf['cbd']:.3f}, cbh={pf['cbh']:.2f}")
        
        print(f"\n✓ Danger map created")
        valid_danger = danger_grid[~np.isnan(danger_grid)]
        print(f"  Valid pixels: {len(valid_danger):,}")
        print(f"  Danger range: [{valid_danger.min():.3f}, {valid_danger.max():.3f}]")
        print(f"  Mean danger: {valid_danger.mean():.3f}")
        print(f"  High danger pixels (< 0.7): {(valid_danger < 0.7).sum():,}")
        print(f"  Danger zone pixels captured: {len(danger_pixel_data):,}")
        print(f"{'='*60}\n")
        
        # Store danger pixel data for saving
        self.danger_pixel_data = danger_pixel_data
        
        return danger_grid
    
    def save_danger_features(self, output_dir):
        """Save raw environmental features for danger zone pixels"""
        output_dir = Path(output_dir)
        
        if not hasattr(self, 'danger_pixel_data') or len(self.danger_pixel_data) == 0:
            print("  No danger zone pixels to save")
            return
        
        print(f"Saving danger zone features...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.danger_pixel_data)
        
        # Save to parquet
        parquet_path = output_dir / 'danger_pixel_features.parquet'
        df.to_parquet(parquet_path, index=False, compression='snappy')
        
        print(f"  ✓ Saved {len(df):,} danger zone pixels")
        print(f"  Columns: {list(df.columns)}")
        print(f"  File: {parquet_path}")
    
    def save_results(self, danger_grid, output_dir):
        """Save danger map and visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        H, W = self.landscape_shape
        extent = [0, W, H, 0]  # [x_min, x_max, y_max, y_min] for origin='upper'
        
        # Save danger grid
        np.savez_compressed(
            output_dir / 'danger_grid.npz',
            danger=danger_grid,
            extent=extent
        )
        
        print(f"Creating visualizations...")
        
        # 1. Elevation-only map
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Use raw elevation for visualization
        elevation = self.landscape_raw[0].copy()
        elevation[elevation == -9999] = np.nan
        
        ax.imshow(
            elevation,
            cmap='terrain',
            extent=extent,
            origin='upper',
            interpolation='bilinear'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Elevation: Full Landscape',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Elevation (m)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'elevation_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Danger map with elevation background
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Show elevation background
        ax.imshow(
            elevation,
            cmap='terrain',
            alpha=0.4,
            extent=extent,
            origin='upper',
            interpolation='bilinear'
        )
        
        # Overlay danger map
        # Custom colormap: black (0.0 = max danger) → red → yellow → white (1.0 = safe)
        colors = ['#000000', '#8B0000', '#FF4500', '#FFA500', '#FFFF00', '#FFFFFF']
        cmap_danger = LinearSegmentedColormap.from_list('danger', colors, N=256)
        
        danger_masked = np.ma.masked_invalid(danger_grid)
        
        # Use actual data range
        vmin = 0.0
        vmax = max(danger_masked.max(), 0.5)  # At least 0.5 for scale
        
        im = ax.imshow(
            danger_masked,
            extent=extent,
            origin='upper',
            cmap=cmap_danger,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Fire Danger Map: Distance to Extreme Fire Hypervolume\n'
                    'Darker = Closer to extreme fire conditions',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label(f'Danger Score\n(0.0=Extreme, {vmax:.1f}=Safer)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'danger_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Danger-only map (no background)
        fig, ax = plt.subplots(figsize=(18, 14))
        
        im = ax.imshow(
            danger_masked,
            extent=extent,
            origin='upper',
            cmap=cmap_danger,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Fire Danger Map (Distance to Extreme Fire Envelope)',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label(f'Danger Score\n(0.0=Extreme, {vmax:.1f}=Safer)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'danger_map_only.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save danger zone features
        self.save_danger_features(output_dir)
        
        print(f"\n{'='*60}")
        print(f"✓ Saved Results")
        print(f"{'='*60}")
        print(f"  {output_dir}/danger_grid.npz")
        print(f"  {output_dir}/elevation_map.png")
        print(f"  {output_dir}/danger_map.png")
        print(f"  {output_dir}/danger_map_only.png")
        print(f"  {output_dir}/danger_pixel_features.parquet")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Create Danger Map via Hypervolume Projection')
    parser.add_argument('--u_space', type=str, required=True,
                       help='Directory with U-space transformation data')
    parser.add_argument('--che', type=str, required=True,
                       help='Directory with CHE envelope')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset (for landscape GeoTIFF)')
    parser.add_argument('--output', type=str, default='data/danger_map',
                       help='Output directory for danger map')
    parser.add_argument('--wind_speed', type=float, default=5.0,
                       help='Typical wind speed (m/s) for projection')
    parser.add_argument('--wind_direction', type=float, default=180.0,
                       help='Typical wind direction (degrees) for projection')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Process landscape in chunks (rows)')
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = DangerMapper(args.data_root)
    
    # Load PCA transformation
    mapper.load_transformation(args.u_space)
    
    # Load CHE envelope
    mapper.load_envelope(args.che)
    
    # Create danger map
    danger_grid = mapper.create_danger_map(
        typical_wind_speed=args.wind_speed,
        typical_wind_dir=args.wind_direction,
        chunk_size=args.chunk_size
    )
    
    # Save results
    mapper.save_results(danger_grid, args.output)
    
    print(f"\n{'='*60}")
    print(f"✓ Danger Map Creation Complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
