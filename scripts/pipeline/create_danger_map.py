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
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import os

# Add project root to path for imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from data.transforms import LandscapeNormalize


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
        
        print(f"  Normalized shape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
        print(f"  Raw shape: {self.landscape_raw.shape[1]} × {self.landscape_raw.shape[2]} pixels")
        print(f"  Shapes match: {self.landscape.shape == self.landscape_raw.shape}")
        print(f"  Bands: {self.landscape.shape[0]}")
        print(f"  Normalized: min={self.landscape.min():.2f}, max={self.landscape.max():.2f}")
        
        # Create valid data mask from RAW landscape (source of truth)
        # This ensures we only process pixels that exist in the elevation map
        self.valid_mask = self.landscape_raw[0] != -9999
        
        print(f"  Valid pixels (from raw elevation): {self.valid_mask.sum():,} / {self.valid_mask.size:,} "
              f"({100*self.valid_mask.mean():.1f}%)")
        
        # Verify normalized landscape has same valid region
        normalized_valid = self.landscape[0] != self.nodata
        if not np.array_equal(self.valid_mask, normalized_valid):
            print(f"  WARNING: Normalized and raw valid masks differ!")
            print(f"    Normalized has {normalized_valid.sum():,} valid pixels")
            print(f"    Using raw mask as source of truth")
    
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
            # Wind features removed - landscape determinants only
        
        return np.array(features, dtype=np.float32)
    
    def project_to_uspace(self, X):
        """Project features to U-space using saved PCA (simple z-score)"""
        # Simple Z-score normalize (matching training)
        X_scaled = (X - self.scaler_mean) / (self.scaler_std + 1e-8)

        # PCA projection
        U = X_scaled @ self.pca_components.T

        return U
    
    def compute_distance_to_envelope(self, U):
        """
        Compute distance from U-space point to CHE envelope boundary
        
        Returns:
            distance: 0 = high occupancy (dangerous), 1 = low occupancy (safe)
        """
        n_dims = self.envelope_bounds[0].shape[0]
        U_check = U[:n_dims]
        
        # Get occupancy at this point (how many bootstrap hulls contain it)
        occupancy = self._get_occupancy(U_check, n_dims)
        
        # Return inverse of occupancy: high occupancy = low distance (dangerous)
        # occupancy 1.0 -> distance 0.0 (most dangerous)
        # occupancy 0.0 -> distance 1.0 (safest)
        return 1.0 - occupancy
    
    def _get_occupancy(self, U_check, n_dims):
        """Get occupancy value at U-space point"""
        # Check if outside bounds
        for i in range(n_dims):
            if U_check[i] < self.envelope_bounds[0][i] or U_check[i] > self.envelope_bounds[1][i]:
                return 0.0  # Outside bounds = 0 occupancy
        
        # Map to grid indices
        if n_dims == 2:
            u1_min, u2_min = self.envelope_bounds[0]
            u1_max, u2_max = self.envelope_bounds[1]
            i = int((U_check[0] - u1_min) / (u1_max - u1_min) * (self.occupancy_grid.shape[1] - 1))
            j = int((U_check[1] - u2_min) / (u2_max - u2_min) * (self.occupancy_grid.shape[0] - 1))
            i = np.clip(i, 0, self.occupancy_grid.shape[1] - 1)
            j = np.clip(j, 0, self.occupancy_grid.shape[0] - 1)
            return float(self.occupancy_grid[j, i])
        elif n_dims == 3:
            # 3D grid indexing
            indices = []
            for dim in range(3):
                idx = int((U_check[dim] - self.envelope_bounds[0][dim]) / 
                         (self.envelope_bounds[1][dim] - self.envelope_bounds[0][dim]) * 
                         (self.occupancy_grid.shape[dim] - 1))
                idx = np.clip(idx, 0, self.occupancy_grid.shape[dim] - 1)
                indices.append(idx)
            return float(self.occupancy_grid[indices[0], indices[1], indices[2]])
        else:
            raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")
    
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
    
    def create_danger_map(self, chunk_size=1000):
        """
        Create danger map over full landscape
        
        Args:
            chunk_size: Process landscape in chunks (rows at a time)
        
        Returns:
            danger_grid: [H, W] array with danger scores
        """
        print(f"\n{'='*60}")
        print(f"Creating Danger Map via U-Space Projection")
        print(f"{'='*60}")
        print(f"Landscape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
        print(f"NOTE: Wind features excluded (landscape determinants only)")
        
        H, W = self.landscape_shape
        danger_grid = np.full((H, W), np.nan, dtype=np.float32)
        
        # Verify we only assign to valid pixels
        print(f"  Initial danger_grid: all NaN = {np.all(np.isnan(danger_grid))}")
        
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
        
        # Store raw features for pixels inside danger zone
        danger_pixel_data = []
        distances_list = []
        
        # Process only valid pixels (where elevation is valid)
        print(f"  Processing {self.valid_mask.sum():,} valid pixels...")
        
        y_coords, x_coords = np.where(self.valid_mask)
        for idx in tqdm(range(len(y_coords)), desc="Computing danger scores"):
            y, x = y_coords[idx], x_coords[idx]
            
            # Extract environmental features for this pixel (LANDSCAPE ONLY, no wind)
            pixel_features = {
                'forest': self.landscape[band_idx['forest'], y, x],
                'arqueo': self.landscape[band_idx['arqueo'], y, x],
                'cbd': self.landscape[band_idx['cbd'], y, x],
                'cbh': self.landscape[band_idx['cbh'], y, x],
                'elevation': self.landscape[band_idx['elevation'], y, x],
                'flora': self.landscape[band_idx['flora'], y, x],
                'paleo': self.landscape[band_idx['paleo'], y, x],
                'urbana': self.landscape[band_idx['urbana'], y, x],
            }
            
            # Engineer features
            X = self.engineer_pixel_features(pixel_features)
            
            # Project to U-space
            U = self.project_to_uspace(X)
            
            # Compute distance to envelope (0 = high occupancy/dangerous, 1 = low/safe)
            distance = self.compute_distance_to_envelope(U)
            distances_list.append(distance)
            
            # Danger score = distance (already 0-1 scale)
            # 0 = inside high-occupancy region (most dangerous)
            # 1 = outside envelope (safest)
            danger_score = distance
            
            danger_grid[y, x] = danger_score
            
            # Save high-danger pixels (low distance = high occupancy)
            if distance < 0.3:
                danger_record = {
                    'y': int(y),
                    'x': int(x),
                    'forest': float(pixel_features['forest']),
                    'arqueo': float(pixel_features['arqueo']),
                    'cbd': float(pixel_features['cbd']),
                    'cbh': float(pixel_features['cbh']),
                    'elevation': float(pixel_features['elevation']),
                    'flora': float(pixel_features['flora']),
                    'paleo': float(pixel_features['paleo']),
                    'urbana': float(pixel_features['urbana']),
                    'distance_to_envelope': float(distance),
                    'danger_score': float(danger_score),
                }
                danger_pixel_data.append(danger_record)
        
        # Statistics
        distances_arr = np.array(distances_list)
        print(f"\n  Distance statistics:")
        print(f"    Min: {distances_arr.min():.4f}")
        print(f"    Max: {distances_arr.max():.4f}")
        print(f"    Mean: {distances_arr.mean():.4f}")
        print(f"    Median: {np.median(distances_arr):.4f}")
        print(f"    Inside envelope (dist=0): {(distances_arr == 0.0).sum():,} ({(distances_arr == 0.0).sum()/len(distances_arr)*100:.1f}%)")
        
        print(f"\n✓ Danger map created")
        print(f"  NaN pixels in danger_grid: {np.isnan(danger_grid).sum():,} / {danger_grid.size:,}")
        print(f"  Valid pixels match mask: {(~np.isnan(danger_grid) == self.valid_mask).all()}")
        
        valid_danger = danger_grid[~np.isnan(danger_grid)]
        print(f"  Valid pixels: {len(valid_danger):,}")
        print(f"  Danger score range: [{valid_danger.min():.3f}, {valid_danger.max():.3f}] (0=extreme, 1=safe)")
        print(f"  Mean danger score: {valid_danger.mean():.3f}")
        print(f"  Extreme danger (score < 0.2): {(valid_danger < 0.2).sum():,} ({(valid_danger < 0.2).sum()/len(valid_danger)*100:.1f}%)")
        print(f"  High danger (score < 0.4): {(valid_danger < 0.4).sum():,} ({(valid_danger < 0.4).sum()/len(valid_danger)*100:.1f}%)")
        print(f"  Moderate danger (score < 0.6): {(valid_danger < 0.6).sum():,} ({(valid_danger < 0.6).sum()/len(valid_danger)*100:.1f}%)")
        print(f"  Danger zone pixels saved: {len(danger_pixel_data):,}")
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
        # Custom colormap: black (0.0 = max danger) → red → orange → yellow → white (1.0 = safe)
        colors = ['#000000', '#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFFF00', '#FFFFFF']
        cmap_danger = LinearSegmentedColormap.from_list('danger', colors, N=256)
        cmap_danger.set_bad(color='none', alpha=0)  # Make NaN transparent
        
        danger_masked = np.ma.masked_invalid(danger_grid)
        
        # Full scale [0, 1] - occupancy-based
        vmin = 0.0
        vmax = 1.0
        
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
        ax.set_title('Fire Danger Map: CHE Occupancy in U-Space\n'
                    'Darker = Higher occupancy (more extreme fire conditions)',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Danger Score\n(0=Extreme, 1=Safe)', fontsize=11, fontweight='bold')
        
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
        ax.set_title('Fire Danger Map (CHE Occupancy)',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Danger Score\n(0=Extreme, 1=Safe)', fontsize=11, fontweight='bold')
        
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
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Process landscape in chunks (rows)')
    
    args = parser.parse_args()
    
    # Append git commit hash to output directory
    commit_hash = get_git_commit_hash()
    output_dir = Path(args.output) / commit_hash
    print(f"Git commit: {commit_hash}")
    print(f"Output directory: {output_dir}\n")
    
    # Initialize mapper
    mapper = DangerMapper(args.data_root)
    
    # Load PCA transformation
    mapper.load_transformation(args.u_space)
    
    # Load CHE envelope
    mapper.load_envelope(args.che)
    
    # Create danger map
    danger_grid = mapper.create_danger_map(chunk_size=args.chunk_size)
    
    # Save results
    mapper.save_results(danger_grid, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Danger Map Creation Complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
