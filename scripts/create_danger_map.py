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


class DangerMapper:
    """Create danger map by projecting landscape to U-space and checking envelope membership"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        
        # Load landscape
        landscape_path = self.data_root / 'landscape' / 'Input_Geotiff.tif'
        print(f"Loading landscape from {landscape_path}...")
        
        with rasterio.open(landscape_path) as src:
            self.landscape = src.read().astype(np.float32)  # [8, H, W]
            self.landscape_shape = src.shape
            self.landscape_transform = src.transform
            self.landscape_crs = src.crs
            self.nodata = -9999.0
        
        print(f"  Shape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
        print(f"  Bands: {self.landscape.shape[0]}")
        print(f"  CRS: {self.landscape_crs}")
        
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
            if feat_name == 'elevation':
                features.append(pixel_features['elevation'])
            elif feat_name == 'slope':
                features.append(pixel_features['slope'])
            elif feat_name == 'fuel_load':
                features.append(pixel_features['fuel_load'])
            elif feat_name == 'vegetation':
                features.append(pixel_features['vegetation'])
            elif feat_name == 'canopy_height':
                features.append(pixel_features['canopy_height'])
            elif feat_name == 'canopy_density':
                features.append(pixel_features['canopy_density'])
            elif feat_name == 'aspect_cos':
                aspect_rad = pixel_features['aspect'] * np.pi / 180
                features.append(np.cos(aspect_rad))
            elif feat_name == 'aspect_sin':
                aspect_rad = pixel_features['aspect'] * np.pi / 180
                features.append(np.sin(aspect_rad))
            elif feat_name == 'wind_speed':
                features.append(pixel_features['wind_speed'])
            elif feat_name == 'wind_u':
                # Use typical summer wind (or could load from weather data)
                features.append(pixel_features['wind_speed'] * np.cos(pixel_features['wind_direction'] * np.pi / 180))
            elif feat_name == 'wind_v':
                features.append(pixel_features['wind_speed'] * np.sin(pixel_features['wind_direction'] * np.pi / 180))
        
        return np.array(features, dtype=np.float32)
    
    def project_to_uspace(self, X):
        """Project features to U-space using saved PCA"""
        # Z-score normalize
        X_scaled = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # PCA projection
        U = X_scaled @ self.pca_components.T
        
        return U
    
    def check_envelope_membership(self, U):
        """Check if U-space point is inside CHE envelope"""
        u1, u2 = U[0], U[1]
        
        # Get bounds
        u1_min, u2_min = self.envelope_bounds[0]
        u1_max, u2_max = self.envelope_bounds[1]
        
        # Check if outside bounds
        if u1 < u1_min or u1 > u1_max or u2 < u2_min or u2 > u2_max:
            return False, float('inf')
        
        # Map to grid indices
        i = int((u1 - u1_min) / (u1_max - u1_min) * (self.occupancy_grid.shape[1] - 1))
        j = int((u2 - u2_min) / (u2_max - u2_min) * (self.occupancy_grid.shape[0] - 1))
        
        # Clamp to grid bounds
        i = np.clip(i, 0, self.occupancy_grid.shape[1] - 1)
        j = np.clip(j, 0, self.occupancy_grid.shape[0] - 1)
        
        # Check envelope mask
        inside = bool(self.envelope_mask[j, i])
        
        # Get occupancy score (distance proxy)
        occupancy = float(self.occupancy_grid[j, i])
        
        return inside, 1.0 - occupancy
    
    def create_danger_map(self, typical_wind_speed=5.0, typical_wind_dir=180.0, chunk_size=1000):
        """
        Create danger map over full landscape
        
        Args:
            typical_wind_speed: Typical summer wind speed (m/s) to use for all pixels
            typical_wind_dir: Typical wind direction (degrees)
            chunk_size: Process landscape in chunks (rows at a time)
        
        Returns:
            danger_grid: [H, W] array with danger scores
        """
        print(f"\n{'='*60}")
        print(f"Creating Danger Map via U-Space Projection")
        print(f"{'='*60}")
        print(f"Landscape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
        print(f"Using typical wind: {typical_wind_speed} m/s @ {typical_wind_dir}°")
        print(f"Processing in chunks of {chunk_size} rows...")
        
        H, W = self.landscape_shape
        danger_grid = np.full((H, W), np.nan, dtype=np.float32)
        
        # Band mapping
        band_idx = {
            'elevation': 0,
            'slope': 1,
            'aspect': 2,
            'fuel_load': 3,
            'vegetation': 4,
            'canopy_height': 5,
            'canopy_density': 6,
        }
        
        # Process in chunks (row-wise)
        for start_row in tqdm(range(0, H, chunk_size), desc="Processing landscape"):
            end_row = min(start_row + chunk_size, H)
            
            for y in range(start_row, end_row):
                for x in range(W):
                    # Skip invalid pixels
                    if not self.valid_mask[y, x]:
                        continue
                    
                    # Extract environmental features
                    pixel_features = {
                        'elevation': self.landscape[band_idx['elevation'], y, x],
                        'slope': self.landscape[band_idx['slope'], y, x],
                        'aspect': self.landscape[band_idx['aspect'], y, x],
                        'fuel_load': self.landscape[band_idx['fuel_load'], y, x],
                        'vegetation': self.landscape[band_idx['vegetation'], y, x],
                        'canopy_height': self.landscape[band_idx['canopy_height'], y, x],
                        'canopy_density': self.landscape[band_idx['canopy_density'], y, x],
                        'wind_speed': typical_wind_speed,
                        'wind_direction': typical_wind_dir,
                    }
                    
                    # Engineer features
                    X = self.engineer_pixel_features(pixel_features)
                    
                    # Project to U-space
                    U = self.project_to_uspace(X)
                    
                    # Check envelope membership
                    inside, distance = self.check_envelope_membership(U)
                    
                    if inside:
                        # Inside envelope = high danger (0.5-0.7 based on occupancy)
                        danger_grid[y, x] = 0.5 + 0.2 * distance
                    else:
                        # Outside envelope = lower danger (0.7-1.0 based on distance)
                        danger_grid[y, x] = 0.7 + 0.3 * min(distance, 1.0)
        
        print(f"\n✓ Danger map created")
        valid_danger = danger_grid[~np.isnan(danger_grid)]
        print(f"  Valid pixels: {len(valid_danger):,}")
        print(f"  Danger range: [{valid_danger.min():.3f}, {valid_danger.max():.3f}]")
        print(f"  Mean danger: {valid_danger.mean():.3f}")
        print(f"  High danger pixels (< 0.7): {(valid_danger < 0.7).sum():,}")
        print(f"{'='*60}\n")
        
        return danger_grid
    
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
        
        elevation = self.landscape[0].copy()
        elevation[elevation == self.nodata] = np.nan
        
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
        # Custom colormap: black-orange (0.5) → white (1.0)
        colors = ['#000000', '#FF4500', '#FFA500', '#FFFF00', '#FFFFFF']
        cmap_danger = LinearSegmentedColormap.from_list('danger', colors, N=100)
        
        danger_masked = np.ma.masked_invalid(danger_grid)
        
        im = ax.imshow(
            danger_masked,
            extent=extent,
            origin='upper',
            cmap=cmap_danger,
            vmin=0.5,
            vmax=1.0,
            alpha=0.8,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Fire Danger Map: Environmental Hypervolume Projection\n'
                    'Areas with conditions matching extreme fire events',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Danger Score\n(0.5=Extreme, 1.0=Safe)', fontsize=11, fontweight='bold')
        
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
            vmin=0.5,
            vmax=1.0,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Fire Danger Map (Danger Only)',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Danger Score\n(0.5=Extreme, 1.0=Safe)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'danger_map_only.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print(f"✓ Saved Results")
        print(f"{'='*60}")
        print(f"  {output_dir}/danger_grid.npz")
        print(f"  {output_dir}/elevation_map.png")
        print(f"  {output_dir}/danger_map.png")
        print(f"  {output_dir}/danger_map_only.png")
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
