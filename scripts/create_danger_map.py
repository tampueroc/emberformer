"""
Create Spatial Danger Map from Hypervolume Top 1% Pixels

Stage 5 of hypervolume pipeline:
- Load top 1% important pixels from U-space
- Map back to absolute landscape coordinates using indices.json
- Account for resize transformation (406×406 → original)
- Compute spatial danger map based on proximity to extreme pixels

Usage:
    python scripts/create_danger_map.py \
        --u_space data/u_space \
        --salience data/salience \
        --data_root ~/data/deep_crown_dataset/organized_spreads \
        --output data/danger_map
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show


class DangerMapper:
    """Map U-space top 1% pixels back to absolute landscape coordinates"""
    
    def __init__(self, data_root, resize_to=406):
        self.data_root = Path(data_root)
        self.resize_to = resize_to
        
        # Load spatial indices (crop boundaries)
        with open(self.data_root / 'landscape' / 'indices.json', 'r') as f:
            self.indices = json.load(f)
        
        # Load landscape geotiff to get full extent
        landscape_path = self.data_root / 'landscape' / 'Input_Geotiff.tif'
        with rasterio.open(landscape_path) as src:
            self.landscape_shape = src.shape  # (height, width)
            self.landscape_transform = src.transform
            self.landscape_crs = src.crs
        
        print(f"Loaded indices for {len(self.indices)} fire sequences")
        print(f"Landscape shape: {self.landscape_shape[0]} × {self.landscape_shape[1]} pixels")
    
    def load_extreme_and_top1pct_pixels(self, u_space_dir, salience_dir):
        """Load all extreme fire pixels and identify top 1% for danger sources"""
        print(f"\n{'='*60}")
        print(f"Loading Extreme Fire Pixels")
        print(f"{'='*60}")
        
        # Load U-space data
        u_space_dir = Path(u_space_dir)
        data = np.load(u_space_dir / 'extreme.npz')
        gradcam = data['gradcam']
        
        # Filter for top 1% as danger sources
        threshold_99 = np.percentile(gradcam, 99)
        top_1pct_mask = gradcam > threshold_99
        
        print(f"Top 1% Grad-CAM threshold: {threshold_99:.4f}")
        print(f"Top 1% pixels (danger sources): {top_1pct_mask.sum():,} / {len(gradcam):,}")
        
        # Load salience parquet files - get ALL extreme fire pixels
        salience_dir = Path(salience_dir)
        
        # Get extreme fire threshold from quantiles
        with open(salience_dir / 'quantiles.json', 'r') as f:
            quantiles = json.load(f)
        extreme_threshold = quantiles['p99']
        
        print(f"\nLoading ALL extreme fire pixels (fire_intensity > {extreme_threshold:.2f})...")
        
        parquet_files = sorted(salience_dir.glob('part-*.parquet'))
        
        # Load all extreme fire pixels
        all_rows = []
        top_1pct_rows = []
        row_offset = 0
        
        for pfile in tqdm(parquet_files, desc="Loading parquet"):
            # Filter for extreme fires during read
            table = pq.read_table(pfile, filters=[
                ('fire_intensity', '>', extreme_threshold)
            ])
            df = table.to_pandas()
            
            if len(df) == 0:
                continue
            
            # Add all extreme pixels
            all_rows.append(df)
            
            # Track which rows are in top 1% Grad-CAM (danger sources)
            file_size = len(df)
            file_mask = top_1pct_mask[row_offset:row_offset + file_size]
            
            if file_mask.sum() > 0:
                df_top = df[file_mask].copy()
                top_1pct_rows.append(df_top)
            
            row_offset += file_size
        
        if len(all_rows) == 0:
            raise ValueError("No extreme fire pixels found!")
        
        df_all_extreme = pd.concat(all_rows, ignore_index=True)
        df_top_1pct = pd.concat(top_1pct_rows, ignore_index=True) if top_1pct_rows else pd.DataFrame()
        
        print(f"\n✓ Loaded {len(df_all_extreme):,} total extreme fire pixels")
        print(f"✓ Identified {len(df_top_1pct):,} top 1% danger sources")
        print(f"{'='*60}\n")
        
        return df_all_extreme, df_top_1pct
    
    def map_to_absolute_coords(self, df):
        """
        Map resized coordinates back to absolute landscape coordinates
        
        Process:
        1. Use sequence_id from dataframe (tracked during extraction)
        2. Lookup crop boundaries from indices.json
        3. Map (y_resized, x_resized) → (y_tile, x_tile)
        """
        print(f"\n{'='*60}")
        print(f"Mapping to Absolute Coordinates")
        print(f"{'='*60}")
        
        # Get unique sequences
        unique_sequences = df['sequence_id'].unique()
        print(f"Found {len(unique_sequences)} unique fire sequences")
        
        absolute_coords = []
        
        # Process each sequence
        for seq_id in tqdm(unique_sequences, desc="Mapping sequences"):
            df_seq = df[df['sequence_id'] == seq_id]
            
            if seq_id not in self.indices:
                print(f"  ⚠️  Sequence {seq_id} not in indices.json, skipping {len(df_seq)} pixels")
                continue
            
            # Get crop boundaries for this sequence
            y_min, y_max, x_min, x_max = self.indices[seq_id]
            orig_height = y_max - y_min
            orig_width = x_max - x_min
            
            # Map all pixels from this sequence
            for _, row in df_seq.iterrows():
                # Inverse resize: 406×406 → original crop size
                y_orig = (row['y'] / self.resize_to) * orig_height
                x_orig = (row['x'] / self.resize_to) * orig_width
                
                # Add crop offset to get absolute tile coordinates
                y_abs = y_min + y_orig
                x_abs = x_min + x_orig
                
                absolute_coords.append({
                    'y_abs': int(y_abs),
                    'x_abs': int(x_abs),
                    'y_resized': row['y'],
                    'x_resized': row['x'],
                    'gradcam': row['gradcam'],
                    'fire_intensity': row['fire_intensity'],
                    'sequence_id': seq_id,
                    'sample_id': row['sample_id'],
                })
        
        df_abs = pd.DataFrame(absolute_coords)
        
        print(f"\n✓ Mapped {len(df_abs):,} pixels to absolute coordinates")
        print(f"  Y range: [{df_abs['y_abs'].min()}, {df_abs['y_abs'].max()}]")
        print(f"  X range: [{df_abs['x_abs'].min()}, {df_abs['x_abs'].max()}]")
        print(f"  Sequences processed: {len(unique_sequences)}")
        print(f"{'='*60}\n")
        
        return df_abs
    
    def create_danger_grid(self, df_abs_all, df_abs_top1pct, grid_resolution=10, sigma=100):
        """
        Create danger map over FULL LANDSCAPE using KDTree for proximity calculation
        
        Args:
            df_abs_all: DataFrame with ALL extreme pixel absolute coordinates
            df_abs_top1pct: DataFrame with top 1% danger source coordinates
            grid_resolution: Downsampling factor (10 = every 10th pixel)
            sigma: Distance decay parameter (pixels)
        
        Returns:
            danger_grid, extent, df_abs_all
        """
        print(f"\n{'='*60}")
        print(f"Creating Danger Grid Over Full Landscape")
        print(f"{'='*60}")
        
        # Get extent from FULL LANDSCAPE (not just extreme pixels)
        landscape_height, landscape_width = self.landscape_shape
        y_min, y_max = 0, landscape_height
        x_min, x_max = 0, landscape_width
        
        print(f"Full landscape extent: Y=[{y_min}, {y_max}], X=[{x_min}, {x_max}]")
        print(f"Grid resolution: 1/{grid_resolution} sampling")
        print(f"Distance decay σ: {sigma} pixels")
        
        # Create grid over FULL landscape
        y_grid = np.arange(y_min, y_max, grid_resolution)
        x_grid = np.arange(x_min, x_max, grid_resolution)
        
        print(f"Grid shape: {len(y_grid)} × {len(x_grid)} = {len(y_grid) * len(x_grid):,} cells")
        
        # Build KDTree from TOP 1% danger sources ONLY
        danger_sources = df_abs_top1pct[['y_abs', 'x_abs']].values
        print(f"Using {len(danger_sources):,} top 1% pixels as danger sources")
        tree = KDTree(danger_sources)
        
        # Compute danger for each grid cell
        danger_grid = np.zeros((len(y_grid), len(x_grid)))
        
        for i, y in enumerate(tqdm(y_grid, desc="Computing danger")):
            for j, x in enumerate(x_grid):
                # Find distance to nearest top 1% danger source
                dist, _ = tree.query([y, x])
                
                # Danger score: 0.5 (high) near danger sources, 1.0 (low) far away
                danger_grid[i, j] = 0.5 + 0.5 * (1 - np.exp(-dist / sigma))
        
        extent = [x_min, x_max, y_min, y_max]
        
        print(f"\n✓ Danger grid created")
        print(f"  Danger range: [{danger_grid.min():.3f}, {danger_grid.max():.3f}]")
        print(f"  Mean danger: {danger_grid.mean():.3f}")
        print(f"{'='*60}\n")
        
        return danger_grid, extent, df_abs_all, df_abs_top1pct
    
    def save_results(self, danger_grid, extent, df_abs_all, df_abs_top1pct, output_dir):
        """Save danger map and extreme pixel locations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load landscape to create valid data mask
        landscape_path = self.data_root / 'landscape' / 'Input_Geotiff.tif'
        with rasterio.open(landscape_path) as src:
            # Band 4 = fuel_load (most relevant for fire)
            fuel_load = src.read(4).astype(float)
            # Create mask for valid landscape data (not NoData)
            valid_mask = fuel_load != -9999
            # Mask NoData for visualization
            fuel_load[~valid_mask] = np.nan
        
        # Downsample valid_mask to match danger_grid resolution
        from scipy.ndimage import zoom
        zoom_factor = (danger_grid.shape[0] / valid_mask.shape[0], 
                      danger_grid.shape[1] / valid_mask.shape[1])
        valid_mask_ds = zoom(valid_mask.astype(float), zoom_factor, order=0) > 0.5
        
        # Apply landscape mask to danger grid
        danger_grid_masked = danger_grid.copy()
        danger_grid_masked[~valid_mask_ds] = np.nan
        
        # Save danger grid as numpy
        np.savez_compressed(
            output_dir / 'danger_grid.npz',
            danger=danger_grid_masked,
            extent=extent,
            valid_mask=valid_mask_ds
        )
        
        # Save all extreme pixel locations
        df_abs_all.to_csv(output_dir / 'extreme_pixel_locations_all.csv', index=False)
        
        # Save top 1% danger sources
        df_abs_top1pct.to_csv(output_dir / 'danger_sources_top1pct.csv', index=False)
        
        # Visualize with landscape background
        print(f"Creating visualization with fuel load background...")
        
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Show fuel load as background
        im_bg = ax.imshow(
            fuel_load, 
            cmap='YlGn',  # Yellow to Green for fuel
            alpha=0.5, 
            extent=extent, 
            origin='upper',
            interpolation='bilinear'
        )
        
        # Overlay danger map - mask both invalid areas AND low-danger areas
        danger_masked = np.ma.masked_where(
            (danger_grid_masked > 0.95) | np.isnan(danger_grid_masked), 
            danger_grid_masked
        )
        
        im = ax.imshow(
            danger_masked,
            extent=extent,
            origin='upper',
            cmap='YlOrRd',  # Yellow→Orange→Red for danger
            vmin=0.5,
            vmax=0.95,
            alpha=0.8,
            interpolation='bilinear'
        )
        
        # Highlight top 1% danger sources ONLY (don't plot all 1M pixels)
        ax.scatter(
            df_abs_top1pct['x_abs'],
            df_abs_top1pct['y_abs'],
            c='darkred',
            s=5,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.5,
            marker='*',
            label=f'Top 1% danger sources ({len(df_abs_top1pct):,})'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Fire Danger Map: Full Landscape\nDanger = Proximity to Extreme Fire Environmental Hypervolume', 
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Danger Level\n(Yellow=Moderate, Red=Extreme)', fontsize=11, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'danger_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create fuel-only map (no danger overlay)
        print(f"Creating fuel-only map...")
        
        fig, ax = plt.subplots(figsize=(18, 14))
        
        im_fuel = ax.imshow(
            fuel_load, 
            cmap='YlGn',
            extent=extent, 
            origin='upper',
            interpolation='bilinear'
        )
        
        ax.set_xlabel('X (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (landscape pixels)', fontsize=13, fontweight='bold')
        ax.set_title('Fuel Load: Full Landscape', 
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im_fuel, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
        cbar.set_label('Fuel Load', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fuel_load_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{'='*60}")
        print(f"✓ Saved Results")
        print(f"{'='*60}")
        print(f"  {output_dir}/danger_grid.npz")
        print(f"  {output_dir}/extreme_pixel_locations_all.csv ({len(df_abs_all):,} pixels)")
        print(f"  {output_dir}/danger_sources_top1pct.csv ({len(df_abs_top1pct):,} pixels)")
        print(f"  {output_dir}/danger_map.png")
        print(f"  {output_dir}/fuel_load_map.png")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Create Spatial Danger Map')
    parser.add_argument('--u_space', type=str, required=True,
                       help='Directory with U-space data')
    parser.add_argument('--salience', type=str, required=True,
                       help='Directory with salience parquet files')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset (for indices.json)')
    parser.add_argument('--output', type=str, default='data/danger_map',
                       help='Output directory for danger map')
    parser.add_argument('--resize_to', type=int, default=406,
                       help='Resize dimension used in training')
    parser.add_argument('--grid_resolution', type=int, default=10,
                       help='Spatial downsampling factor')
    parser.add_argument('--sigma', type=float, default=100,
                       help='Distance decay parameter (pixels)')
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = DangerMapper(args.data_root, args.resize_to)
    
    # Load all extreme pixels and top 1%
    df_all_extreme, df_top1pct = mapper.load_extreme_and_top1pct_pixels(args.u_space, args.salience)
    
    # Map to absolute coordinates
    df_abs_all = mapper.map_to_absolute_coords(df_all_extreme)
    df_abs_top1pct = mapper.map_to_absolute_coords(df_top1pct)
    
    # Create danger grid
    danger_grid, extent, df_abs_all, df_abs_top1pct = mapper.create_danger_grid(
        df_abs_all,
        df_abs_top1pct,
        grid_resolution=args.grid_resolution,
        sigma=args.sigma
    )
    
    # Save results
    mapper.save_results(danger_grid, extent, df_abs_all, df_abs_top1pct, args.output)
    
    print(f"\n{'='*60}")
    print(f"✓ Danger Map Creation Complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
