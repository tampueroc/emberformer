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
import pickle

# Add project root to path for imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from data.transforms import LandscapeNormalize

# Forest fuel type codes from spain_lookup_table.csv (39 categories)
# Must match prep_u_space_umap.py exactly
FOREST_CODES = np.array([
    0, 91, 92, 93, 98, 99,  # Non-fuel (6)
    101, 102, 103, 104, 105, 106, 107, 108,  # GR1-GR8 (8)
    121, 122, 123, 124,  # GS1-GS4 (4)
    142, 143, 144, 145, 146, 147, 148, 149,  # SH2-SH9 (8)
    161, 162, 163, 164, 165,  # TU1-TU5 (5)
    181, 182, 183, 185, 186, 188, 189,  # TL1-TL9 (7, some missing)
], dtype=np.int32)
N_FOREST_CATEGORIES = len(FOREST_CODES)  # 39

# Build code to index mapping once
FOREST_CODE_TO_IDX = {code: idx for idx, code in enumerate(FOREST_CODES)}

# Raw feature ranges for denormalization (must match prep_u_space_umap.py)
FEATURE_RANGES = {
    'forest': (0.0, 189.0),
    'cbd': (0.0, 0.4467),
    'cbh': (0.0, 13.8692),
    'elevation': (345.9422, 3012.5251),
}

# Number of nearest triangles to check for surface distance (3D only)
K_NEAREST_TRIS = 32


def point_triangle_distance_batch(points, triangles):
    """
    Compute minimum distance from each point to its set of candidate triangles.
    
    Uses the closest-point-on-triangle algorithm (Ericson, Real-Time Collision Detection).
    
    Args:
        points: [M, 3] array of query points
        triangles: [M, K, 3, 3] array of K candidate triangles per point
                   Each triangle has 3 vertices of 3 coordinates
    
    Returns:
        distances: [M] array of minimum distances to any triangle
    """
    M, K, _, _ = triangles.shape
    
    # Reshape for vectorized computation: [M*K, 3, 3]
    tri_flat = triangles.reshape(M * K, 3, 3)
    # Repeat points K times: [M*K, 3]
    pts_flat = np.repeat(points, K, axis=0)
    
    # Triangle vertices
    a = tri_flat[:, 0, :]  # [M*K, 3]
    b = tri_flat[:, 1, :]
    c = tri_flat[:, 2, :]
    p = pts_flat
    
    # Compute closest point on each triangle
    ab = b - a
    ac = c - a
    ap = p - a
    
    # Barycentric coordinates
    d1 = np.sum(ab * ap, axis=1)  # [M*K]
    d2 = np.sum(ac * ap, axis=1)
    d3 = np.sum(ab * ab, axis=1)
    d4 = np.sum(ab * ac, axis=1)
    d5 = np.sum(ac * ac, axis=1)
    
    bp = p - b
    d6 = np.sum(ab * bp, axis=1)
    d7 = np.sum(ac * bp, axis=1)
    
    cp = p - c
    d8 = np.sum(ab * cp, axis=1)
    d9 = np.sum(ac * cp, axis=1)
    
    # Check if P in vertex region outside A
    va = d1 <= 0
    vb = d2 <= 0
    region_a = va & vb
    
    # Check if P in edge region AB
    vc = d6 >= 0
    vd = d1 >= 0
    ve = d1 * d7 - d6 * d2 <= 0
    region_ab = vd & vc & ve & ~region_a
    
    # Check if P in vertex region outside B
    vf = d6 >= 0
    vg = d7 <= 0
    region_b = vf & vg & ~region_a & ~region_ab
    
    # Check if P in edge region BC
    vh = d8 <= 0
    vi = d9 >= 0
    vj = d6 * d9 - d8 * d7 >= 0
    region_bc = vi & vh & vj & ~region_a & ~region_ab & ~region_b
    
    # Check if P in vertex region outside C
    vk = d8 <= 0
    vl = d9 >= d5
    region_c = vk & vl & ~region_a & ~region_ab & ~region_b & ~region_bc
    
    # Check if P in edge region AC
    vm = d2 >= 0
    vn = d9 <= 0
    vo = d4 * d2 - d1 * d5 <= 0
    region_ac = vm & vn & vo & ~region_a & ~region_ab & ~region_b & ~region_bc & ~region_c
    
    # Otherwise P is inside face region
    region_face = ~(region_a | region_ab | region_b | region_bc | region_c | region_ac)
    
    # Compute closest points for each region
    closest = np.zeros_like(p)
    
    # Region A: closest is vertex A
    closest[region_a] = a[region_a]
    
    # Region B: closest is vertex B
    closest[region_b] = b[region_b]
    
    # Region C: closest is vertex C
    closest[region_c] = c[region_c]
    
    # Region AB: project onto edge AB
    t_ab = np.clip(d1 / (d3 + 1e-10), 0, 1)
    closest_ab = a + t_ab[:, None] * ab
    closest[region_ab] = closest_ab[region_ab]
    
    # Region BC: project onto edge BC
    bc = c - b
    t_bc = np.clip(np.sum((p - b) * bc, axis=1) / (np.sum(bc * bc, axis=1) + 1e-10), 0, 1)
    closest_bc = b + t_bc[:, None] * bc
    closest[region_bc] = closest_bc[region_bc]
    
    # Region AC: project onto edge AC
    t_ac = np.clip(d2 / (d5 + 1e-10), 0, 1)
    closest_ac = a + t_ac[:, None] * ac
    closest[region_ac] = closest_ac[region_ac]
    
    # Region face: project onto plane
    denom = d3 * d5 - d4 * d4 + 1e-10
    v_bary = (d4 * d2 - d5 * d1) / denom
    w_bary = (d4 * d1 - d3 * d2) / denom
    v_bary = np.clip(v_bary, 0, 1)
    w_bary = np.clip(w_bary, 0, 1)
    closest_face = a + v_bary[:, None] * ab + w_bary[:, None] * ac
    closest[region_face] = closest_face[region_face]
    
    # Compute distances
    dist_flat = np.linalg.norm(p - closest, axis=1)  # [M*K]
    
    # Reshape and take minimum over K triangles
    dist_per_tri = dist_flat.reshape(M, K)
    min_dist = dist_per_tri.min(axis=1)  # [M]
    
    return min_dist


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
        """Load transformation metadata from prep_u_space stage (PCA or UMAP)"""
        u_space_dir = Path(u_space_dir)
        
        with open(u_space_dir / 'transform.json', 'r') as f:
            self.transform_meta = json.load(f)
        
        self.transform_method = self.transform_meta.get('method', 'pca')
        self.feature_names = self.transform_meta['feature_names']
        self.n_components = self.transform_meta['n_components']
        
        print(f"\nLoading {self.transform_method.upper()} transformation from {u_space_dir}...")
        
        if self.transform_method == 'umap':
            # Load fitted UMAP model and scaler
            with open(u_space_dir / 'umap_model.pkl', 'rb') as f:
                self.umap_model = pickle.load(f)
            # Load scaler for standardization before UMAP
            scaler_path = u_space_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.umap_scaler = pickle.load(f)
                print(f"  Scaler: StandardScaler loaded")
            else:
                self.umap_scaler = None
            print(f"  Features: {self.feature_names}")
            print(f"  UMAP components: {self.n_components}")
        elif self.transform_method == 'direct':
            # Load fitted scaler only (no dimensionality reduction)
            with open(u_space_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"  Features: {self.feature_names}")
            print(f"  Dimensions: {self.n_components} (no reduction)")
        else:
            # PCA: load components for matrix multiplication
            self.scaler_mean = np.array(self.transform_meta['scaler_mean'])
            self.scaler_std = np.array(self.transform_meta['scaler_std'])
            self.pca_components = np.array(self.transform_meta['pca_components'])
            print(f"  Features: {self.feature_names}")
            print(f"  PCA components: {self.n_components}")
            print(f"  Total variance: {self.transform_meta['total_variance']*100:.2f}%")
    
    def load_envelope(self, che_dir):
        """Load convex hull envelope from CHE ensemble"""
        che_dir = Path(che_dir)
        
        print(f"\nLoading CHE ensemble from {che_dir}...")
        
        # Load CHE ensemble data
        che_data = np.load(che_dir / 'che_ensemble.npz')
        vertices_concat = che_data['vertices_concat']
        offsets = che_data['offsets']
        self.n_dims = int(che_data['n_dims'])
        n_bootstraps = int(che_data['n_bootstraps'])
        
        # For now, use first hull for membership test (TODO: occupancy-based)
        # Extract first hull's vertices
        first_hull_verts = vertices_concat[offsets[0]:offsets[1]]
        
        # Reconstruct Delaunay for point-in-hull queries
        from scipy.spatial import Delaunay, ConvexHull, cKDTree
        self.delaunay = Delaunay(first_hull_verts)
        
        # Store all hull vertices for potential occupancy computation
        self.all_hull_vertices = []
        for i in range(n_bootstraps):
            self.all_hull_vertices.append(vertices_concat[offsets[i]:offsets[i+1]])
        
        # For 3D: build acceleration structure for surface distance
        self.tri_tree = None
        self.tri_verts = None
        if self.n_dims == 3:
            # Build hull from first bootstrap's vertices
            self.hull3 = ConvexHull(first_hull_verts, qhull_options='QJ')
            simplices = self.hull3.simplices  # [T, 3] triangle vertex indices
            self.tri_verts = first_hull_verts[simplices]  # [T, 3, 3] triangle vertex coords
            
            # Build KDTree on triangle centroids for fast nearest-triangle lookup
            centroids = self.tri_verts.mean(axis=1)  # [T, 3]
            self.tri_tree = cKDTree(centroids)
            
            print(f"  3D surface distance: {len(simplices)} triangles, KDTree built")
        
        # Load summary
        with open(che_dir / 'summary.json', 'r') as f:
            che_summary = json.load(f)
        
        print(f"  Method: {che_summary['method']}")
        print(f"  Bootstraps: {n_bootstraps}")
        print(f"  Mean vertices: {che_summary['vertices_mean']:.1f}")
        print(f"  Mean volume: {che_summary['volume_mean']:.2e}")
        print(f"  Dimensions: {self.n_dims}")
    
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
        """Project features to U-space using saved transformation (PCA or UMAP)"""
        if self.transform_method == 'umap':
            # UMAP: use fitted model's transform on raw features
            X_input = X.reshape(1, -1) if X.ndim == 1 else X
            U = self.umap_model.transform(X_input)
            return U.flatten() if X.ndim == 1 else U
        elif self.transform_method == 'direct':
            # Direct: just z-score normalize using fitted scaler
            X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
            return X_scaled.flatten() if X.ndim == 1 else X_scaled
        else:
            # PCA: simple z-score + matrix multiplication
            X_scaled = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
            U = X_scaled @ self.pca_components.T
            return U
    
    def is_inside_envelope(self, U):
        """
        Check if point is inside convex hull envelope
        
        Returns:
            bool: True if inside (dangerous), False if outside (safe)
        """
        U_check = U[:self.n_dims].reshape(1, -1)
        return self.delaunay.find_simplex(U_check)[0] >= 0
    
    def distance_to_hull(self, U):
        """
        Compute distance from point to convex hull boundary.
        Returns 0 if inside, positive distance if outside.
        """
        from scipy.spatial import distance
        
        U_check = U[:self.n_dims]
        
        # Get hull vertices
        hull_points = self.delaunay.points
        
        # Compute distance to nearest hull vertex (approximation)
        # For exact distance to hull surface, would need to check all facets
        distances = distance.cdist([U_check], hull_points, 'euclidean')[0]
        return distances.min()
    
    def compute_danger_score(self, U, max_distance=5.0):
        """
        Danger classification based on proximity to hypervolume:
        - Inside envelope: 0.5 (dangerous)
        - Outside envelope: 0.5 to 1.0 based on distance (closer = more dangerous)
        
        Args:
            max_distance: Distance at which score reaches 1.0 (safe)
        """
        if self.is_inside_envelope(U):
            return 0.5  # Dangerous - inside the extreme fire hypervolume
        else:
            # Outside: scale by distance to hull
            dist = self.distance_to_hull(U)
            # Normalize: 0 distance -> 0.5, max_distance -> 1.0
            normalized = min(dist / max_distance, 1.0)
            return 0.5 + 0.5 * normalized  # Range: 0.5 to 1.0
    
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
        
        # Process only valid pixels (where elevation is valid)
        print(f"  Processing {self.valid_mask.sum():,} valid pixels...")
        
        y_coords, x_coords = np.where(self.valid_mask)
        n_valid = len(y_coords)
        
        # Extract and engineer features (with one-hot forest encoding)
        print(f"  Extracting and encoding features...")
        
        # Check if using one-hot encoding (feature names start with 'forest_')
        uses_onehot_forest = any(fn.startswith('forest_') for fn in self.feature_names)
        
        if uses_onehot_forest:
            # One-hot encode forest from raw landscape (integer fuel codes)
            forest_raw = self.landscape_raw[band_idx['forest'], y_coords, x_coords]
            forest_codes = np.round(forest_raw).astype(np.int32)
            
            # Create one-hot encoding (vectorized)
            forest_onehot = np.zeros((n_valid, N_FOREST_CATEGORIES), dtype=np.float32)
            for code, idx in FOREST_CODE_TO_IDX.items():
                mask = forest_codes == code
                forest_onehot[mask, idx] = 1.0
            
            # Build feature matrix in correct order
            feature_arrays = [forest_onehot]
            
            # Add continuous features in order - DENORMALIZED to raw scale
            for feat_name in self.feature_names:
                if feat_name.startswith('forest_'):
                    continue  # Already handled
                if feat_name in band_idx:
                    # Get normalized values from landscape
                    feat_normalized = self.landscape[band_idx[feat_name], y_coords, x_coords]
                    # Denormalize to raw scale
                    fmin, fmax = FEATURE_RANGES[feat_name]
                    feat_raw = feat_normalized * (fmax - fmin) + fmin
                    feature_arrays.append(feat_raw.reshape(-1, 1))
                    print(f"    {feat_name}: denormalized to raw scale ({fmin:.2f}-{fmax:.2f})")
            
            X_all = np.hstack(feature_arrays)
            print(f"    One-hot forest: {N_FOREST_CATEGORIES} categories")
            print(f"    Feature matrix shape: {X_all.shape}")
        else:
            # Legacy: direct feature extraction
            X_all = np.zeros((n_valid, len(self.feature_names)), dtype=np.float32)
            for i, feat_name in enumerate(self.feature_names):
                if feat_name in band_idx:
                    X_all[:, i] = self.landscape[band_idx[feat_name], y_coords, x_coords]
        
        # Project all to U-space in batch
        print(f"  Projecting to U-space (batch)...")
        if self.transform_method == 'umap':
            # Apply scaler if available (standardization before UMAP)
            if hasattr(self, 'umap_scaler') and self.umap_scaler is not None:
                print(f"    Applying StandardScaler...")
                X_scaled = self.umap_scaler.transform(X_all)
                U_all = self.umap_model.transform(X_scaled)
            else:
                U_all = self.umap_model.transform(X_all)
        else:
            U_all = self.project_to_uspace(X_all)
        
        # Check inside/outside hull in batch
        print(f"  Computing danger scores (batch)...")
        inside_mask = self.delaunay.find_simplex(U_all[:, :self.n_dims]) >= 0
        
        # For inside: score = 0.5
        # For outside: compute distance-based score
        danger_scores = np.ones(n_valid, dtype=np.float32)
        danger_scores[inside_mask] = 0.5
        
        # Distance computation for outside points
        outside_indices = np.where(~inside_mask)[0]
        max_distance = 5.0
        
        if len(outside_indices) > 0:
            U_outside = U_all[outside_indices, :self.n_dims]
            
            if self.n_dims == 3 and self.tri_tree is not None:
                # 3D: use exact distance to hull surface (triangle mesh)
                print(f"  Computing 3D surface distances for {len(outside_indices):,} outside points...")
                
                # Query K nearest triangles for each outside point
                _, tri_indices = self.tri_tree.query(U_outside, k=K_NEAREST_TRIS)
                
                # Get candidate triangles for each point: [M, K, 3, 3]
                candidate_tris = self.tri_verts[tri_indices]
                
                # Compute exact distance to nearest triangle surface
                distances = point_triangle_distance_batch(U_outside, candidate_tris)
                
                normalized = np.clip(distances / max_distance, 0.0, 1.0)
                danger_scores[outside_indices] = 0.5 + 0.5 * normalized
            else:
                # Non-3D: fall back to distance to nearest vertex
                from scipy.spatial import distance
                hull_points = self.delaunay.points
                for idx in tqdm(outside_indices, desc="Computing distances for outside points"):
                    U_check = U_all[idx, :self.n_dims]
                    dist = distance.cdist([U_check], hull_points, 'euclidean')[0].min()
                    normalized = min(dist / max_distance, 1.0)
                    danger_scores[idx] = 0.5 + 0.5 * normalized
        
        # Fill danger grid
        danger_grid[y_coords, x_coords] = danger_scores
        
        # Collect dangerous pixel data
        danger_pixel_data = []
        danger_indices = np.where(inside_mask)[0]
        for idx in danger_indices:
            y, x = y_coords[idx], x_coords[idx]
            danger_record = {
                'y': int(y),
                'x': int(x),
                'forest': float(self.landscape[band_idx['forest'], y, x]),
                'cbd': float(self.landscape[band_idx['cbd'], y, x]),
                'cbh': float(self.landscape[band_idx['cbh'], y, x]),
                'elevation': float(self.landscape[band_idx['elevation'], y, x]),
                'danger_score': 0.5,
            }
            danger_pixel_data.append(danger_record)
        
        # Statistics
        valid_danger = danger_grid[~np.isnan(danger_grid)]
        n_inside = (valid_danger == 0.5).sum()
        n_outside = (valid_danger > 0.5).sum()
        
        print(f"\n✓ Danger map created (proximity-based classification)")
        print(f"  Valid pixels: {len(valid_danger):,}")
        print(f"  Inside envelope (score=0.5): {n_inside:,} ({100*n_inside/len(valid_danger):.1f}%)")
        print(f"  Outside envelope (score>0.5): {n_outside:,} ({100*n_outside/len(valid_danger):.1f}%)")
        print(f"  Score range: [{valid_danger.min():.3f}, {valid_danger.max():.3f}]")
        print(f"  Mean score: {valid_danger.mean():.3f}")
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
        
        # Fixed scale: 0.5 (inside/dangerous) to 1.0 (outside/safe)
        vmin = 0.5
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
        cbar.set_label('Danger Score\n(0.5=Inside CHE, 1=Outside)', fontsize=11, fontweight='bold')
        
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
        cbar.set_label('Danger Score\n(0.5=Inside CHE, 1=Outside)', fontsize=11, fontweight='bold')
        
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
