"""
Extract Grad-CAM Salience to Parquet

Stage 1 of hypervolume pipeline:
- Run Grad-CAM on dataset batches
- Extract top-percentile important pixels
- Stream to Parquet files with environmental features
- Track fire intensity quantiles for extreme fire filtering

Usage:
    python scripts/extract_salience.py \
        --checkpoint checkpoints/dino_phase2_best.pt \
        --output data/salience \
        --num_samples 1000 \
        --batch_size 16 \
        --importance_threshold 99
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import yaml
import sys
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import RawFireDataset
from models.emberformer import EmberFormerDINO
from scripts.analyze_gradcam import GradCAM, load_model
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset


SCHEMA = pa.schema([
    ('sample_id', pa.int32()),          # NEW: which sample in dataset
    ('sequence_id', pa.string()),       # NEW: fire sequence ID
    ('y', pa.int32()),                  # Y in resized 406×406 space
    ('x', pa.int32()),                  # X in resized 406×406 space
    ('gradcam', pa.float32()),
    ('forest', pa.float32()),
    ('arqueo', pa.float32()),
    ('cbd', pa.float32()),
    ('cbh', pa.float32()),
    ('elevation', pa.float32()),
    ('flora', pa.float32()),
    ('paleo', pa.float32()),
    ('urbana', pa.float32()),
    ('wind_speed', pa.float32()),
    ('wind_direction', pa.float32()),
    ('fire_intensity', pa.float32()),
])

STATIC_NAMES = [
    'forest', 'arqueo', 'cbd', 'cbh',
    'elevation', 'flora', 'paleo', 'urbana'
]


class SalienceExtractor:
    """Stream Grad-CAM salience pixels to Parquet with fire intensity tracking"""
    
    def __init__(self, model, device='cuda', importance_threshold=99, buffer_size=1_000_000):
        self.model = model
        self.device = device
        self.importance_threshold = importance_threshold
        self.buffer_size = buffer_size
        self.gradcam = GradCAM(model, model.refinement_decoder.output_conv)
        
        self.buffer = []
        self.part_idx = 0
        self.total_pixels = 0
        
        # Track fire intensity for quantile estimation
        self.fire_intensities = []
        
    def flush_buffer(self, output_dir):
        """Write buffer to Parquet part file"""
        if len(self.buffer) == 0:
            return
        
        # Convert to PyArrow Table
        table = pa.Table.from_pydict({
            'sample_id': pa.array([r['sample_id'] for r in self.buffer], type=pa.int32()),
            'sequence_id': pa.array([r['sequence_id'] for r in self.buffer], type=pa.string()),
            'y': pa.array([r['y'] for r in self.buffer], type=pa.int32()),
            'x': pa.array([r['x'] for r in self.buffer], type=pa.int32()),
            'gradcam': pa.array([r['gradcam'] for r in self.buffer], type=pa.float32()),
            'forest': pa.array([r['forest'] for r in self.buffer], type=pa.float32()),
            'arqueo': pa.array([r['arqueo'] for r in self.buffer], type=pa.float32()),
            'cbd': pa.array([r['cbd'] for r in self.buffer], type=pa.float32()),
            'cbh': pa.array([r['cbh'] for r in self.buffer], type=pa.float32()),
            'elevation': pa.array([r['elevation'] for r in self.buffer], type=pa.float32()),
            'flora': pa.array([r['flora'] for r in self.buffer], type=pa.float32()),
            'paleo': pa.array([r['paleo'] for r in self.buffer], type=pa.float32()),
            'urbana': pa.array([r['urbana'] for r in self.buffer], type=pa.float32()),
            'wind_speed': pa.array([r['wind_speed'] for r in self.buffer], type=pa.float32()),
            'wind_direction': pa.array([r['wind_direction'] for r in self.buffer], type=pa.float32()),
            'fire_intensity': pa.array([r['fire_intensity'] for r in self.buffer], type=pa.float32()),
        }, schema=SCHEMA)
        
        # Write to Parquet
        output_path = Path(output_dir) / f'part-{self.part_idx:05d}.parquet'
        pq.write_table(table, output_path, compression='snappy')
        
        self.total_pixels += len(self.buffer)
        self.part_idx += 1
        self.buffer = []
        
    def extract_batch(self, batch_fire, batch_static, batch_wind, batch_target, batch_valid_t, batch_indices, dataset):
        """Extract important pixels from a batch"""
        B = batch_fire.shape[0]
        
        # Compute Grad-CAM for entire batch
        with torch.enable_grad():
            cam = self.gradcam(batch_fire, batch_static, batch_wind, batch_valid_t)
        
        # Process each sample
        for b in range(B):
            fire_intensity = batch_target[b].sum().item()
            self.fire_intensities.append(fire_intensity)
            
            # Get sample metadata
            sample_idx = batch_indices[b].item() if hasattr(batch_indices[b], 'item') else batch_indices[b]
            sample_info = dataset.samples[sample_idx]
            sequence_id = sample_info['sequence_id']
            
            cam_np = cam[b].cpu().numpy()
            
            # Get important pixels (top percentile)
            threshold = np.percentile(cam_np, self.importance_threshold)
            important_mask = cam_np > threshold
            y_coords, x_coords = np.where(important_mask)
            
            if len(y_coords) == 0:
                continue
            
            # Extract environmental data
            static_np = batch_static[b].cpu().numpy()
            wind_np = batch_wind[b].cpu().numpy()
            
            for y, x in zip(y_coords, x_coords):
                row = {
                    'sample_id': int(sample_idx),
                    'sequence_id': str(sequence_id),
                    'y': int(y),
                    'x': int(x),
                    'gradcam': float(cam_np[y, x]),
                    'fire_intensity': float(fire_intensity),
                    'wind_speed': float(wind_np[-1, 0]),
                    'wind_direction': float(wind_np[-1, 1]),
                }
                
                # Add static features
                for i, name in enumerate(STATIC_NAMES[:min(8, static_np.shape[0])]):
                    if name in ['forest', 'arqueo', 'cbd', 'cbh', 
                               'elevation', 'flora', 'paleo', 'urbana']:
                        row[name] = float(static_np[i, y, x])
                
                self.buffer.append(row)
        
        return len(y_coords)
    
    def process_dataset(self, dataset, output_dir, num_samples=-1, batch_size=16):
        """Process dataset and write to Parquet"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the underlying dataset (unwrap Subset if needed)
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        
        # Create subset if needed
        if num_samples > 0 and num_samples < len(dataset):
            indices = list(range(num_samples))
            dataset_subset = Subset(base_dataset, indices)
        else:
            dataset_subset = dataset
        
        # Custom collate that returns indices
        def collate_with_indices(batch):
            from scripts.train_dino import collate_raw_dino
            indices = [item[0] for item in batch]
            samples = [item[1] for item in batch]
            collated = collate_raw_dino(samples)
            return collated, indices
        
        # Wrap dataset to return (index, sample)
        class IndexedDataset:
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                return idx, self.dataset[idx]
        
        indexed_dataset = IndexedDataset(dataset_subset)
        
        # Create dataloader
        loader = DataLoader(
            indexed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_with_indices
        )
        
        print(f"\n{'='*60}")
        print(f"Extracting Salience: {len(dataset_subset)} samples (batch_size={batch_size})")
        print(f"Output: {output_dir}")
        print(f"Importance threshold: top {100-self.importance_threshold}%")
        print(f"{'='*60}\n")
        
        # Process batches
        for (batch_fire, batch_static, batch_wind, batch_target, batch_valid_t), batch_indices in tqdm(loader, desc="Processing"):
            batch_fire = batch_fire.to(self.device)
            batch_static = batch_static.to(self.device)
            batch_wind = batch_wind.to(self.device)
            batch_valid_t = batch_valid_t.to(self.device)
            
            self.extract_batch(batch_fire, batch_static, batch_wind, batch_target, batch_valid_t, 
                             batch_indices, base_dataset)
            
            # Flush if buffer is large
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer(output_dir)
                torch.cuda.empty_cache()
        
        # Final flush
        self.flush_buffer(output_dir)
        
        # Save fire intensity quantiles
        quantiles = {
            'min': float(np.min(self.fire_intensities)),
            'p50': float(np.percentile(self.fire_intensities, 50)),
            'p75': float(np.percentile(self.fire_intensities, 75)),
            'p90': float(np.percentile(self.fire_intensities, 90)),
            'p95': float(np.percentile(self.fire_intensities, 95)),
            'p99': float(np.percentile(self.fire_intensities, 99)),
            'max': float(np.max(self.fire_intensities)),
            'n_samples': len(self.fire_intensities),
        }
        
        with open(output_dir / 'quantiles.json', 'w') as f:
            json.dump(quantiles, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction Complete")
        print(f"{'='*60}")
        print(f"Total pixels extracted: {self.total_pixels:,}")
        print(f"Parquet parts written: {self.part_idx}")
        print(f"Fire intensity quantiles:")
        print(f"  p50: {quantiles['p50']:.2f}")
        print(f"  p95: {quantiles['p95']:.2f}")
        print(f"  p99: {quantiles['p99']:.2f}")
        print(f"  max: {quantiles['max']:.2f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract Grad-CAM Salience to Parquet')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str,
                       default='~/data/deep_crown_dataset/organized_spreads',
                       help='Path to dataset root')
    parser.add_argument('--output', type=str, default='data/salience',
                       help='Output directory for Parquet files')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to process (-1 = all)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for GPU processing')
    parser.add_argument('--importance_threshold', type=int, default=99,
                       help='Percentile threshold for important pixels (99 = top 1%)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--buffer_size', type=int, default=1_000_000,
                       help='Buffer size before flushing to disk')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_root}...")
    with open('configs/emberformer_dino.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    resize_to = cfg['data'].get('resize_to', 406)
    
    class ResizeTransform:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return TF.resize(img, [self.size, self.size],
                           interpolation=TF.InterpolationMode.BILINEAR,
                           antialias=True)
    
    transform = ResizeTransform(resize_to)
    full_dataset = RawFireDataset(args.data_root, sequence_length=4, transform=transform)
    
    # Split dataset
    total_samples = len(full_dataset.samples)
    train_size = int(cfg['split']['train'] * total_samples)
    val_size = int(cfg['split']['val'] * total_samples)
    
    if args.split == 'train':
        indices = list(range(0, train_size))
    elif args.split == 'val':
        indices = list(range(train_size, train_size + val_size))
    else:
        indices = list(range(train_size + val_size, total_samples))
    
    dataset = Subset(full_dataset, indices)
    
    print(f"Dataset split: {args.split}")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Extract salience
    extractor = SalienceExtractor(
        model,
        device=device,
        importance_threshold=args.importance_threshold,
        buffer_size=args.buffer_size
    )
    
    extractor.process_dataset(dataset, args.output, args.num_samples, args.batch_size)


if __name__ == '__main__':
    main()
