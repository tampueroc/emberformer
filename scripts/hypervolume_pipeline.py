"""
Hypervolume Analysis Pipeline Orchestrator

Multi-stage pipeline reproducing Pais et al. (2020) methodology:
1. Extract salience (Grad-CAM + environmental features) → Parquet
2. Prepare U-space (feature engineering + PCA)
3. Build CHE envelope (Convex Hull Ensemble)
4. Visualize importance surface in U-space

Usage:
    # Run full pipeline
    python scripts/hypervolume_pipeline.py all --checkpoint checkpoints/dino_phase2_best.pt
    
    # Run individual stages
    python scripts/hypervolume_pipeline.py extract --checkpoint checkpoints/dino_phase2_best.pt
    python scripts/hypervolume_pipeline.py prep
    python scripts/hypervolume_pipeline.py che
    python scripts/hypervolume_pipeline.py viz
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def load_config(config_path='configs/hypervolume.yaml'):
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_stage(cmd, stage_name):
    """Run a pipeline stage with error handling"""
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Stage '{stage_name}' failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ Stage '{stage_name}' completed successfully")


def extract(args, config):
    """Stage 1: Extract salience to Parquet"""
    cfg = config['extraction']
    paths = config['paths']
    
    cmd = [
        'python', 'scripts/extract_salience.py',
        '--checkpoint', args.checkpoint,
        '--output', paths['salience'],
        '--batch_size', str(cfg['batch_size']),
        '--importance_threshold', str(cfg['importance_threshold']),
        '--split', cfg['split'],
        '--buffer_size', str(cfg['buffer_size']),
    ]
    
    if args.data_root:
        cmd.extend(['--data_root', args.data_root])
    
    if cfg['num_samples'] > 0:
        cmd.extend(['--num_samples', str(cfg['num_samples'])])
    
    if args.device:
        cmd.extend(['--device', args.device])
    
    run_stage(cmd, 'Extract Salience')


def prep(args, config):
    """Stage 2: Prepare U-space"""
    cfg = config['u_space']
    paths = config['paths']
    
    cmd = [
        'python', 'scripts/prep_u_space.py',
        '--input', paths['salience'],
        '--output', paths['u_space'],
        '--extreme_threshold', str(cfg['extreme_threshold']),
        '--max_components', str(cfg['max_components']),
        '--variance_threshold', str(cfg['variance_threshold']),
    ]
    
    run_stage(cmd, 'Prepare U-Space')


def che(args, config):
    """Stage 3: Build CHE envelope"""
    cfg = config['che']
    paths = config['paths']
    
    cmd = [
        'python', 'scripts/envelope_che.py',
        '--input', paths['u_space'],
        '--output', paths['che'],
        '--n_bootstraps', str(cfg['n_bootstraps']),
        '--occupancy_threshold', str(cfg['occupancy_threshold']),
        '--max_hull_points', str(cfg['max_hull_points']),
        '--grid_resolution', str(cfg['grid_resolution']),
        '--seed', str(cfg['seed']),
    ]
    
    run_stage(cmd, 'CHE Envelope')


def viz(args, config):
    """Stage 4: Visualize results"""
    cfg = config['visualization']
    paths = config['paths']
    
    cmd = [
        'python', 'scripts/viz_report.py',
        '--u_space', paths['u_space'],
        '--che', paths['che'],
        '--output', paths['output'],
        '--grid_resolution', str(cfg['grid_resolution']),
    ]
    
    run_stage(cmd, 'Visualize')


def danger_map(args, config):
    """Stage 5: Create spatial danger map"""
    cfg = config.get('danger_map', {})
    paths = config['paths']
    
    cmd = [
        'python', 'scripts/create_danger_map.py',
        '--u_space', paths['u_space'],
        '--che', paths['che'],
        '--data_root', args.data_root or '~/data/deep_crown_dataset/organized_spreads',
        '--output', paths.get('danger_map', 'data/danger_map'),
        '--wind_speed', str(cfg.get('wind_speed', 5.0)),
        '--wind_direction', str(cfg.get('wind_direction', 180.0)),
        '--chunk_size', str(cfg.get('chunk_size', 100)),
    ]
    
    run_stage(cmd, 'Danger Map')


def all_stages(args, config):
    """Run all pipeline stages"""
    print(f"\n{'#'*70}")
    print(f"# HYPERVOLUME PIPELINE: Full Run")
    print(f"{'#'*70}\n")
    
    extract(args, config)
    prep(args, config)
    che(args, config)
    viz(args, config)
    danger_map(args, config)
    
    print(f"\n{'#'*70}")
    print(f"# ✓ PIPELINE COMPLETE")
    print(f"{'#'*70}")
    print(f"\nResults saved to: {config['paths']['output']}/")
    print(f"CHE data: {config['paths']['che']}/")
    print(f"U-space data: {config['paths']['u_space']}/")
    print(f"Salience data: {config['paths']['salience']}/")
    print(f"Danger map: {config['paths'].get('danger_map', 'data/danger_map')}/")
    print(f"{'#'*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Hypervolume Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/hypervolume_pipeline.py all --checkpoint checkpoints/dino_phase2_best.pt
  
  # Individual stages
  python scripts/hypervolume_pipeline.py extract --checkpoint checkpoints/dino_phase2_best.pt
  python scripts/hypervolume_pipeline.py prep
  python scripts/hypervolume_pipeline.py che
  python scripts/hypervolume_pipeline.py viz
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline stage')
    subparsers.required = True
    
    # Shared arguments
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument('--config', type=str, default='configs/hypervolume.yaml',
                              help='Pipeline configuration file')
    
    # All stages
    parser_all = subparsers.add_parser('all', parents=[shared_parser],
                                       help='Run full pipeline')
    parser_all.add_argument('--checkpoint', type=str, required=True,
                           help='Model checkpoint path')
    parser_all.add_argument('--data_root', type=str, default=None,
                           help='Dataset root directory')
    parser_all.add_argument('--device', type=str, default=None,
                           help='Device (cuda/cpu)')
    
    # Extract stage
    parser_extract = subparsers.add_parser('extract', parents=[shared_parser],
                                          help='Extract salience to Parquet')
    parser_extract.add_argument('--checkpoint', type=str, required=True,
                               help='Model checkpoint path')
    parser_extract.add_argument('--data_root', type=str, default=None,
                               help='Dataset root directory')
    parser_extract.add_argument('--device', type=str, default=None,
                               help='Device (cuda/cpu)')
    
    # Prep stage
    parser_prep = subparsers.add_parser('prep', parents=[shared_parser],
                                       help='Prepare U-space')
    
    # CHE stage
    parser_che = subparsers.add_parser('che', parents=[shared_parser],
                                      help='Build CHE envelope')
    
    # Viz stage
    parser_viz = subparsers.add_parser('viz', parents=[shared_parser],
                                      help='Visualize results')
    
    # Danger map stage
    parser_danger = subparsers.add_parser('danger', parents=[shared_parser],
                                         help='Create spatial danger map')
    parser_danger.add_argument('--data_root', type=str, default=None,
                              help='Dataset root directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Route to appropriate handler
    if args.command == 'all':
        all_stages(args, config)
    elif args.command == 'extract':
        extract(args, config)
    elif args.command == 'prep':
        prep(args, config)
    elif args.command == 'che':
        che(args, config)
    elif args.command == 'viz':
        viz(args, config)
    elif args.command == 'danger':
        danger_map(args, config)


if __name__ == '__main__':
    main()
