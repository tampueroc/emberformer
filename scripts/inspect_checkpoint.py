"""
Utility script to inspect checkpoint contents and metadata
"""
import torch
import argparse
import json

def inspect_checkpoint(path):
    """Print checkpoint contents and metrics"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {path}")
    print('='*60)
    
    checkpoint = torch.load(path, map_location='cpu')
    
    print("\n📊 Metrics:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    print(f"  Val IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
    print(f"  Val Precision: {checkpoint.get('val_prec', 'N/A'):.4f}")
    print(f"  Val Recall: {checkpoint.get('val_rec', 'N/A'):.4f}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    print("\n🔑 Keys in checkpoint:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            num_params = len(checkpoint[key])
            print(f"  - {key}: {num_params} tensors")
        elif key == 'optimizer_state_dict':
            print(f"  - {key}: optimizer state")
        elif key == 'config':
            print(f"  - {key}: configuration dict")
        else:
            print(f"  - {key}: {checkpoint[key]}")
    
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        print("\n⚙️  Model Configuration:")
        if 'model' in cfg:
            tcfg = cfg['model'].get('temporal', {})
            scfg = cfg['model'].get('spatial', {})
            print(f"  Temporal:")
            print(f"    - d_model: {tcfg.get('d_model')}")
            print(f"    - nhead: {tcfg.get('nhead')}")
            print(f"    - num_layers: {tcfg.get('num_layers')}")
            print(f"  Spatial:")
            print(f"    - decoder: {scfg.get('decoder_type')}")
            print(f"    - model_name: {scfg.get('model_name')}")
        
        if 'train' in cfg:
            print(f"  Training:")
            print(f"    - lr_temporal: {cfg['train'].get('lr_temporal')}")
            print(f"    - lr_spatial: {cfg['train'].get('lr_spatial')}")
    
    print("\n" + "="*60)
    
    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect checkpoint file')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--export-json', type=str, help='Export metadata to JSON file')
    args = parser.parse_args()
    
    checkpoint = inspect_checkpoint(args.checkpoint)
    
    if args.export_json:
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'val_loss': float(checkpoint.get('val_loss', 0)),
            'val_f1': float(checkpoint.get('val_f1', 0)),
            'val_iou': float(checkpoint.get('val_iou', 0)),
            'val_prec': float(checkpoint.get('val_prec', 0)),
            'val_rec': float(checkpoint.get('val_rec', 0)),
            'val_acc': float(checkpoint.get('val_acc', 0)),
        }
        
        with open(args.export_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📝 Exported metadata to: {args.export_json}")
