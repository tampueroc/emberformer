"""
Grad-CAM and Guided Grad-CAM Analysis for EmberFormer-DINO

Following Pais et al. (2020) methodology for fire spread prediction interpretability.

Usage:
    python scripts/analyze_gradcam.py --checkpoint checkpoints/dino_phase1_best.pt --num_samples 20
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import RawFireDataset
from models.emberformer import EmberFormerDINO


class GradCAM:
    """
    Grad-CAM implementation following Pais et al. (2020)
    
    Computes importance of each pixel in the last convolutional layer
    by evaluating gradients of predicted class with respect to activations.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: EmberFormerDINO model
            target_layer: layer to compute Grad-CAM for (e.g., model.refinement_decoder.output_conv)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, fire_hist, static, wind, valid_t):
        """
        Compute Grad-CAM heatmap
        
        Args:
            fire_hist: [B, T, 1, H, W] fire history
            static: [B, Cs, H, W] static features
            wind: [B, T, 2] wind vectors
            valid_t: [B, T] temporal validity mask
        
        Returns:
            cam: [B, H, W] Grad-CAM heatmap (normalized to [0, 1])
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(fire_hist, static, wind, valid_t)  # [B, 1, H, W]
        
        # Backward pass (sum of all predictions)
        self.model.zero_grad()
        loss = logits.sum()
        loss.backward()
        
        # Compute Grad-CAM weights (alpha_k)
        # Average gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze(1)  # [B, H, W]
        for i in range(cam.shape[0]):
            cam[i] = (cam[i] - cam[i].min()) / (cam[i].max() - cam[i].min() + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        """Clean up hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()


class GuidedBackprop:
    """
    Guided Backpropagation for high-resolution saliency maps
    
    Modifies ReLU backward pass to only propagate positive gradients
    """
    def __init__(self, model):
        self.model = model
        self.gradient = None
        self.forward_relu_outputs = []
        self.handles = []
        
        # Register hooks on all ReLU layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on all ReLU layers in the model"""
        def forward_hook(module, input, output):
            self.forward_relu_outputs.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            # Guided backprop: only pass positive gradients
            forward_output = self.forward_relu_outputs.pop()
            forward_output[forward_output > 0] = 1
            
            # Element-wise multiply with incoming gradient
            positive_grad_output = torch.clamp(grad_output[0], min=0.0)
            new_grad_input = positive_grad_output * forward_output
            
            return (new_grad_input,)
        
        # Find all ReLU layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handle_forward = module.register_forward_hook(forward_hook)
                handle_backward = module.register_full_backward_hook(backward_hook)
                self.handles.append(handle_forward)
                self.handles.append(handle_backward)
    
    def __call__(self, fire_hist, static, wind, valid_t):
        """
        Compute guided backpropagation saliency map
        
        Returns:
            saliency: [B, T, 1, H, W] gradient w.r.t. input fire frames
        """
        # Ensure input requires gradient
        fire_hist = fire_hist.clone().requires_grad_(True)
        
        self.model.eval()
        
        # Forward pass
        logits = self.model(fire_hist, static, wind, valid_t)
        
        # Backward pass
        self.model.zero_grad()
        loss = logits.sum()
        loss.backward()
        
        # Get gradient w.r.t. input
        saliency = fire_hist.grad.abs()  # [B, T, 1, H, W]
        
        return saliency
    
    def remove_hooks(self):
        """Clean up hooks"""
        for handle in self.handles:
            handle.remove()


def compute_guided_gradcam(gradcam, guided_backprop, fire_hist, static, wind, valid_t):
    """
    Combine Grad-CAM and Guided Backpropagation
    
    Returns:
        guided_gradcam: [B, T, H, W] high-resolution saliency per timestep
    """
    # Compute Grad-CAM (coarse, but spatially accurate)
    cam = gradcam(fire_hist, static, wind, valid_t)  # [B, H, W]
    
    # Compute Guided Backprop (fine-grained, but less spatially accurate)
    saliency = guided_backprop(fire_hist, static, wind, valid_t)  # [B, T, 1, H, W]
    saliency = saliency.squeeze(2)  # [B, T, H, W]
    
    # Normalize saliency per timestep
    B, T, H, W = saliency.shape
    for b in range(B):
        for t in range(T):
            s = saliency[b, t]
            saliency[b, t] = (s - s.min()) / (s.max() - s.min() + 1e-8)
    
    # Combine: element-wise multiply Grad-CAM with each timestep's saliency
    # Grad-CAM is shared across timesteps (global importance)
    cam_expanded = cam.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, H, W]
    
    guided_gradcam = cam_expanded * saliency  # [B, T, H, W]
    
    # Normalize again
    for b in range(B):
        for t in range(T):
            gg = guided_gradcam[b, t]
            guided_gradcam[b, t] = (gg - gg.min()) / (gg.max() - gg.min() + 1e-8)
    
    return guided_gradcam


def visualize_gradcam(model, dataset, num_samples=10, output_dir='results/gradcam'):
    """
    Generate Grad-CAM visualizations for validation samples
    
    Args:
        model: trained EmberFormerDINO
        dataset: RawFireDataset
        num_samples: number of samples to visualize
        output_dir: directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, model.refinement_decoder.output_conv)
    
    device = next(model.parameters()).device
    
    for sample_idx in range(min(num_samples, len(dataset))):
        fire_hist, static, wind, target = dataset[sample_idx]
        
        # Add batch dimension
        fire_hist_batch = fire_hist.unsqueeze(0).to(device)
        static_batch = static.unsqueeze(0).to(device)
        wind_batch = wind.unsqueeze(0).to(device)
        
        B, T, C, H, W = fire_hist_batch.shape
        valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
        
        # Compute Grad-CAM
        with torch.enable_grad():
            cam = gradcam(fire_hist_batch, static_batch, wind_batch, valid_t)
        
        cam_np = cam[0].cpu().numpy()  # [H, W]
        
        # Get prediction
        with torch.no_grad():
            logits = model(fire_hist_batch, static_batch, wind_batch, valid_t)
            pred = torch.sigmoid(logits[0, 0]).cpu().numpy()
        
        # Visualize
        fig, axes = plt.subplots(2, T + 2, figsize=(4*(T+2), 8))
        
        # Row 1: Fire history
        for t in range(T):
            fire_t = fire_hist[t, 0].cpu().numpy()
            axes[0, t].imshow(fire_t, cmap='hot', vmin=0, vmax=1)
            axes[0, t].set_title(f't-{T-t-1}', fontsize=10)
            axes[0, t].axis('off')
        
        # Row 1: Target and prediction
        axes[0, T].imshow(target.cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[0, T].set_title('Target', fontsize=10)
        axes[0, T].axis('off')
        
        axes[0, T+1].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[0, T+1].set_title('Prediction', fontsize=10)
        axes[0, T+1].axis('off')
        
        # Row 2: Grad-CAM overlays
        for t in range(T):
            fire_t = fire_hist[t, 0].cpu().numpy()
            axes[1, t].imshow(fire_t, cmap='gray', alpha=0.7)
            im = axes[1, t].imshow(cam_np, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[1, t].set_title(f'Grad-CAM t-{T-t-1}', fontsize=10)
            axes[1, t].axis('off')
        
        # Row 2: Grad-CAM on target and prediction
        axes[1, T].imshow(target.cpu().numpy(), cmap='gray', alpha=0.7)
        axes[1, T].imshow(cam_np, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, T].set_title('Grad-CAM on Target', fontsize=10)
        axes[1, T].axis('off')
        
        axes[1, T+1].imshow(pred, cmap='gray', alpha=0.7)
        im = axes[1, T+1].imshow(cam_np, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, T+1].set_title('Grad-CAM on Pred', fontsize=10)
        axes[1, T+1].axis('off')
        
        # Colorbar
        fig.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Sample {sample_idx}: Grad-CAM Saliency Map', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gradcam_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Grad-CAM for sample {sample_idx}")
    
    gradcam.remove_hooks()
    print(f"\n✓ Grad-CAM analysis complete. Results saved to {output_dir}/")


def visualize_guided_gradcam(model, dataset, num_samples=10, output_dir='results/guided_gradcam'):
    """
    Generate Guided Grad-CAM visualizations
    
    Shows pixel-level importance for each input timestep
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize methods
    gradcam = GradCAM(model, model.refinement_decoder.output_conv)
    guided_backprop = GuidedBackprop(model)
    
    device = next(model.parameters()).device
    
    for sample_idx in range(min(num_samples, len(dataset))):
        fire_hist, static, wind, target = dataset[sample_idx]
        
        # Add batch dimension
        fire_hist_batch = fire_hist.unsqueeze(0).to(device)
        static_batch = static.unsqueeze(0).to(device)
        wind_batch = wind.unsqueeze(0).to(device)
        
        B, T, C, H, W = fire_hist_batch.shape
        valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
        
        # Compute Guided Grad-CAM
        with torch.enable_grad():
            guided_gradcam_maps = compute_guided_gradcam(
                gradcam, guided_backprop,
                fire_hist_batch, static_batch, wind_batch, valid_t
            )  # [B, T, H, W]
        
        gg_np = guided_gradcam_maps[0].cpu().detach().numpy()  # [T, H, W]
        
        # Get prediction
        with torch.no_grad():
            logits = model(fire_hist_batch, static_batch, wind_batch, valid_t)
            pred = torch.sigmoid(logits[0, 0]).cpu().numpy()
        
        # Visualize
        fig, axes = plt.subplots(3, T + 1, figsize=(4*(T+1), 12))
        
        # Row 1: Fire history
        for t in range(T):
            fire_t = fire_hist[t, 0].cpu().numpy()
            axes[0, t].imshow(fire_t, cmap='hot', vmin=0, vmax=1)
            axes[0, t].set_title(f't-{T-t-1}', fontsize=10)
            axes[0, t].axis('off')
        
        axes[0, T].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[0, T].set_title('Prediction', fontsize=10)
        axes[0, T].axis('off')
        
        # Row 2: Guided Grad-CAM per timestep
        for t in range(T):
            im = axes[1, t].imshow(gg_np[t], cmap='jet', vmin=0, vmax=1)
            axes[1, t].set_title(f'Saliency t-{T-t-1}', fontsize=10)
            axes[1, t].axis('off')
        
        # Average saliency
        avg_saliency = gg_np.mean(axis=0)
        im = axes[1, T].imshow(avg_saliency, cmap='jet', vmin=0, vmax=1)
        axes[1, T].set_title('Avg Saliency', fontsize=10)
        axes[1, T].axis('off')
        
        # Row 3: Overlay on fire frames
        for t in range(T):
            fire_t = fire_hist[t, 0].cpu().numpy()
            axes[2, t].imshow(fire_t, cmap='gray', alpha=0.7)
            im = axes[2, t].imshow(gg_np[t], cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[2, t].set_title(f'Overlay t-{T-t-1}', fontsize=10)
            axes[2, t].axis('off')
        
        # Overlay on prediction
        axes[2, T].imshow(pred, cmap='gray', alpha=0.7)
        im = axes[2, T].imshow(avg_saliency, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2, T].set_title('Overlay Pred', fontsize=10)
        axes[2, T].axis('off')
        
        # Colorbar
        fig.colorbar(im, ax=axes[2, -1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Sample {sample_idx}: Guided Grad-CAM (Pixel-Level Saliency)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/guided_gradcam_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Guided Grad-CAM for sample {sample_idx}")
    
    gradcam.remove_hooks()
    guided_backprop.remove_hooks()
    print(f"\n✓ Guided Grad-CAM analysis complete. Results saved to {output_dir}/")


def load_model(checkpoint_path, device='cuda'):
    """Load trained EmberFormerDINO model from checkpoint"""
    # Load config
    config_path = Path('configs/emberformer_dino.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = EmberFormerDINO(
        dino_model=cfg['model']['dino']['model_name'],
        freeze_dino=False,  # Set to False to allow gradient flow for analysis
        d_model=cfg['model']['temporal']['d_model'],
        nhead=cfg['model']['temporal']['nhead'],
        num_layers=cfg['model']['temporal']['num_layers'],
        dim_feedforward=cfg['model']['temporal']['dim_feedforward'],
        dropout=cfg['model']['temporal']['dropout'],
        spatial_hidden=cfg['model']['spatial']['hidden_channels'],
        patch_size=cfg['model']['refinement']['patch_size'],
        static_channels=cfg['data']['static_channels'],
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded Phase {checkpoint.get('phase', 'Unknown')} model from epoch {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Val F1: {checkpoint.get('val_f1', 0.0):.3f}")
    print(f"  Val IoU: {checkpoint.get('val_iou', 0.0):.3f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Analysis for EmberFormer-DINO')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/dino_phase1_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, 
                       default='~/data/deep_crown_dataset/organized_spreads',
                       help='Path to dataset root')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--sequence_length', type=int, default=4,
                       help='Sequence length for temporal history')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to analyze')
    parser.add_argument('--method', type=str, default='both', 
                       choices=['gradcam', 'guided', 'both'],
                       help='Which method to run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    dataset = RawFireDataset(
        args.data_root, 
        sequence_length=args.sequence_length,
        split=args.split
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Run analysis
    if args.method in ['gradcam', 'both']:
        print(f"\n{'='*60}")
        print("Running Grad-CAM Analysis")
        print(f"{'='*60}")
        visualize_gradcam(model, dataset, num_samples=args.num_samples)
    
    if args.method in ['guided', 'both']:
        print(f"\n{'='*60}")
        print("Running Guided Grad-CAM Analysis")
        print(f"{'='*60}")
        visualize_guided_gradcam(model, dataset, num_samples=args.num_samples)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
