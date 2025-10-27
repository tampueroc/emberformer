# EmberFormer-DINO Interpretability Analysis

**Goal:** Understand how DINO vision features combine with temporal dynamics to predict fire spread, and identify what spatial patterns, temporal dependencies, and terrain-wind interactions drive predictions.

**⚠️ IMPORTANT: Run these analyses AFTER model training converges (val/f1 > 0.65, early stopping triggered)**

**📁 Output:** All analysis results are saved to `results/` directory with organized subdirectories

## Prerequisites

1. ✅ Phase 1 trained to convergence (F1 ~0.66, Precision ~0.59, Recall ~0.74)
2. ✅ Best checkpoint saved: `checkpoints/dino_phase1_best.pt`
3. ✅ Validation dataset available
4. ✅ Optional: Phase 2 checkpoint for comparison

## Model Loading

```python
from models.emberformer import EmberFormerDINO
import torch
import yaml

# Load config
with open('configs/emberformer_dino.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Load checkpoint
checkpoint = torch.load('checkpoints/dino_phase1_best.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = EmberFormerDINO(
    dino_model=cfg['model']['dino']['model_name'],
    freeze_dino=False,  # False to inspect all layers
    d_model=cfg['model']['temporal']['d_model'],
    nhead=cfg['model']['temporal']['nhead'],
    num_layers=cfg['model']['temporal']['num_layers'],
    dim_feedforward=cfg['model']['temporal']['dim_feedforward'],
    dropout=cfg['model']['temporal']['dropout'],
    spatial_hidden=cfg['model']['spatial']['hidden_channels'],
    patch_size=cfg['model']['refinement']['patch_size'],
    static_channels=8,
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Loaded Phase {checkpoint['phase']} model from epoch {checkpoint['epoch']}")
print(f"  Val F1: {checkpoint['val_f1']:.3f}")
print(f"  Val IoU: {checkpoint['val_iou']:.3f}")
```

---

## Architecture Overview

EmberFormer-DINO has three components:

1. **DINO Vision Encoder** (`fire_encoder`):
   - Processes pixel fire frames [B, T, 1, H, W]
   - Frozen in Phase 1, fine-tuned in Phase 2
   - Outputs: [B, T, num_patches, 768] patch features

2. **Temporal Transformer** (`temporal_transformer`):
   - Aggregates temporal sequence of DINO features
   - Learns fire dynamics and dependencies
   - Outputs: [B, d_model, Gy, Gx] spatial features

3. **Spatial Decoder** (`decoder` + `refinement`):
   - Upsamples to full resolution
   - Fuses with static terrain features
   - Outputs: [B, 1, H, W] predictions

---

## Research Questions

### 1. DINO Feature Learning
**Question:** What spatial patterns does DINO extract from fire images?

**Hypotheses:**
- DINO learns fire boundary/edge features
- Attention focuses on active fire fronts
- Features encode local spread patterns

### 2. Temporal Dependencies
**Question:** How does temporal transformer aggregate fire history?

**Hypotheses:**
- Recent frames (t-1, t-2) dominate
- Longer history helps in complex scenarios
- Attention reveals trend vs. snapshot reliance

### 3. Static-Fire Interactions
**Question:** How do terrain features modulate predictions?

**Hypotheses:**
- Steep slopes → uphill spread bias
- Fuel load → spread magnitude
- Wind × terrain → directional effects

### 4. Phase 1 vs Phase 2
**Question:** Does fine-tuning DINO improve representations?

**Hypotheses:**
- Phase 2 learns fire-specific features
- Sharper attention on boundaries
- Better extreme event handling

---

## Analysis 1: Grad-CAM Saliency Maps (Pais et al. 2020)

**Methodology:** Following Pais et al. (2020), we apply Grad-CAM to identify which spatial regions in the model's internal representations drive fire spread predictions.

**Target Layer:** `model.refinement_decoder.output_conv` - the final convolutional layer before pixel-level predictions (406×406 resolution).

### Grad-CAM Theory

Grad-CAM computes class activation maps by:
1. Forward pass: compute predictions `y_c` (fire spread logits)
2. Backward: compute gradients `∂y_c / ∂A^k` where `A^k` are activations at target layer
3. Global average pooling of gradients: `α_k = (1/Z) Σ_i Σ_j (∂y_c / ∂A^k_{ij})`
4. Weighted combination: `L_Grad-CAM = ReLU(Σ_k α_k · A^k)`
5. Upsample to input resolution if needed

### Implementation

Create `scripts/analyze_gradcam.py`:

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data import RawFireDataset
from models.emberformer import EmberFormerDINO

# Ensure results directory exists
Path('results/gradcam').mkdir(parents=True, exist_ok=True)

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


# Usage
if __name__ == '__main__':
    from scripts.train_phase1 import load_model
    
    # Load model
    model = load_model('checkpoints/dino_phase1_best.pt')
    
    # Load validation dataset
    dataset = RawFireDataset('~/data/deep_crown_dataset/organized_spreads', 
                             sequence_length=4, split='val')
    
    # Generate Grad-CAM visualizations
    visualize_gradcam(model, dataset, num_samples=20)
```

**Interpretation Guide:**
- **Hot regions (red/yellow):** High importance - these features strongly drive predictions
- **Cool regions (blue):** Low importance - these features have minimal impact
- Overlay on fire frames shows which spatial locations the model "looks at" when predicting spread

---

## Analysis 2: Guided Grad-CAM (Pais et al. 2020)

**Methodology:** Guided Grad-CAM combines the spatial localization of Grad-CAM with the fine-grained details of Guided Backpropagation to produce high-resolution pixel-level saliency maps.

**Purpose:** Shows exactly which input pixels drive fire spread predictions with pixel-perfect precision.

### Guided Backpropagation Theory

Guided Backpropagation modifies standard backpropagation by only propagating positive gradients through ReLU layers:
- Standard backprop: passes gradients through all activations
- Guided backprop: suppresses negative gradients at ReLU layers
- Result: highlights pixels that contribute positively to the prediction

### Guided Grad-CAM = Grad-CAM ⊙ Guided Backprop

1. Compute Grad-CAM heatmap (coarse, 406×406)
2. Compute Guided Backpropagation (fine, 406×406)
3. Element-wise multiply (Hadamard product)
4. Result: High-resolution saliency with correct spatial localization

### Implementation

Add to `scripts/analyze_gradcam.py`:

```python
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
```

**Interpretation Guide:**
- **Saliency maps show pixel-level importance per timestep**
- Hot pixels (red/yellow) in frame t-i strongly influence predictions
- Compare across timesteps: which historical frames matter most?
- Overlay shows: model focuses on fire boundaries, active fronts, spread direction

---

## Analysis 3: Temporal Importance Analysis

### Implementation

```python
def compute_temporal_importance(model, sample, device='cuda'):
    """
    Measure importance of each timestep in history
    
    Method: Gradient magnitude w.r.t. each timestep
    """
    fire_hist, static, wind, target = sample
    
    fire_hist = fire_hist.unsqueeze(0).to(device).requires_grad_(True)
    static = static.unsqueeze(0).to(device)
    wind = wind.unsqueeze(0).to(device)
    
    B, T, C, H, W = fire_hist.shape
    valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
    
    model.eval()
    logits = model(fire_hist, static, wind, valid_t)
    
    # Backward
    loss = logits.sum()
    loss.backward()
    
    # Average gradient magnitude per timestep
    temporal_importance = fire_hist.grad.abs().mean(dim=(0, 2, 3, 4))  # [T]
    
    return temporal_importance.cpu().numpy()

def analyze_temporal_patterns(model, dataset, num_samples=50):
    """
    Analyze temporal attention patterns across multiple samples
    
    Question: Do recent frames dominate, or is history used?
    """
    results_by_length = {2: [], 3: [], 4: []}
    
    for i in range(num_samples):
        fire_hist, static, wind, target = dataset[i]
        T = fire_hist.shape[-1]
        
        if T not in results_by_length:
            continue
        
        importance = compute_temporal_importance(
            model, (fire_hist, static, wind, target)
        )
        
        results_by_length[T].append(importance)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (T, importance_list) in zip(axes, results_by_length.items()):
        if not importance_list:
            continue
        
        # Average importance across samples
        avg_importance = np.mean(importance_list, axis=0)
        std_importance = np.std(importance_list, axis=0)
        
        timesteps = [f't-{T-i-1}' for i in range(T)]
        
        ax.bar(timesteps, avg_importance, yerr=std_importance, 
               alpha=0.7, capsize=5)
        ax.set_ylabel('Importance (Gradient Magnitude)')
        ax.set_title(f'Temporal Importance (T={T}, n={len(importance_list)})')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight most recent
        ax.axhline(avg_importance.mean(), color='r', 
                  linestyle='--', alpha=0.5, label='Mean')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/temporal_importance_analysis.png', dpi=150)
    plt.close()
    
    # Print statistics
    print("\n=== Temporal Importance Analysis ===")
    for T, importance_list in results_by_length.items():
        if importance_list:
            avg = np.mean(importance_list, axis=0)
            print(f"\nT={T}:")
            for t in range(T):
                print(f"  t-{T-t-1}: {avg[t]:.4f} ({avg[t]/avg.sum()*100:.1f}%)")
            
            # Check recency bias
            recent_ratio = avg[-1] / avg.mean()
            print(f"  Recency bias: {recent_ratio:.2f}x (t-0 vs mean)")
```

---

## Analysis 3: Static Feature Ablation

### Implementation

```python
from tqdm import tqdm
import torchmetrics

def evaluate_model_f1(model, loader, device):
    """Compute F1 score on validation set"""
    f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            fire_hist, static, wind, targets, valid_t = batch
            fire_hist = fire_hist.to(device)
            static = static.to(device)
            wind = wind.to(device)
            targets = targets.to(device)
            valid_t = valid_t.to(device)
            
            logits = model(fire_hist, static, wind, valid_t)
            preds = (torch.sigmoid(logits) > 0.5).int()
            
            mask = torch.ones_like(targets, dtype=torch.bool)
            f1_metric.update(preds[mask].flatten(), targets[mask].flatten().int())
    
    return f1_metric.compute().item()

def static_feature_ablation(model, val_dataset, device, batch_size=16):
    """
    Measure impact of each static terrain channel
    
    Channels (typical): 
        0: elevation
        1: slope  
        2: aspect
        3: fuel_load
        4: vegetation
        5: canopy_height
        6: canopy_density
        7: other
    """
    from torch.utils.data import DataLoader, Subset
    from scripts.train_dino import collate_raw_dino
    
    # Use subset for faster ablation
    val_subset = Subset(val_dataset, list(range(1000)))
    
    static_channels = ['elevation', 'slope', 'aspect', 'fuel_load', 
                      'vegetation', 'canopy_height', 'canopy_density', 'other']
    
    # Baseline performance
    print("Computing baseline F1...")
    baseline_loader = DataLoader(val_subset, batch_size=batch_size, 
                                collate_fn=collate_raw_dino)
    baseline_f1 = evaluate_model_f1(model, baseline_loader, device)
    print(f"  Baseline F1: {baseline_f1:.4f}\n")
    
    importance = {}
    
    for i, channel_name in enumerate(static_channels):
        print(f"Ablating channel {i}: {channel_name}...")
        
        # Create wrapper dataset that zeros channel i
        class AblatedDataset:
            def __init__(self, dataset, channel_idx):
                self.dataset = dataset
                self.channel_idx = channel_idx
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                fire_hist, static, wind, target = self.dataset[idx]
                # Zero out channel
                static[self.channel_idx] = 0.0
                return fire_hist, static, wind, target
        
        ablated_dataset = AblatedDataset(val_subset, i)
        ablated_loader = DataLoader(ablated_dataset, batch_size=batch_size,
                                    collate_fn=collate_raw_dino)
        
        # Evaluate
        f1_without = evaluate_model_f1(model, ablated_loader, device)
        drop = baseline_f1 - f1_without
        importance[channel_name] = drop
        
        print(f"  F1 without: {f1_without:.4f} (drop: {drop:+.4f})\n")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = list(importance.keys())
    drops = list(importance.values())
    colors = ['red' if d > 0 else 'green' for d in drops]
    
    ax.barh(channels, drops, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('F1 Score Drop When Feature Removed')
    ax.set_title('Static Feature Importance (Ablation Study)')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/static_feature_importance.png', dpi=150)
    plt.close()
    
    # Print ranking
    print("\n=== Feature Importance Ranking ===")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (feature, drop) in enumerate(sorted_features, 1):
        print(f"{rank}. {feature:20s}: {drop:+.4f}")
    
    return importance
```

---

## Analysis 4: Wind Direction Analysis

### Implementation

```python
def analyze_wind_direction_impact(model, dataset, num_samples=100, device='cuda'):
    """
    Test if model learns wind-driven spread correctly
    
    Method: Measure directional alignment of predictions with wind
    """
    results = []
    
    for i in range(num_samples):
        fire_hist, static, wind, target = dataset[i]
        
        # Prepare inputs
        fire_hist_batch = fire_hist.unsqueeze(0).to(device)
        static_batch = static.unsqueeze(0).to(device)
        wind_batch = wind.unsqueeze(0).to(device)
        
        B, T = fire_hist_batch.shape[:2]
        valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(fire_hist_batch, static_batch, wind_batch, valid_t)
            pred = torch.sigmoid(pred)[0, 0].cpu()  # [H, W]
        
        # Wind at last timestep
        wind_speed = wind[-1, 0].item()
        wind_dir = wind[-1, 1].item()  # Radians
        
        # Analyze spread direction
        fire_current = fire_hist[0, :, :, -1] > 0.5
        fire_pred = pred > 0.5
        new_fire = fire_pred & ~fire_current
        
        if new_fire.sum() > 10 and fire_current.sum() > 0:
            # Compute fire center
            y_fire, x_fire = torch.where(fire_current)
            cy, cx = y_fire.float().mean(), x_fire.float().mean()
            
            # Vector from center to new fire
            y_new, x_new = torch.where(new_fire)
            dy = y_new.float() - cy
            dx = x_new.float() - cx
            
            # Angle of each spread pixel
            spread_angles = torch.atan2(dy, dx)
            
            # Alignment with wind (lower = better)
            angle_diff = torch.abs((spread_angles - wind_dir + np.pi) % (2*np.pi) - np.pi)
            
            results.append({
                'wind_speed': wind_speed,
                'wind_dir': wind_dir,
                'mean_alignment': angle_diff.mean().item(),
                'spread_area': new_fire.sum().item(),
            })
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wind speed vs alignment
    axes[0].scatter(df['wind_speed'], df['mean_alignment'], alpha=0.5)
    axes[0].set_xlabel('Wind Speed')
    axes[0].set_ylabel('Spread-Wind Alignment (rad)')
    axes[0].set_title('Wind Speed vs Directional Alignment')
    axes[0].axhline(np.pi/2, color='red', linestyle='--', label='Perpendicular')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Polar histogram of wind directions
    ax_polar = plt.subplot(122, projection='polar')
    wind_bins = np.linspace(-np.pi, np.pi, 9)
    alignment_by_dir = []
    
    for i in range(len(wind_bins)-1):
        mask = (df['wind_dir'] >= wind_bins[i]) & (df['wind_dir'] < wind_bins[i+1])
        if mask.sum() > 0:
            alignment_by_dir.append(df[mask]['mean_alignment'].mean())
        else:
            alignment_by_dir.append(0)
    
    theta = (wind_bins[:-1] + wind_bins[1:]) / 2
    ax_polar.bar(theta, alignment_by_dir, width=2*np.pi/8, alpha=0.7)
    ax_polar.set_title('Spread Alignment by Wind Direction')
    
    plt.tight_layout()
    plt.savefig('results/wind_direction_analysis.png', dpi=150)
    plt.close()
    
    # Statistics
    print("\n=== Wind-Fire Alignment Analysis ===")
    print(f"Mean alignment: {df['mean_alignment'].mean():.3f} rad")
    print(f"  < π/4 (45°) = well-aligned: {(df['mean_alignment'] < np.pi/4).mean()*100:.1f}%")
    print(f"  > π/2 (90°) = perpendicular: {(df['mean_alignment'] > np.pi/2).mean()*100:.1f}%")
    print(f"Correlation (wind_speed, alignment): {df['wind_speed'].corr(df['mean_alignment']):.3f}")
    
    return df
```

---

## Analysis 3: DINO Attention Visualization

### Extract DINO Self-Attention Maps

```python
def extract_dino_attention(model, sample, device='cuda', layer_idx=-1):
    """
    Extract attention maps from DINO vision transformer
    
    Args:
        layer_idx: Which layer to visualize (-1 = last layer)
    
    Returns:
        attention_maps: [B, T, num_heads, num_patches, num_patches]
    """
    fire_hist, static, wind, target = sample
    
    B = 1
    fire_hist = fire_hist.unsqueeze(0).to(device)
    T, C, H, W = fire_hist.shape[1:]
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # DINO uses scaled_dot_product_attention
        # Access attention weights if available
        if hasattr(module, 'attn_drop'):
            attention_weights.append(output)
    
    # Register hook on DINO attention layers
    target_layer = model.fire_encoder.blocks[layer_idx].attn
    handle = target_layer.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        # Forward through DINO encoder only
        fire_features = model.fire_encoder(fire_hist.reshape(B*T, C, H, W))
        # fire_features: [B*T, num_patches+1, 768]
    
    handle.remove()
    
    # Process attention if captured
    if attention_weights:
        attn = attention_weights[0]  # [B*T, num_heads, num_patches+1, num_patches+1]
        attn = attn.reshape(B, T, *attn.shape[1:])
        return attn
    else:
        print("⚠️  Attention weights not captured. DINO may not expose attention.")
        return None

def visualize_dino_attention_rollout(model, sample, output_path, device='cuda'):
    """
    Visualize attention rollout from DINO CLS token
    
    Shows: Which spatial regions DINO focuses on for fire features
    """
    fire_hist, static, wind, target = sample
    T = fire_hist.shape[-1]
    
    # Extract attention from last layer
    attn = extract_dino_attention(model, sample, device, layer_idx=-1)
    
    if attn is None:
        # Fallback: use gradient-based importance
        print("Using gradient-based spatial importance instead...")
        importance = compute_spatial_importance_map(model, sample, device)
        
        fig, axes = plt.subplots(2, T, figsize=(4*T, 8))
        for t in range(T):
            axes[0, t].imshow(fire_hist[0, :, :, t].numpy(), cmap='hot')
            axes[0, t].set_title(f'Fire t-{T-t-1}')
            axes[0, t].axis('off')
            
            axes[1, t].imshow(importance[t].numpy(), cmap='viridis')
            axes[1, t].set_title(f'Importance t-{T-t-1}')
            axes[1, t].axis('off')
        
        plt.suptitle('DINO Spatial Importance (Gradient-Based)')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Attention rollout: average over heads, use CLS token attention
    # attn: [B, T, num_heads, num_patches+1, num_patches+1]
    B, T, num_heads, num_patches_plus1, _ = attn.shape
    
    # Average over heads
    attn_avg = attn.mean(dim=2)  # [B, T, num_patches+1, num_patches+1]
    
    # Extract CLS token attention to patches (row 0, cols 1:)
    cls_attn = attn_avg[0, :, 0, 1:]  # [T, num_patches]
    
    # Reshape to spatial grid
    patch_size = 14  # DINO default
    num_patches_per_dim = int(np.sqrt(cls_attn.shape[1]))
    cls_attn_spatial = cls_attn.reshape(T, num_patches_per_dim, num_patches_per_dim)
    
    # Upsample to image size
    import torch.nn.functional as F
    cls_attn_upsampled = F.interpolate(
        cls_attn_spatial.unsqueeze(1),  # [T, 1, H_patch, W_patch]
        size=(fire_hist.shape[1], fire_hist.shape[2]),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # [T, H, W]
    
    # Visualize
    fig, axes = plt.subplots(2, T, figsize=(4*T, 8))
    
    for t in range(T):
        # Original fire
        axes[0, t].imshow(fire_hist[0, :, :, t].numpy(), cmap='hot')
        axes[0, t].set_title(f'Fire t-{T-t-1}')
        axes[0, t].axis('off')
        
        # DINO attention overlay
        axes[1, t].imshow(fire_hist[0, :, :, t].numpy(), cmap='gray', alpha=0.5)
        im = axes[1, t].imshow(cls_attn_upsampled[t].cpu().numpy(), 
                               cmap='jet', alpha=0.6)
        axes[1, t].set_title(f'DINO Attention t-{T-t-1}')
        axes[1, t].axis('off')
    
    plt.colorbar(im, ax=axes[1, -1])
    plt.suptitle('DINO Vision Transformer Attention (CLS Token)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved DINO attention visualization to {output_path}")

def compare_dino_features_phase1_vs_phase2(model_phase1, model_phase2, sample, device='cuda'):
    """
    Compare DINO embeddings: frozen (Phase 1) vs fine-tuned (Phase 2)
    
    Question: Does fine-tuning make features more fire-specific?
    """
    fire_hist, static, wind, target = sample
    
    B = 1
    fire_hist = fire_hist.unsqueeze(0).to(device)
    T, C, H, W = fire_hist.shape[1:]
    
    # Extract features from both models
    model_phase1.eval()
    model_phase2.eval()
    
    with torch.no_grad():
        # Phase 1 (frozen)
        feat_p1 = model_phase1.fire_encoder(fire_hist.reshape(B*T, C, H, W))
        # [B*T, num_patches+1, 768]
        
        # Phase 2 (fine-tuned)
        feat_p2 = model_phase2.fire_encoder(fire_hist.reshape(B*T, C, H, W))
    
    # Use CLS token embeddings
    cls_p1 = feat_p1[:, 0, :]  # [B*T, 768]
    cls_p2 = feat_p2[:, 0, :]
    
    # Compute cosine similarity between timesteps
    from torch.nn.functional import cosine_similarity
    
    sim_p1 = torch.zeros(T, T)
    sim_p2 = torch.zeros(T, T)
    
    for i in range(T):
        for j in range(T):
            sim_p1[i, j] = cosine_similarity(cls_p1[i:i+1], cls_p1[j:j+1])
            sim_p2[i, j] = cosine_similarity(cls_p2[i:i+1], cls_p2[j:j+1])
    
    # Visualize similarity matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(sim_p1.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Phase 1 (Frozen DINO)\nCLS Token Similarity')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Timestep')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(sim_p2.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Phase 2 (Fine-tuned DINO)\nCLS Token Similarity')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Timestep')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('results/dino_phase_comparison_similarity.png', dpi=150)
    plt.close()
    
    print("\n=== DINO Feature Comparison ===")
    print(f"Phase 1 avg similarity (off-diagonal): {sim_p1[~torch.eye(T, dtype=bool)].mean():.3f}")
    print(f"Phase 2 avg similarity (off-diagonal): {sim_p2[~torch.eye(T, dtype=bool)].mean():.3f}")
    print("Lower similarity → more discriminative features")

def visualize_dino_patch_embeddings(model, dataset, num_samples=100, device='cuda'):
    """
    t-SNE visualization of DINO patch embeddings
    
    Question: Do fire patches cluster separately from non-fire?
    """
    from sklearn.manifold import TSNE
    
    all_embeddings = []
    all_labels = []  # 0=non-fire, 1=fire
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            fire_hist, static, wind, target = dataset[i]
            
            fire_hist = fire_hist.unsqueeze(0).to(device)
            B, T, C, H, W = fire_hist.shape
            
            # Get DINO features
            features = model.fire_encoder(fire_hist.reshape(B*T, C, H, W))
            # [B*T, num_patches+1, 768]
            
            # Exclude CLS token, keep patch tokens
            patch_features = features[:, 1:, :]  # [B*T, num_patches, 768]
            
            # Determine which patches contain fire
            # Downsample fire mask to patch resolution
            patch_size = 14
            num_patches_per_dim = H // patch_size
            
            for t in range(T):
                fire_mask_t = fire_hist[0, t, 0].cpu()
                fire_mask_patches = F.avg_pool2d(
                    fire_mask_t.unsqueeze(0).unsqueeze(0),
                    kernel_size=patch_size,
                    stride=patch_size
                ).squeeze()  # [num_patches_per_dim, num_patches_per_dim]
                
                fire_mask_flat = (fire_mask_patches > 0.5).flatten()  # [num_patches]
                
                # Collect embeddings and labels
                patch_emb_t = patch_features[t].cpu()  # [num_patches, 768]
                
                all_embeddings.append(patch_emb_t)
                all_labels.append(fire_mask_flat)
    
    # Concatenate all patches
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # [N_total, 768]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # [N_total]
    
    print(f"Computing t-SNE for {all_embeddings.shape[0]} patches...")
    
    # Subsample for speed
    n_viz = min(10000, all_embeddings.shape[0])
    indices = np.random.choice(all_embeddings.shape[0], n_viz, replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings[indices])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fire patches
    fire_mask = all_labels[indices] == 1
    ax.scatter(embeddings_2d[~fire_mask, 0], embeddings_2d[~fire_mask, 1],
              c='gray', alpha=0.3, s=5, label='Non-fire')
    ax.scatter(embeddings_2d[fire_mask, 0], embeddings_2d[fire_mask, 1],
              c='red', alpha=0.5, s=10, label='Fire')
    
    ax.set_title('t-SNE of DINO Patch Embeddings', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dino_patch_embeddings_tsne.png', dpi=150)
    plt.close()
    
    print("✓ Saved t-SNE visualization")
    
    # Compute separation metric
    from sklearn.metrics import silhouette_score
    if len(np.unique(all_labels[indices])) > 1:
        sil_score = silhouette_score(embeddings_2d, all_labels[indices])
        print(f"Silhouette score (fire vs non-fire): {sil_score:.3f}")
        print("  >0.5 = well-separated, <0.25 = overlapping")
```

---

## Analysis 4: Temporal Transformer Attention

### Extract Temporal Attention Weights

```python
def extract_temporal_attention(model, sample, device='cuda'):
    """
    Extract attention weights from temporal transformer
    
    Shows: Which past timesteps the model attends to
    """
    fire_hist, static, wind, target = sample
    
    B = 1
    fire_hist = fire_hist.unsqueeze(0).to(device)
    static = static.unsqueeze(0).to(device)
    wind = wind.unsqueeze(0).to(device)
    T = fire_hist.shape[1]
    valid_t = torch.ones(B, T, dtype=torch.bool, device=device)
    
    # Hook to capture attention
    attention_maps = []
    
    def attention_hook(module, input, output):
        # For nn.TransformerEncoder, we need to hook MultiheadAttention
        # Store attention weights
        if hasattr(module, 'in_proj_weight'):
            # This is a MultiheadAttention layer
            attention_maps.append(output[1] if len(output) > 1 else None)
    
    # Register hooks on transformer layers
    handles = []
    for layer in model.temporal_transformer.transformer.layers:
        handle = layer.self_attn.register_forward_hook(attention_hook)
        handles.append(handle)
    
    model.eval()
    with torch.no_grad():
        # Need to modify model to return attention weights
        # Alternative: manually forward through temporal transformer
        
        # Encode fire history with DINO
        fire_features = model.fire_encoder(fire_hist.reshape(B*T, 1, 406, 406))
        # [B*T, num_patches+1, 768]
        
        # Project to d_model
        fire_tokens = model.fire_proj(fire_features[:, 0, :])  # Use CLS token
        # [B*T, d_model]
        
        fire_tokens = fire_tokens.reshape(B, T, -1)  # [B, T, d_model]
        
        # Add positional encoding
        fire_tokens = model.temporal_pos_enc(fire_tokens)
        
        # Prepare for transformer (expects [T, B, d_model])
        fire_tokens = fire_tokens.permute(1, 0, 2)  # [T, B, d_model]
        
        # Create attention mask for valid timesteps
        src_key_padding_mask = ~valid_t  # [B, T]
        
        # Forward through transformer (with attention return)
        # NOTE: nn.TransformerEncoder doesn't return attention by default
        # Need custom implementation or use average_attn_weights=False
        output = model.temporal_transformer.transformer(
            fire_tokens,
            src_key_padding_mask=src_key_padding_mask
        )
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    if not attention_maps or attention_maps[0] is None:
        print("⚠️  Temporal attention not captured.")
        print("    nn.TransformerEncoder needs modification to return attention.")
        print("    See: https://github.com/pytorch/pytorch/issues/32590")
        return None
    
    return attention_maps

def visualize_temporal_attention_heatmap(model, dataset, num_samples=20):
    """
    Visualize temporal attention patterns averaged over samples
    
    Shows: Which past timesteps are most important
    """
    # Since extracting attention from nn.TransformerEncoder is complex,
    # use gradient-based temporal importance instead
    
    print("Computing gradient-based temporal importance...")
    
    all_importance = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        importance = compute_temporal_importance(model, sample)
        all_importance.append(importance)
    
    # Group by sequence length
    from collections import defaultdict
    importance_by_T = defaultdict(list)
    
    for i in range(min(num_samples, len(dataset))):
        T = dataset[i][0].shape[-1]
        importance_by_T[T].append(all_importance[i])
    
    # Plot heatmap for each T
    fig, axes = plt.subplots(1, len(importance_by_T), figsize=(4*len(importance_by_T), 6))
    if len(importance_by_T) == 1:
        axes = [axes]
    
    for ax, (T, imp_list) in zip(axes, sorted(importance_by_T.items())):
        # Stack into matrix [num_samples, T]
        imp_matrix = np.stack(imp_list)
        
        # Plot heatmap
        im = ax.imshow(imp_matrix, cmap='hot', aspect='auto')
        ax.set_title(f'T={T} ({len(imp_list)} samples)')
        ax.set_xlabel('Timestep (0=oldest)')
        ax.set_ylabel('Sample')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Temporal Importance Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/temporal_attention_heatmap.png', dpi=150)
    plt.close()
    
    # Statistics
    print("\n=== Temporal Attention Statistics ===")
    for T, imp_list in sorted(importance_by_T.items()):
        imp_avg = np.mean(imp_list, axis=0)
        recency_bias = imp_avg[-1] / imp_avg.mean()
        print(f"T={T}: Recency bias = {recency_bias:.2f}x")
        print(f"       Importance: " + " ".join([f"{v:.3f}" for v in imp_avg]))
```

---

## Analysis 5: Phase 1 vs Phase 2 Comparison

### Comprehensive Phase Comparison Framework

```python
def compare_phases_comprehensive(model_p1, model_p2, val_dataset, device='cuda'):
    """
    Side-by-side comparison of Phase 1 vs Phase 2 models
    
    Metrics:
        - Prediction accuracy (F1, IoU, Precision, Recall)
        - Feature quality (DINO embeddings)
        - Attention patterns
        - Extreme event handling
    """
    from tqdm import tqdm
    import torchmetrics
    
    results_p1 = {'f1': [], 'iou': [], 'precision': [], 'recall': []}
    results_p2 = {'f1': [], 'iou': [], 'precision': [], 'recall': []}
    
    model_p1.eval()
    model_p2.eval()
    
    print("Evaluating Phase 1 and Phase 2 models...")
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset))):
            fire_hist, static, wind, target = val_dataset[i]
            
            # Prepare inputs
            fire_hist_b = fire_hist.unsqueeze(0).to(device)
            static_b = static.unsqueeze(0).to(device)
            wind_b = wind.unsqueeze(0).to(device)
            target_b = target.unsqueeze(0).to(device)
            T = fire_hist.shape[-1]
            valid_t = torch.ones(1, T, dtype=torch.bool, device=device)
            
            # Phase 1 prediction
            logits_p1 = model_p1(fire_hist_b, static_b, wind_b, valid_t)
            pred_p1 = (torch.sigmoid(logits_p1) > 0.5).float()
            
            # Phase 2 prediction
            logits_p2 = model_p2(fire_hist_b, static_b, wind_b, valid_t)
            pred_p2 = (torch.sigmoid(logits_p2) > 0.5).float()
            
            # Compute metrics
            for results, pred in [(results_p1, pred_p1), (results_p2, pred_p2)]:
                f1 = torchmetrics.functional.f1_score(
                    pred.int(), target_b.int(), task='binary'
                ).item()
                iou = torchmetrics.functional.jaccard_index(
                    pred.int(), target_b.int(), task='binary'
                ).item()
                precision = torchmetrics.functional.precision(
                    pred.int(), target_b.int(), task='binary'
                ).item()
                recall = torchmetrics.functional.recall(
                    pred.int(), target_b.int(), task='binary'
                ).item()
                
                results['f1'].append(f1)
                results['iou'].append(iou)
                results['precision'].append(precision)
                results['recall'].append(recall)
    
    # Aggregate results
    print("\n" + "="*60)
    print("PHASE COMPARISON RESULTS")
    print("="*60)
    
    for metric in ['f1', 'iou', 'precision', 'recall']:
        p1_mean = np.mean(results_p1[metric])
        p1_std = np.std(results_p1[metric])
        p2_mean = np.mean(results_p2[metric])
        p2_std = np.std(results_p2[metric])
        improvement = ((p2_mean - p1_mean) / p1_mean) * 100
        
        print(f"\n{metric.upper()}:")
        print(f"  Phase 1: {p1_mean:.4f} ± {p1_std:.4f}")
        print(f"  Phase 2: {p2_mean:.4f} ± {p2_std:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    print("="*60)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['f1', 'iou', 'precision', 'recall']
    
    for ax, metric in zip(axes.flat, metrics):
        data = [results_p1[metric], results_p2[metric]]
        ax.boxplot(data, labels=['Phase 1', 'Phase 2'])
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Distribution')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Phase 1 vs Phase 2: Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/phase_comparison_metrics.png', dpi=150)
    plt.close()
    
    return results_p1, results_p2

def analyze_phase_improvement_by_difficulty(model_p1, model_p2, val_dataset, device='cuda'):
    """
    Analyze where Phase 2 improvements come from
    
    Stratify by:
        - Sequence length (short vs long)
        - Spread magnitude (small vs large fires)
        - Wind speed (calm vs windy)
    """
    from collections import defaultdict
    
    improvements = defaultdict(list)
    
    model_p1.eval()
    model_p2.eval()
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            fire_hist, static, wind, target = val_dataset[i]
            
            T = fire_hist.shape[-1]
            wind_speed = wind[-1, 0].item()
            spread_area = target.sum().item()
            
            # Prepare inputs
            fire_hist_b = fire_hist.unsqueeze(0).to(device)
            static_b = static.unsqueeze(0).to(device)
            wind_b = wind.unsqueeze(0).to(device)
            target_b = target.unsqueeze(0).to(device)
            valid_t = torch.ones(1, T, dtype=torch.bool, device=device)
            
            # Get F1 scores
            logits_p1 = model_p1(fire_hist_b, static_b, wind_b, valid_t)
            pred_p1 = (torch.sigmoid(logits_p1) > 0.5).float()
            f1_p1 = torchmetrics.functional.f1_score(
                pred_p1.int(), target_b.int(), task='binary'
            ).item()
            
            logits_p2 = model_p2(fire_hist_b, static_b, wind_b, valid_t)
            pred_p2 = (torch.sigmoid(logits_p2) > 0.5).float()
            f1_p2 = torchmetrics.functional.f1_score(
                pred_p2.int(), target_b.int(), task='binary'
            ).item()
            
            improvement = f1_p2 - f1_p1
            
            # Categorize
            T_cat = 'short' if T <= 3 else 'long'
            wind_cat = 'calm' if wind_speed < 5 else 'windy'
            spread_cat = 'small' if spread_area < 1000 else 'large'
            
            improvements[f'T_{T_cat}'].append(improvement)
            improvements[f'wind_{wind_cat}'].append(improvement)
            improvements[f'spread_{spread_cat}'].append(improvement)
    
    # Print analysis
    print("\n=== Phase 2 Improvement by Scenario ===")
    for category, imp_list in sorted(improvements.items()):
        avg_imp = np.mean(imp_list)
        print(f"{category:20s}: {avg_imp:+.4f} F1 (n={len(imp_list)})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(improvements.keys())
    means = [np.mean(improvements[c]) for c in categories]
    
    bars = ax.barh(categories, means, color=['green' if m > 0 else 'red' for m in means])
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('F1 Improvement (Phase 2 - Phase 1)')
    ax.set_title('Phase 2 Improvements by Scenario Type', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/phase_improvement_by_scenario.png', dpi=150)
    plt.close()
```

---

## Analysis 6: Patch-Level Prediction Analysis

### Analyze Spatial Prediction Accuracy

```python
def analyze_patch_level_accuracy(model, dataset, patch_size=14, num_samples=50, device='cuda'):
    """
    Measure prediction accuracy at patch level
    
    Question: Which spatial regions does the model predict well vs poorly?
    """
    from torch.nn.functional import avg_pool2d
    
    patch_correct = []
    patch_fire_density = []
    
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            fire_hist, static, wind, target = dataset[i]
            
            # Predict
            fire_hist_b = fire_hist.unsqueeze(0).to(device)
            static_b = static.unsqueeze(0).to(device)
            wind_b = wind.unsqueeze(0).to(device)
            T = fire_hist.shape[-1]
            valid_t = torch.ones(1, T, dtype=torch.bool, device=device)
            
            logits = model(fire_hist_b, static_b, wind_b, valid_t)
            pred = (torch.sigmoid(logits[0, 0]) > 0.5).float().cpu()
            target_cpu = target[0].cpu()
            
            # Downsample to patches
            H, W = pred.shape
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            
            pred_patches = avg_pool2d(
                pred.unsqueeze(0).unsqueeze(0),
                kernel_size=patch_size,
                stride=patch_size
            ).squeeze()  # [num_patches_h, num_patches_w]
            
            target_patches = avg_pool2d(
                target_cpu.unsqueeze(0),
                kernel_size=patch_size,
                stride=patch_size
            ).squeeze()
            
            # Compute accuracy per patch
            correct = ((pred_patches > 0.5) == (target_patches > 0.5)).float()
            fire_density = target_patches
            
            patch_correct.append(correct.flatten())
            patch_fire_density.append(fire_density.flatten())
    
    # Aggregate
    patch_correct = torch.cat(patch_correct)  # [N_patches_total]
    patch_fire_density = torch.cat(patch_fire_density)
    
    # Bin by fire density
    density_bins = torch.linspace(0, 1, 11)
    accuracy_by_density = []
    
    for i in range(len(density_bins) - 1):
        mask = (patch_fire_density >= density_bins[i]) & (patch_fire_density < density_bins[i+1])
        if mask.sum() > 0:
            acc = patch_correct[mask].mean().item()
            accuracy_by_density.append(acc)
        else:
            accuracy_by_density.append(np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (density_bins[:-1] + density_bins[1:]) / 2
    ax.plot(bin_centers.numpy(), accuracy_by_density, marker='o', linewidth=2)
    ax.set_xlabel('Fire Density in Patch', fontsize=12)
    ax.set_ylabel('Prediction Accuracy', fontsize=12)
    ax.set_title('Patch-Level Prediction Accuracy vs Fire Density', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('results/patch_level_accuracy.png', dpi=150)
    plt.close()
    
    print("\n=== Patch-Level Analysis ===")
    print(f"Overall patch accuracy: {patch_correct.mean():.3f}")
    print(f"Accuracy on non-fire patches: {patch_correct[patch_fire_density < 0.1].mean():.3f}")
    print(f"Accuracy on fire patches: {patch_correct[patch_fire_density > 0.5].mean():.3f}")

def visualize_prediction_confidence_map(model, sample, output_path, device='cuda'):
    """
    Visualize model prediction confidence spatially
    
    Shows: Where is the model confident vs uncertain?
    """
    fire_hist, static, wind, target = sample
    
    # Predict
    fire_hist_b = fire_hist.unsqueeze(0).to(device)
    static_b = static.unsqueeze(0).to(device)
    wind_b = wind.unsqueeze(0).to(device)
    T = fire_hist.shape[-1]
    valid_t = torch.ones(1, T, dtype=torch.bool, device=device)
    
    model.eval()
    with torch.no_grad():
        logits = model(fire_hist_b, static_b, wind_b, valid_t)
        confidence = torch.sigmoid(logits[0, 0]).cpu()  # [H, W]
    
    # Compute uncertainty (entropy-like measure)
    uncertainty = -confidence * torch.log(confidence + 1e-8) - (1-confidence) * torch.log(1-confidence + 1e-8)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(fire_hist[0, :, :, -1].numpy(), cmap='hot')
    axes[0, 0].set_title('Fire at t')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target[0].numpy(), cmap='hot')
    axes[0, 1].set_title('Actual t+1')
    axes[0, 1].axis('off')
    
    im2 = axes[0, 2].imshow(confidence.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Predicted Confidence')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2
    pred_binary = (confidence > 0.5).float()
    axes[1, 0].imshow(pred_binary.numpy(), cmap='hot')
    axes[1, 0].set_title('Predicted t+1 (Binary)')
    axes[1, 0].axis('off')
    
    im4 = axes[1, 1].imshow(uncertainty.numpy(), cmap='viridis')
    axes[1, 1].set_title('Prediction Uncertainty')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Error map
    error = pred_binary - target[0]
    im5 = axes[1, 2].imshow(error.numpy(), cmap='RdYlGn_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('Error (Red=FP, Green=FN)')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.suptitle('Prediction Confidence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Analysis 7: Extreme Event Analysis

### Implementation

```python
def identify_extreme_events(dataset, percentile=95):
    """Find samples with extreme fire spread"""
    spread_areas = []
    
    for i in range(len(dataset)):
        fire_hist, static, wind, target = dataset[i]
        T = fire_hist.shape[-1]
        
        fire_t = fire_hist[0, :, :, T-1].sum().item()
        fire_t1 = target.sum().item()
        spread = fire_t1 - fire_t
        
        spread_areas.append((i, spread))
    
    spread_areas.sort(key=lambda x: x[1], reverse=True)
    
    threshold_idx = int(len(spread_areas) * (1 - percentile/100))
    extreme_indices = [idx for idx, _ in spread_areas[:threshold_idx]]
    
    print(f"Identified {len(extreme_indices)} extreme events (>{percentile}th percentile)")
    print(f"  Spread range: {spread_areas[0][1]:.0f} to {spread_areas[threshold_idx][1]:.0f} pixels")
    
    return extreme_indices, spread_areas

def visualize_extreme_event(model, sample, sample_idx, output_path, device='cuda'):
    """Detailed visualization of an extreme event"""
    fire_hist, static, wind, target = sample
    T = fire_hist.shape[-1]
    
    # Get prediction
    fire_hist_batch = fire_hist.unsqueeze(0).to(device)
    static_batch = static.unsqueeze(0).to(device)
    wind_batch = wind.unsqueeze(0).to(device)
    valid_t = torch.ones(1, T, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        logits = model(fire_hist_batch, static_batch, wind_batch, valid_t)
        pred = torch.sigmoid(logits)[0, 0].cpu()
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Fire history
    for i in range(min(T, 4)):
        ax = fig.add_subplot(gs[0, i])
        fire_frame = fire_hist[0, :, :, i].numpy() if i < T else np.zeros((406, 406))
        ax.imshow(fire_frame, cmap='hot')
        ax.set_title(f'Fire t-{T-i-1}')
        ax.axis('off')
    
    # Row 2: Landscape features
    terrain_names = ['Fuels', 'Arqueo', 'Canopy Bulk Density', 'Canopy Base Height', 
                     'Elevation', 'Flora', 'Paleo', 'Urban']
    for i in range(min(8, static.shape[0])):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(static[i].numpy(), cmap='terrain')
        ax.set_title(terrain_names[i] if i < len(terrain_names) else f'Static {i}')
        ax.axis('off')
    
    # Row 3: Predictions
    ax_last = fig.add_subplot(gs[2, 0])
    ax_last.imshow(fire_hist[0, :, :, -1].numpy(), cmap='hot')
    ax_last.set_title('Fire at t')
    ax_last.axis('off')
    
    ax_pred = fig.add_subplot(gs[2, 1])
    ax_pred.imshow(pred.numpy(), cmap='hot', vmin=0, vmax=1)
    ax_pred.set_title('Predicted t+1')
    ax_pred.axis('off')
    
    ax_true = fig.add_subplot(gs[2, 2])
    ax_true.imshow(target[0].numpy(), cmap='hot')
    ax_true.set_title('Actual t+1')
    ax_true.axis('off')
    
    ax_error = fig.add_subplot(gs[2, 3])
    error_map = (pred > 0.5).float() - target[0]
    ax_error.imshow(error_map.numpy(), cmap='RdYlGn_r', vmin=-1, vmax=1)
    ax_error.set_title('Error (Red=FP, Green=FN)')
    ax_error.axis('off')
    
    # Add wind arrow
    wind_t = wind[-1].numpy()
    ax_last.arrow(30, 30, wind_t[0]*50, wind_t[1]*50, 
                 color='cyan', width=3, head_width=10)
    
    plt.suptitle(f'Extreme Event #{sample_idx} | T={T} | Wind={wind_t}', fontsize=14)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Complete Analysis Script

Create `scripts/run_interpretability_dino.py`:

```python
#!/usr/bin/env python
"""
Complete interpretability analysis for EmberFormer-DINO

Usage:
    python scripts/run_interpretability_dino.py --checkpoint checkpoints/dino_phase1_best.pt
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--num-samples', type=int, default=100)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("EmberFormer-DINO Interpretability Analysis")
    print("=" * 60)
    
    # Load model and data
    print("\n1. Loading model and data...")
    # ... implementation ...
    
    # Analysis 1: Spatial importance
    print("\n2. Analyzing spatial importance...")
    visualize_spatial_importance(model, dataset, num_samples=10,
                                 output_dir=output_dir / 'spatial')
    
    # Analysis 2: Temporal importance
    print("\n3. Analyzing temporal patterns...")
    analyze_temporal_patterns(model, dataset, num_samples=args.num_samples)
    
    # Analysis 3: Feature ablation
    print("\n4. Running feature ablation...")
    importance = static_feature_ablation(model, dataset, device)
    
    # Analysis 4: Wind direction
    print("\n5. Analyzing wind-fire alignment...")
    wind_df = analyze_wind_direction_impact(model, dataset, num_samples=args.num_samples)
    
    # Analysis 5: Extreme events
    print("\n6. Analyzing extreme events...")
    extreme_indices, _ = identify_extreme_events(dataset, percentile=95)
    
    extreme_dir = output_dir / 'extreme_events'
    extreme_dir.mkdir(exist_ok=True)
    
    for i, idx in enumerate(extreme_indices[:10]):
        sample = dataset[idx]
        visualize_extreme_event(model, sample, idx,
                              extreme_dir / f'extreme_{i:02d}_idx{idx}.png')
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
```

---

## Expected Findings

### 1. Spatial Importance
- **Frozen DINO (Phase 1):** Distributed attention across fire region
- **Fine-tuned (Phase 2):** Concentrated on fire boundaries

### 2. Temporal Dependencies
- **Short sequences (T=2-3):** Recency bias (t-1 dominates ~60-70%)
- **Long sequences (T≥4):** More balanced (~40-50% on t-1)

### 3. Feature Importance (Expected Ranking)
1. **Fuels** (~0.02-0.04 F1 drop) - spread magnitude and rate
2. **Canopy bulk density** (~0.01-0.03) - fire intensity
3. **Elevation** (~0.01-0.02) - indirect terrain effects
4. **Flora** (~0.005-0.01) - vegetation-specific patterns
5. **Arqueo, Paleo, Urban** - Lower individual impact

### 4. Wind Alignment
- Mean alignment < π/4 (45°) indicates good learning
- Correlation between wind speed and spread area

### 5. Extreme Events
- **Correlates:** High wind + steep slope + fuel
- **Model bias:** Tends to underpredict (conservative)

---

## Next Steps

1. Run Phase 2 training (fine-tune DINO)
2. Compare Phase 1 vs Phase 2 feature quality
3. Implement attention rollout for DINO layers
4. Test on holdout sequences

---

## References

- **DinoV2:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
- **Attention Analysis:** Vaswani et al., "Attention Is All You Need", 2017
- **Gradient Attribution:** Sundararajan et al., "Axiomatic Attribution for Deep Networks", 2017
