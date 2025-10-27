# EmberFormer-DINO Architecture Review for Grad-CAM Analysis

## Target Convolutional Layers for Grad-CAM

### 1. **Spatial Decoder** (`SimpleSpatialDecoder`)
**Location:** `models/emberformer.py` lines 774-815

**Architecture:**
```python
self.decoder = nn.Sequential(
    # Stage 1
    nn.Conv2d(d_model, hidden_channels, 3, padding=1),
    nn.BatchNorm2d(hidden_channels),
    nn.ReLU(inplace=True),
    
    # Stage 2
    nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
    nn.BatchNorm2d(hidden_channels // 2),
    nn.ReLU(inplace=True),
    
    # Output head
    nn.Conv2d(hidden_channels // 2, 1, 1),
)
```

**Grad-CAM Target Layers:**
- **Primary:** `decoder[5]` - Conv2d(hidden_channels // 2, 1, 1) - **Last conv before output**
- **Secondary:** `decoder[4]` - Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1)
- **Tertiary:** `decoder[0]` - Conv2d(d_model, hidden_channels, 3, padding=1)

**Input:** `[B, d_model, Gy, Gx]` - Features from temporal transformer  
**Output:** `[B, 1, Gy, Gx]` - Patch-level logits  
**Resolution:** 29×29 patches (406÷14 for DINOv2-small)

---

### 2. **Refinement Decoder** (`RefinementDecoder`)
**Location:** `models/emberformer.py` lines 261-352

**Architecture (for patch_size=14):**
```python
self.upsample_layers = nn.Sequential(
    # Stage 1: 2× upsampling
    nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(base_channels * 4),
    nn.ReLU(inplace=True),
    nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
    nn.BatchNorm2d(base_channels * 4),
    nn.ReLU(inplace=True),
    
    # Stage 2: 7× upsampling via bilinear + conv
    nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False),
    nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
    nn.BatchNorm2d(base_channels * 2),
    nn.ReLU(inplace=True),
    nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
    nn.BatchNorm2d(base_channels * 2),
    nn.ReLU(inplace=True),
)

self.output_conv = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
```

**Grad-CAM Target Layers:**
- **Primary:** `output_conv` - Conv2d(base_channels * 2, 1, 1) - **Last conv before pixel logits**
- **Secondary:** `upsample_layers[10]` - Conv2d(base_channels * 2, base_channels * 2, 3, padding=1) - Last refinement conv
- **Tertiary:** `upsample_layers[7]` - Conv2d(base_channels * 4, base_channels * 2, 3, padding=1) - After upsampling

**Input:** 
- `features`: `[B, d_model, Gy, Gx]` - From temporal transformer
- `coarse_pred`: `[B, 1, Gy, Gx]` - From spatial decoder

**Output:** `[B, 1, 406, 406]` - Pixel-level logits  
**Resolution:** Full 406×406 pixels

---

## Architecture Flow for Grad-CAM

```
Fire History [B, T, 1, 406, 406]
    ↓
DINO Encoder (ViT) - Use Attention Rollout here
    ↓
Fire Features [B, T, 841, 384]  (841 = 29×29 patches)
    ↓
Feature Fusion (with static DINO + wind)
    ↓
Temporal Transformer - Use Attention Maps here
    ↓
Grid Features [B, 256, 29, 29]
    ↓
┌──────────────────────────────────────┐
│ Spatial Decoder (SimpleSpatialDecoder) │
│ ✓ Apply Grad-CAM here                  │
│ Target: decoder[5] (last conv)         │
└──────────────────────────────────────┘
    ↓
Coarse Logits [B, 1, 29, 29]
    ↓
┌──────────────────────────────────────┐
│ Refinement Decoder (RefinementDecoder) │
│ ✓ Apply Grad-CAM here                  │
│ Target: output_conv (last conv)        │
└──────────────────────────────────────┘
    ↓
Pixel Logits [B, 1, 406, 406]
```

---

## Grad-CAM Implementation Strategy

### **Single Target: Refinement Decoder Output**
**Target Layer:** `model.refinement_decoder.output_conv`  
**Output Resolution:** 406×406 pixels  
**Purpose:** Shows which pixel-level features drive final fire spread predictions

**Rationale:**
- Final convolutional layer before pixel logits
- Full spatial resolution (406×406)
- Most interpretable for understanding what drives predictions
- Direct connection to model output

---

## Pais et al. (2020) Methodology

### 1. **CAM (Class Activation Mapping)**
- Requires GAP (Global Average Pooling) layer
- **Not applicable** to our architecture (no GAP before final conv)
- **Skip CAM**, use Grad-CAM instead

### 2. **Grad-CAM**
**Steps:**
1. Forward pass: compute predictions `y_c` (fire class logits)
2. Backward: compute gradients `∂y_c / ∂A_k` where `A_k` is activation map at layer k
3. Average gradients spatially: `α_k = (1/Z) Σ_i Σ_j (∂y_c / ∂A_k^{ij})`
4. Weight activations: `L_Grad-CAM = ReLU(Σ_k α_k · A_k)`
5. Upsample to input resolution

**Target Layers:**
- `spatial_decoder.decoder[5]` → 29×29 heatmap
- `refinement_decoder.output_conv` → 406×406 heatmap

### 3. **Guided Grad-CAM**
**Steps:**
1. Compute Grad-CAM heatmap (coarse)
2. Compute Guided Backpropagation (fine-grained gradients to input)
3. Element-wise multiply: `Guided Grad-CAM = Grad-CAM ⊙ Guided Backprop`

**Purpose:** Combines spatial localization (Grad-CAM) with high-resolution details (Guided Backprop)

---

## Hook Registration Points

### For Spatial Decoder:
```python
model.spatial_decoder.decoder[5].register_forward_hook(hook_fn)
model.spatial_decoder.decoder[5].register_backward_hook(grad_hook_fn)
```

### For Refinement Decoder:
```python
model.refinement_decoder.output_conv.register_forward_hook(hook_fn)
model.refinement_decoder.output_conv.register_backward_hook(grad_hook_fn)
```

---

## Expected Outputs

### Spatial Decoder Grad-CAM
- **Resolution:** 29×29
- **Shows:** Which patch regions in the feature grid are most important
- **Interpretation:** Coarse spatial importance (e.g., "upper-left quadrant drives prediction")

### Refinement Decoder Grad-CAM
- **Resolution:** 406×406
- **Shows:** Which exact pixels in the refined features are most important
- **Interpretation:** Fine-grained spatial importance (e.g., "fire boundary pixels at coordinates (x, y)")

### Guided Grad-CAM
- **Resolution:** 406×406
- **Shows:** High-resolution pixel-level importance map
- **Interpretation:** Exact input pixels that drive predictions (e.g., "these specific fire front pixels")

---

## Comparison with Attention Methods

| Method | Target | Resolution | What It Shows |
|--------|--------|------------|---------------|
| **Attention Rollout** | DINO Encoder | 29×29 patches | Which input patches DINO attends to |
| **Temporal Attention** | Transformer | T×T matrix | Which past timesteps are important |
| **Grad-CAM (Spatial)** | Spatial Decoder | 29×29 | Which features drive coarse predictions |
| **Grad-CAM (Refinement)** | Refinement Decoder | 406×406 | Which features drive pixel predictions |
| **Guided Grad-CAM** | Full Pipeline | 406×406 | Which input pixels drive predictions |

---

## Summary

✅ **Two primary Grad-CAM targets identified:**
1. `spatial_decoder.decoder[5]` - Last conv in spatial decoder (29×29)
2. `refinement_decoder.output_conv` - Last conv in refinement decoder (406×406)

✅ **Pais et al. methodology applies perfectly** to these convolutional layers

✅ **Next steps:**
1. Implement Grad-CAM hooks for both layers
2. Implement Guided Backpropagation
3. Combine into Guided Grad-CAM
4. Generate saliency maps on validation samples
