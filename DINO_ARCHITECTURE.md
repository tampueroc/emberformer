# EmberFormer Architecture Migration: DINO + Transformer

**Date:** Oct 18, 2025  
**Status:** Proposed  
**Goal:** Replace SegFormer with DINO for superior spatial encoding while keeping transformer for temporal modeling

---

## Executive Summary

**Current Issue:** Using SegFormer incorrectly with fake multi-scale pyramid. Not leveraging proper spatial pretraining.

**Solution:** Use DINOv2 for spatial feature extraction + keep temporal transformer for fire spread dynamics.

**Expected Improvement:** 
- Better spatial understanding (boundaries, textures, terrain patterns)
- Proper use of pretrained vision models
- Simpler, more efficient architecture
- Matches approach that achieved "spectacular results" in similar work

---

## Architecture Comparison

### Current Architecture (Problematic)

```
Input: fire_hist [B, N, T] (raw fire tokens)
  ↓
Temporal Token Embedder → [B, N, T, d_model]
  - Fire: Linear(1 → d_model)
  - Static: Linear(Cs → d_model)  
  - Wind: Linear(2 → d_model)
  ↓
Temporal Transformer (per-patch attention) → [B, N, d_model]
  ↓
Reshape to grid → [B, d_model, Gy, Gx]
  ↓
❌ SegFormer Decoder (WRONG)
  - Creates fake pyramid via avg_pool
  - Loads full pretrained model but only uses decode head
  - Pretrained on ADE20K (150 semantic classes)
  - Not aligned with binary fire prediction
  ↓
Output: [B, 1, Gy, Gx]
  ↓
Refinement Decoder → [B, 1, H, W]
```

**Problems:**
1. ❌ Raw fire tokens lack spatial semantics
2. ❌ SegFormer fake pyramid doesn't provide real multi-scale features
3. ❌ ADE20K pretraining irrelevant to fire/terrain patterns
4. ❌ Loading unused encoder wastes memory
5. ❌ Overcomplicated for binary segmentation

---

### Proposed Architecture (DINO + Transformer)

```
Input: fire_hist [B, T, 1, H, W] (pixel frames)
       static [B, Cs, H, W] (terrain)
       wind [B, T, 2]
  ↓
┌─────────────────────────────────────────┐
│ SPATIAL ENCODING (per timestep)         │
│                                          │
│ DINO Encoder (frozen/fine-tuned)        │
│   - Input: [B*T, 1, H, W]               │
│   - Output: [B*T, N, d_dino]            │
│   - Uses: DINOv2 ViT-S/B                │
│   - Patch size: 16x16                   │
│                                          │
│ Static Encoder (DINO, frozen)           │
│   - Input: [B, Cs, H, W]                │
│   - Output: [B, N, d_dino]              │
│   - Single pass (terrain doesn't change)│
└─────────────────────────────────────────┘
  ↓
Reshape DINO features → [B, T, N, d_dino]
  ↓
┌─────────────────────────────────────────┐
│ FEATURE FUSION                           │
│                                          │
│ Fire features:   [B, T, N, d_dino]      │
│ Static features: [B, N, d_dino]         │
│   → Broadcast to [B, T, N, d_dino]      │
│ Wind: [B, T, 2]                          │
│   → MLP to [B, T, d_wind]               │
│   → Broadcast to [B, T, N, d_wind]      │
│                                          │
│ Fusion Strategy:                         │
│   Option A: Concatenate + Project       │
│     [fire; static; wind] → d_model      │
│   Option B: Cross-attention             │
│     fire attends to static context      │
└─────────────────────────────────────────┘
  ↓
Fused features → [B, N, T, d_model]
  ↓
┌─────────────────────────────────────────┐
│ TEMPORAL TRANSFORMER (unchanged)         │
│                                          │
│ Per-patch temporal attention             │
│   - Input:  [B, N, T, d_model]          │
│   - Layers: 3-6 transformer blocks      │
│   - Heads:  4-8                          │
│   - Output: [B, N, d_model]             │
│                                          │
│ Learns: Fire spread dynamics over time  │
└─────────────────────────────────────────┘
  ↓
Reshape to spatial grid → [B, d_model, Gy, Gx]
  ↓
┌─────────────────────────────────────────┐
│ SPATIAL DECODER (simplified)             │
│                                          │
│ Option A: Simple UNet Decoder           │
│   - Up1: d_model → 128 → 64             │
│   - Up2: 64 → 32                         │
│   - Head: 32 → 1                         │
│                                          │
│ Option B: Lightweight FPN               │
│   - Single-scale input                   │
│   - 2-3 upsampling stages               │
│   - Skip connections from transformer    │
│                                          │
│ Output: [B, 1, Gy, Gx]                  │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ REFINEMENT DECODER (unchanged)           │
│                                          │
│ Unpatchify + refine to pixel precision  │
│   - Input:  [B, 1, Gy, Gx]              │
│   - Output: [B, 1, H, W]                │
└─────────────────────────────────────────┘
```

---

## Detailed Component Specifications

### 1. DINO Spatial Encoder

**Purpose:** Extract rich spatial features from fire frames using self-supervised pretrained vision transformer

**Implementation:**
```python
from transformers import Dinov2Model

class DinoSpatialEncoder(nn.Module):
    """
    DINOv2 encoder for extracting spatial features from fire frames
    
    Args:
        model_name: 'facebook/dinov2-small' or 'facebook/dinov2-base'
        frozen: whether to freeze DINO weights
        input_channels: 1 (grayscale fire) or Cs (static terrain)
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        frozen: bool = True,
        input_channels: int = 1,
    ):
        super().__init__()
        
        # Load pretrained DINO
        self.dino = Dinov2Model.from_pretrained(model_name)
        self.d_dino = self.dino.config.hidden_size  # 384 (small) or 768 (base)
        self.patch_size = self.dino.config.patch_size  # 14 or 16
        
        # Adapt input channels if needed (DINO expects 3-channel RGB)
        if input_channels != 3:
            self.input_projection = nn.Conv2d(
                input_channels, 3, 
                kernel_size=1, 
                bias=False
            )
        else:
            self.input_projection = nn.Identity()
        
        # Freeze DINO if requested
        if frozen:
            for param in self.dino.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
        Returns:
            features: [B, N, d_dino] patch features
        """
        # Project to 3 channels
        x = self.input_projection(x)  # [B, 3, H, W]
        
        # DINO forward
        outputs = self.dino(x)
        
        # Get patch tokens (exclude CLS token)
        features = outputs.last_hidden_state[:, 1:, :]  # [B, N, d_dino]
        
        return features
```

**Model Options:**
- `facebook/dinov2-small`: 384-dim, 22M params, faster
- `facebook/dinov2-base`: 768-dim, 86M params, more expressive
- `facebook/dinov2-large`: 1024-dim, 300M params, best quality (GPU intensive)

**Recommended:** Start with `dinov2-small` frozen, fine-tune later if needed

---

### 2. Feature Fusion Module

**Purpose:** Combine DINO fire features, static terrain features, and wind into unified representation

**Strategy A: Concatenation + Projection (Simple)**

```python
class FeatureFusion(nn.Module):
    """
    Fuse DINO fire features, static features, and wind
    via concatenation and projection
    """
    def __init__(
        self,
        d_dino: int = 384,
        d_wind: int = 32,
        d_model: int = 256,
    ):
        super().__init__()
        
        # Wind embedding
        self.wind_embed = nn.Sequential(
            nn.Linear(2, d_wind),
            nn.LayerNorm(d_wind),
            nn.GELU(),
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_dino + d_dino + d_wind, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
    
    def forward(
        self,
        fire_features: torch.Tensor,    # [B, T, N, d_dino]
        static_features: torch.Tensor,  # [B, N, d_dino]
        wind: torch.Tensor,             # [B, T, 2]
    ) -> torch.Tensor:
        """
        Returns:
            fused: [B, N, T, d_model]
        """
        B, T, N, _ = fire_features.shape
        
        # Embed wind
        wind_emb = self.wind_embed(wind)  # [B, T, d_wind]
        
        # Broadcast static and wind to match fire shape
        static_expanded = static_features.unsqueeze(1).expand(B, T, N, -1)
        wind_expanded = wind_emb.unsqueeze(2).expand(B, T, N, -1)
        
        # Concatenate
        combined = torch.cat([
            fire_features,
            static_expanded,
            wind_expanded
        ], dim=-1)  # [B, T, N, d_dino + d_dino + d_wind]
        
        # Project to d_model
        fused = self.fusion_proj(combined)  # [B, T, N, d_model]
        
        # Rearrange for transformer: [B, N, T, d_model]
        fused = fused.transpose(1, 2)
        
        return fused
```

**Strategy B: Cross-Attention (Advanced)**

```python
class FeatureFusionCrossAttn(nn.Module):
    """
    Fuse features via cross-attention:
    Fire features attend to static context
    """
    def __init__(
        self,
        d_dino: int = 384,
        d_model: int = 256,
        nhead: int = 4,
    ):
        super().__init__()
        
        # Project DINO features to d_model
        self.fire_proj = nn.Linear(d_dino, d_model)
        self.static_proj = nn.Linear(d_dino, d_model)
        
        # Cross-attention: fire queries attend to static keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        # Wind embedding
        self.wind_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Final fusion
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        fire_features: torch.Tensor,
        static_features: torch.Tensor,
        wind: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, _ = fire_features.shape
        
        # Project
        fire_proj = self.fire_proj(fire_features)  # [B, T, N, d_model]
        static_proj = self.static_proj(static_features)  # [B, N, d_model]
        
        # Reshape for cross-attention
        fire_flat = fire_proj.reshape(B * T, N, -1)
        static_repeat = static_proj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, -1)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=fire_flat,
            key=static_repeat,
            value=static_repeat
        )  # [B*T, N, d_model]
        
        # Residual + reshape
        fire_contextualized = (fire_flat + attn_out).reshape(B, T, N, -1)
        
        # Add wind
        wind_emb = self.wind_embed(wind)  # [B, T, d_model]
        wind_expanded = wind_emb.unsqueeze(2).expand(-1, -1, N, -1)
        
        # Combine
        fused = self.norm(fire_contextualized + wind_expanded)
        
        # Rearrange: [B, N, T, d_model]
        fused = fused.transpose(1, 2)
        
        return fused
```

**Recommendation:** Start with Strategy A (simpler), upgrade to B if attention analysis shows benefit

---

### 3. Temporal Transformer (Unchanged)

Keep existing `TemporalTransformerEncoder` - it's already well-designed for per-patch temporal modeling.

**No changes needed** - just receives different input from DINO fusion instead of raw tokens.

---

### 4. Spatial Decoder (Simplified)

**Replace SegFormer with lightweight UNet decoder**

```python
class SimpleSpatialDecoder(nn.Module):
    """
    Lightweight decoder for patch-grid to binary segmentation
    
    Much simpler than SegFormer - appropriate for binary task
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()
        
        # Simple upsampling path
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, Gy, Gx]
        Returns:
            logits: [B, 1, Gy, Gx]
        """
        return self.decoder(x)
```

**Alternative: Keep existing UNet decoder** - `SpatialDecoderUNet` is already good, just remove SegFormer option.

---

### 5. Full Model Integration

```python
class EmberFormerDINO(nn.Module):
    """
    EmberFormer with DINO spatial encoding
    
    Architecture:
        Fire frames → DINO → [B, T, N, d_dino]
        Static → DINO → [B, N, d_dino]
        Fusion → [B, N, T, d_model]
        Temporal Transformer → [B, N, d_model]
        Spatial Decoder → [B, 1, Gy, Gx]
        Refinement → [B, 1, H, W]
    """
    def __init__(
        self,
        # DINO config
        dino_model: str = "facebook/dinov2-small",
        freeze_dino: bool = True,
        
        # Transformer config
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        
        # Decoder config
        spatial_hidden: int = 128,
        patch_size: int = 16,
        
        # Feature fusion
        fusion_type: str = "concat",  # "concat" or "cross_attn"
    ):
        super().__init__()
        
        # DINO encoders
        self.fire_encoder = DinoSpatialEncoder(
            model_name=dino_model,
            frozen=freeze_dino,
            input_channels=1,  # Grayscale fire
        )
        
        self.static_encoder = DinoSpatialEncoder(
            model_name=dino_model,
            frozen=True,  # Always freeze static encoder
            input_channels=7,  # Static terrain channels
        )
        
        d_dino = self.fire_encoder.d_dino
        
        # Feature fusion
        if fusion_type == "concat":
            self.fusion = FeatureFusion(
                d_dino=d_dino,
                d_wind=32,
                d_model=d_model,
            )
        elif fusion_type == "cross_attn":
            self.fusion = FeatureFusionCrossAttn(
                d_dino=d_dino,
                d_model=d_model,
                nhead=nhead,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Spatial decoder
        self.spatial_decoder = SimpleSpatialDecoder(
            d_model=d_model,
            hidden_channels=spatial_hidden,
        )
        
        # Refinement decoder
        self.refinement_decoder = RefinementDecoder(
            d_model=d_model,
            patch_size=patch_size,
            base_channels=32,
        )
        
        self.patch_size = patch_size
    
    def forward(
        self,
        fire_hist: torch.Tensor,   # [B, T, 1, H, W]
        static: torch.Tensor,      # [B, Cs, H, W]
        wind: torch.Tensor,        # [B, T, 2]
        valid_mask: torch.Tensor,  # [B, N, T]
    ) -> torch.Tensor:
        """
        Returns:
            predictions: [B, 1, H, W]
        """
        B, T, _, H, W = fire_hist.shape
        
        # Encode fire frames with DINO
        # Reshape to process all timesteps together
        fire_frames = fire_hist.reshape(B * T, 1, H, W)
        fire_features = self.fire_encoder(fire_frames)  # [B*T, N, d_dino]
        
        # Reshape back
        N = fire_features.shape[1]
        fire_features = fire_features.reshape(B, T, N, -1)
        
        # Encode static terrain (single pass)
        static_features = self.static_encoder(static)  # [B, N, d_dino]
        
        # Fuse features
        fused = self.fusion(
            fire_features=fire_features,
            static_features=static_features,
            wind=wind,
        )  # [B, N, T, d_model]
        
        # Temporal transformer
        temporal_features = self.temporal_transformer(
            fused, 
            valid_mask=valid_mask
        )  # [B, N, d_model]
        
        # Reshape to grid
        Gy = Gx = int(np.sqrt(N))
        assert Gy * Gx == N, f"N={N} must be square"
        
        grid_features = temporal_features.transpose(1, 2).reshape(B, -1, Gy, Gx)
        
        # Spatial decoder
        patch_logits = self.spatial_decoder(grid_features)  # [B, 1, Gy, Gx]
        
        # Refinement to pixels
        pixel_logits = self.refinement_decoder(
            patch_logits, 
            grid_features
        )  # [B, 1, H, W]
        
        return pixel_logits
    
    def unfreeze_dino(self):
        """Unfreeze DINO fire encoder for fine-tuning"""
        for param in self.fire_encoder.dino.parameters():
            param.requires_grad = True
```

---

## Training Strategy

### Phase 1: Frozen DINO (Fast convergence)

**Setup:**
- Freeze DINO fire encoder
- Freeze DINO static encoder
- Train: Fusion, Temporal Transformer, Spatial Decoder, Refinement

**Rationale:** Leverage DINO's pretrained spatial features, focus on learning temporal dynamics

**Duration:** 20-30 epochs until convergence

**Config:**
```yaml
dino:
  model: "facebook/dinov2-small"
  freeze_fire: true
  freeze_static: true

optimizer:
  lr: 1e-3
  weight_decay: 1e-4

training:
  batch_size: 8
  epochs: 30
```

---

### Phase 2: Fine-tune DINO (Optional)

**Setup:**
- Unfreeze DINO fire encoder
- Keep DINO static encoder frozen
- Lower learning rate

**Rationale:** Adapt DINO to fire-specific visual patterns

**Duration:** 10-20 epochs

**Config:**
```yaml
dino:
  freeze_fire: false
  freeze_static: true

optimizer:
  lr: 1e-5  # Much lower for fine-tuning
  weight_decay: 1e-4

training:
  batch_size: 4  # Reduce if needed
  epochs: 20
```

---

### Phase 3: End-to-end (If needed)

**Setup:**
- Everything trainable
- Very low learning rate
- Careful monitoring for overfitting

**Config:**
```yaml
dino:
  freeze_fire: false
  freeze_static: false

optimizer:
  lr: 5e-6
  weight_decay: 1e-3  # Higher regularization
```

---

## Data Pipeline Changes

### Current (Token-based):
```python
# Loads pre-computed patch tokens
fire_tokens: [T, N]  # Binary values per patch
static_tokens: [N, Cs]  # Aggregated terrain per patch
```

### New (Pixel-based for DINO):
```python
# Loads full pixel frames
fire_frames: [T, 1, H, W]  # Full resolution binary fire
static: [Cs, H, W]  # Full resolution terrain
```

**Good news:** You already have `RawFireDataset` - just need to modify collation

**Changes needed:**
1. Return fire as `[T, 1, H, W]` instead of tokens
2. Keep static as `[Cs, H, W]`
3. DINO handles patchification internally

---

## Implementation Checklist

### Step 1: Add DINO Encoder
- [ ] Create `DinoSpatialEncoder` class
- [ ] Test with single fire frame
- [ ] Test with static terrain
- [ ] Verify output shapes

### Step 2: Update Feature Fusion
- [ ] Implement `FeatureFusion` (concat version)
- [ ] Update wind embedding
- [ ] Test fusion output shape

### Step 3: Integrate with EmberFormer
- [ ] Create `EmberFormerDINO` class
- [ ] Wire DINO → Fusion → Transformer → Decoder
- [ ] Test forward pass end-to-end

### Step 4: Update Data Pipeline
- [ ] Modify `RawFireDataset` to return pixel frames
- [ ] Update collate function
- [ ] Test data loading

### Step 5: Training Script
- [ ] Add DINO config to `emberformer.yaml`
- [ ] Update `train_emberformer.py` to handle pixel inputs
- [ ] Add DINO freeze/unfreeze logic

### Step 6: Experiments
- [ ] Baseline: Frozen DINO-small + Transformer
- [ ] Ablation: DINO-base vs DINO-small
- [ ] Ablation: Fusion concat vs cross-attention
- [ ] Fine-tuning: Unfreeze DINO after convergence

---

## Expected Improvements

### Spatial Understanding
- **Better boundaries:** DINO learns edge detection from natural images
- **Texture awareness:** Fire front patterns, terrain textures
- **Spatial continuity:** Smoother predictions, fewer isolated pixels

### Training Efficiency
- **Faster convergence:** Strong pretrained features reduce training time
- **Better generalization:** DINO features are robust to variations
- **Less overfitting:** Pretrained backbone acts as regularizer

### Performance Metrics
Conservative estimates vs current baseline:

| Metric | Current (SegFormer) | Expected (DINO) |
|--------|---------------------|-----------------|
| F1 Score | 0.XX | +5-10% |
| IoU | 0.XX | +5-15% |
| Boundary Accuracy | Poor | Significantly better |
| Training Time | Baseline | 20-30% faster convergence |

---

## Model Sizes & Memory

### DINO Model Options:

| Model | Hidden Size | Params | Memory (inference) | Speed |
|-------|-------------|--------|-------------------|-------|
| dinov2-small | 384 | 22M | ~500MB | Fast |
| dinov2-base | 768 | 86M | ~1.5GB | Medium |
| dinov2-large | 1024 | 300M | ~4GB | Slow |

### Full Model Comparison:

| Component | Current (SegFormer) | Proposed (DINO-small) |
|-----------|---------------------|----------------------|
| Spatial Encoder | 0 (raw tokens) | 22M (frozen) |
| Spatial Decoder | 3.7M (SegFormer-B0) | 1M (SimpleSpatialDecoder) |
| Temporal Transformer | ~5M | ~5M (same) |
| Refinement | ~1M | ~1M (same) |
| **Total Trainable** | ~10M | ~7M |
| **Total Params** | ~10M | ~29M (with frozen DINO) |

**Memory:** ~2-3GB GPU for batch_size=8 with DINO-small (frozen)

---

## Risks & Mitigations

### Risk 1: DINO expects RGB, fires are grayscale
**Mitigation:** 
- Use 1×1 conv projection: 1 channel → 3 channels
- Alternative: Replicate grayscale to 3 channels
- Fine-tuning phase adapts to grayscale

### Risk 2: DINO trained on natural images, not fire
**Mitigation:**
- Spatial patterns (boundaries, textures) transfer well
- Phase 2 fine-tuning adapts to fire-specific patterns
- Static terrain (vegetation, elevation) is natural imagery

### Risk 3: Increased memory usage
**Mitigation:**
- Start with dinov2-small (22M params)
- Keep DINO frozen (no optimizer states)
- Reduce batch size if needed
- Use gradient checkpointing for DINO

### Risk 4: Slower inference
**Mitigation:**
- DINO forward pass ~10-20ms per frame on GPU
- Can cache static features (computed once)
- For deployment: distill to smaller model

### Risk 5: Dataset compatibility
**Mitigation:**
- Already have `RawFireDataset` for pixels
- Minimal changes to data pipeline
- Backwards compatible with token approach

---

## Ablation Studies to Run

### 1. DINO Model Size
- [ ] dinov2-small (384-dim)
- [ ] dinov2-base (768-dim)
- [ ] dinov2-large (1024-dim) - if GPU allows

**Hypothesis:** Base offers best quality/speed tradeoff

### 2. DINO Freezing Strategy
- [ ] Fully frozen (all phases)
- [ ] Frozen → Fine-tune after convergence
- [ ] Trainable from start

**Hypothesis:** Frozen first, then fine-tune is optimal

### 3. Feature Fusion Strategy
- [ ] Concatenation + projection
- [ ] Cross-attention
- [ ] Add + LayerNorm

**Hypothesis:** Concat is sufficient, cross-attn helps if bottleneck

### 4. Static Encoding
- [ ] DINO (same as fire)
- [ ] Separate conv encoder
- [ ] Linear projection (current)

**Hypothesis:** DINO for static also improves terrain understanding

### 5. Spatial Decoder Complexity
- [ ] Minimal (2-3 conv layers)
- [ ] UNet (current)
- [ ] FPN with skip connections

**Hypothesis:** Minimal is sufficient given strong DINO features

---

## Migration Path

### Week 1: Implementation
- Day 1-2: DINO encoder + feature fusion
- Day 3-4: Integrate with EmberFormer
- Day 5: Data pipeline updates
- Day 6-7: Testing and debugging

### Week 2: Baseline Experiments
- Frozen DINO-small + full training
- Compare to current SegFormer baseline
- Validate improvements

### Week 3: Optimization
- Try DINO-base
- Experiment with fusion strategies
- Fine-tune DINO encoder

### Week 4: Ablations & Analysis
- Run full ablation suite
- Visualize attention maps
- Analyze failure cases

---

## Success Criteria

### Must Have (Phase 1):
- ✅ Model trains without errors
- ✅ Converges faster than current baseline
- ✅ F1/IoU >= current best model
- ✅ No memory issues on available GPUs

### Should Have (Phase 2):
- ✅ F1/IoU > current best by 5%+
- ✅ Better visual quality (boundaries, smoothness)
- ✅ Fine-tuning DINO improves further

### Nice to Have (Phase 3):
- ✅ DINO-base outperforms DINO-small significantly
- ✅ Cross-attention fusion adds measurable benefit
- ✅ Generalizes better to held-out fires/regions

---

## References

### DINO Papers
- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- Original DINO: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

### HuggingFace Models
- `facebook/dinov2-small`: https://huggingface.co/facebook/dinov2-small
- `facebook/dinov2-base`: https://huggingface.co/facebook/dinov2-base

### Alternative Approaches (If DINO doesn't work)
- **SAM**: Segment Anything encoder
- **CLIP**: Vision-language pretrained features
- **MAE**: Masked Autoencoder features

---

## Questions for Consideration

1. **What resolution does DINO expect?**
   - Native: 224×224 or 518×518
   - Can handle arbitrary sizes via interpolation
   - Your fires: likely 256×256 to 512×512 → good fit

2. **Should we use different DINO for fire vs static?**
   - Option A: Same frozen DINO (simpler)
   - Option B: Separate models, static frozen (more capacity)
   - **Recommendation:** Same model, easier to manage

3. **How to handle variable input sizes?**
   - DINO can adapt via patch embedding interpolation
   - Or: resize all inputs to fixed size (e.g., 224×224)
   - **Recommendation:** Resize to 224×224 or 448×448

4. **What about temporal consistency?**
   - DINO processes frames independently
   - Transformer handles temporal consistency
   - Could explore: DINO + temporal position encoding

---

## Next Steps

1. **Review this document** with team/advisor
2. **Decide on DINO model size** (recommend start with small)
3. **Implement DinoSpatialEncoder** as standalone module
4. **Test on single batch** before full integration
5. **Plan experiments** and compute budget

---

**Status:** Ready for implementation  
**Owner:** [Your name]  
**Reviewers:** [Advisors/team]  
**Target Date:** [Set milestone]
