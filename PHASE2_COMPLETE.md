# Phase 2 Complete: EmberFormer Model Implementation

**Branch:** `feat-phase-2`  
**Commit:** `aa0f43d`  
**Date:** Oct 17, 2025  
**Status:** ✅ COMPLETE - All tests passing (5/5)

---

## Summary

Phase 2 successfully implements the EmberFormer architecture: a hybrid temporal transformer + spatial decoder for wildfire spread prediction. The model uses custom temporal attention over patch history combined with a pre-trained SegFormer decoder for segmentation.

**Key Achievement:** 585K trainable parameters, handles variable-length sequences (T=2-8), ready for training with differential learning rates.

---

## Architecture Components

### 1. Temporal Token Embedder ✅

**Class:** `TemporalTokenEmbedder`

**Purpose:** Combines fire, wind, static, and positional information into embeddings

**Features:**
- Fire embedding: Linear(1 → d_model) for per-patch occupancy
- Wind embedding: MLP(2 → d_model) with GELU activation
- Static embedding: Linear(Cs → d_model) broadcast over time
- Positional encoding: Sinusoidal encoding for temporal position
- Configurable: can disable wind or static embeddings

**Output:** `[B, N, T, d_model]` token embeddings

### 2. Temporal Transformer Encoder ✅

**Class:** `TemporalTransformerEncoder`

**Purpose:** Per-patch temporal self-attention

**Architecture:**
- PyTorch TransformerEncoder with configurable layers
- Per-patch processing: treats each spatial patch independently
- Padding mask support: handles variable-length sequences
- Extracts last valid timestep representation per patch

**Key Details:**
- Flattens spatial dimension: `[B, N, T, d] → [B×N, T, d]`
- Applies transformer with `src_key_padding_mask`
- Gathers last valid features: `[B×N, d] → [B, N, d]`

**Output:** `[B, N, d]` patch representations

### 3. Spatial Decoders ✅

#### Option A: SegFormer Decoder (Recommended)

**Class:** `SpatialDecoderSegFormer`

**Purpose:** Pre-trained segmentation decoder

**Features:**
- Loads pre-trained SegFormer from HuggingFace
- Adapts d_model features to SegFormer's expected input channels
- Creates multi-scale feature pyramid via projection + downsampling
- Freeze/unfreeze support for transfer learning

**Default Model:** `nvidia/segformer-b0-finetuned-ade-512-512`

**Output:** `[B, 1, Gy, Gx]` logits

#### Option B: UNet Decoder (Fallback)

**Class:** `SpatialDecoderUNet`

**Purpose:** Lightweight custom decoder

**Features:**
- Small U-Net with skip connections
- No external dependencies
- Fast and memory efficient

**Output:** `[B, 1, Gy, Gx]` logits

### 4. EmberFormer (Main Model) ✅

**Class:** `EmberFormer`

**Complete Pipeline:**
```
Input: fire_hist [B,N,T] + wind_hist [B,T,2] + static [B,N,Cs] + valid_t [B,T]
  ↓
TemporalTokenEmbedder → [B, N, T, d]
  ↓
TemporalTransformerEncoder → [B, N, d]
  ↓
Reshape to grid → [B, d, Gy, Gx]
  ↓
SpatialDecoder (SegFormer/UNet) → [B, 1, Gy, Gx]
  ↓
(Optional) Unpatchify → [B, 1, H, W]
```

**Methods:**
- `forward()`: Returns patch-level logits `[B, 1, Gy, Gx]`
- `forward_pixels()`: Unpatchifies to pixel space `[B, 1, H, W]`
- `unfreeze_decoder()`: Unfreeze spatial decoder for fine-tuning

**Configuration:**
```python
EmberFormer(
    d_model=64,              # Embedding dimension
    static_channels=10,      # Terrain feature channels
    nhead=4,                 # Attention heads
    num_layers=3,            # Transformer layers
    spatial_decoder='segformer',  # or 'unet'
    patch_size=16,
    max_seq_len=8            # T≤6 for >50% of dataset
)
```

---

## Training Script

**File:** `scripts/train_emberformer.py`

**Based on:** `train_stage_c_tokens.py` with temporal extensions

**Key Features:**

1. **Temporal Data Loading**
   - Uses `TokenFireDataset(use_history=True)`
   - Uses `collate_tokens_temporal` for variable-length batching
   - Handles T=2 to T=8+ sequences in same batch

2. **Differential Learning Rates**
   - Temporal components: `lr_temporal = 1e-3`
   - Spatial decoder: `lr_spatial = 1e-4` (lower for pre-trained)
   - Separate parameter groups in optimizer

3. **Transfer Learning Strategy**
   - Option to freeze decoder initially
   - Unfreeze after N epochs (configurable)
   - Example: Train temporal 5 epochs → unfreeze → fine-tune together

4. **Loss & Metrics**
   - Unpatchifies to pixel space for loss computation
   - Masked BCE with auto pos_weight balancing
   - Pixel-level metrics: Accuracy, Precision, Recall, F1, IoU

5. **W&B Integration**
   - Auto logging of train/val metrics
   - Validation preview images
   - Model checkpointing

**Usage:**
```bash
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --gpu 0
```

---

## Configuration

**File:** `configs/emberformer.yaml`

**Key Settings:**

```yaml
model:
  temporal:
    d_model: 64
    nhead: 4
    num_layers: 3
    dropout: 0.1
    max_seq_len: 8  # Dataset constraint: >50% have T≤6
    
  spatial:
    decoder_type: "segformer"
    pretrained: true
    freeze_epochs: 5
    
train:
  epochs: 20
  lr_temporal: 1e-3
  lr_spatial: 1e-4
  weight_decay: 1e-4
  
data:
  batch_size: 8  # Lower than baselines due to temporal sequences
```

---

## Tests

**File:** `tests/test_emberformer.py`

**Results:** ✅ 5/5 passing

### Test 1: Model Initialization
- Creates EmberFormer with various configs
- Verifies parameter counts
- **Result:** 585K parameters with UNet decoder

### Test 2: Forward Pass
- Synthetic batch: [B=2, N=25, T=4, Cs=10]
- Checks output shape: `[B, 1, Gy, Gx]`
- **Result:** ✅ Correct shape

### Test 3: Unpatchification
- Tests `forward_pixels()` method
- Verifies upsampling: `[B, 1, Gy, Gx] → [B, 1, H, W]`
- **Result:** ✅ Correct pixel dimensions

### Test 4: Variable-Length Sequences
- Tests sequences: T=[3, 5, 2] in same batch
- Verifies padding mask handling
- Checks outputs differ per sequence length
- **Result:** ✅ Handles variable lengths correctly

### Test 5: Backward Pass
- Forward + loss + backward
- Verifies gradients computed for all parameters
- **Result:** ✅ 56/56 parameters have gradients

---

## Model Statistics

### Parameters (with UNet decoder)
- **Total:** 584,993
- **Trainable:** 584,993
- **Breakdown:**
  - Token embedder: ~40K
  - Temporal transformer: ~200K
  - Spatial decoder: ~340K

### Memory & Performance (Estimated)
- **Batch=8, T=6, N=400 (20×20 grid):** ~2GB GPU memory
- **Forward pass:** ~50ms on GPU
- **Training throughput:** ~100 samples/sec (single GPU)

### With SegFormer Decoder (B0)
- **Total:** ~3.7M parameters (SegFormer pre-trained)
- **Additional memory:** ~500MB

---

## Integration with Phase 1

Phase 2 seamlessly integrates with Phase 1 data layer:

```python
from data import TokenFireDataset, collate_tokens_temporal
from models import EmberFormer

# Phase 1: Dataset with temporal history
ds = TokenFireDataset(cache_dir, raw_root, use_history=True)

# Phase 1: Temporal collate
loader = DataLoader(ds, batch_size=8, collate_fn=collate_tokens_temporal)

# Phase 2: Model
model = EmberFormer(d_model=64, spatial_decoder='segformer')

# Training loop
for X_batch, y_batch in loader:
    logits = model(
        X_batch["fire_hist"], 
        X_batch["wind_hist"], 
        X_batch["static"], 
        X_batch["valid_t"], 
        grid_shape=(Gy, Gx)
    )
    # ... compute loss and train
```

---

## Next Steps

### Immediate: Training & Validation

1. **Build patch cache** (if not done):
   ```bash
   uv run python scripts/build_patch_cache.py --config configs/emberformer.yaml
   ```

2. **Run baseline training** (20 epochs):
   ```bash
   uv run python scripts/train_emberformer.py --config configs/emberformer.yaml
   ```

3. **Monitor W&B dashboard:**
   - Track train/val F1, IoU
   - Check validation previews
   - Watch for overfitting

### Short-term: Experiments

1. **Ablation studies:**
   - With/without wind embeddings
   - With/without static embeddings
   - 1 vs 3 vs 6 transformer layers
   - History length: T=3 vs 6 vs 8

2. **Decoder comparison:**
   - UNet vs SegFormer
   - SegFormer-B0 vs B1 vs B2

3. **Baseline comparison:**
   - EmberFormer vs UNetS (last-frame only)
   - Quantify improvement from temporal history

### Medium-term: Optimization

1. **Hyperparameter sweeps:**
   - See PROJECT_STATE.md for W&B sweep config
   - Architecture search: d_model, nhead, num_layers
   - Learning rate tuning

2. **Advanced features:**
   - Add Dice loss for boundary quality
   - Implement attention visualization
   - Test on longer sequences (T>8) if needed

---

## Files Changed

```
Added:
  models/emberformer.py          (434 lines) - EmberFormer architecture
  scripts/train_emberformer.py   (441 lines) - Training script
  tests/test_emberformer.py      (353 lines) - Model tests
  configs/emberformer.yaml       (101 lines) - Configuration
  PHASE2_COMPLETE.md             (this file) - Phase 2 summary

Modified:
  models/__init__.py             (+2 lines)  - Export EmberFormer
```

---

## Dependencies

**New Requirement (Optional):**
```toml
[project.optional-dependencies]
segformer = [
    "transformers>=4.30.0",  # For SegFormer decoder
]
```

**Install with:**
```bash
uv pip install transformers
# or
uv pip install -e ".[segformer]"
```

**Note:** UNet decoder works without transformers package

---

## Troubleshooting

### Issue: ImportError for transformers
**Solution:** Install transformers or use `spatial_decoder='unet'`

### Issue: CUDA out of memory
**Solutions:**
- Reduce `batch_size` (8 → 4)
- Reduce `d_model` (64 → 32)
- Reduce `max_seq_len` (8 → 6)
- Use gradient checkpointing (not yet implemented)

### Issue: Loss not decreasing
**Check:**
- Learning rates: temporal should be higher than spatial
- Pos_weight: use "auto" for imbalanced data
- Validation previews: are predictions reasonable?
- Temporal masks: verify padding is correct

### Issue: SegFormer slow to load
**Solution:** First load takes time (downloads ~90MB), subsequent loads are fast

---

## Performance Tips

1. **Start small:** Use UNet decoder for quick iterations
2. **Freeze decoder:** Set `freeze_epochs=5` for faster temporal training
3. **Batch size:** 8 is a good balance for T=6 sequences
4. **Num workers:** 4-8 for data loading (avoid OOM)
5. **Mixed precision:** Enabled by default via `amp.autocast`

---

## Success Criteria

### ✅ Phase 2 MVP (Complete)
- [x] EmberFormer model implemented
- [x] Training script working
- [x] All tests passing
- [x] Config ready

### 🎯 Next: Phase 3 Experiments
- [ ] Train baseline (20 epochs)
- [ ] Beat UNetS last-frame baseline
- [ ] F1 > 0.65 on validation
- [ ] Temporal attention shows meaningful patterns

---

## References

- **Phase 1:** Dataset extensions (feat-phase-1 branch)
- **PROJECT_STATE.md:** Full architecture design and roadmap
- **train_stage_c_tokens.py:** Baseline training script
- **HuggingFace SegFormer:** https://huggingface.co/docs/transformers/model_doc/segformer

---

**Status:** ✅ Phase 2 complete. Model validated and ready for training. Proceed to Phase 3 (Experimentation).
