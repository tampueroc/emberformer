# EmberFormer Project State

**Date:** Oct 17, 2025  
**Goal:** Transformer architecture that captures long dependencies for wildfire spread prediction

---

## Executive Summary

EmberFormer is a next-step fire prediction system. It currently has:
- ✅ Robust data pipelines (raw pixels + patch tokens)
- ✅ Working baselines (CopyLast, 2D/3D Conv, UNet)
- ✅ Training infrastructure with W&B logging
- ✅ **W&B sweep-ready** (see Hyperparameter Tuning section)
- ❌ **No transformer implementation yet**
- ❌ Temporal history not used in token models (only last frame)

**Key Insight:** This is a **bounded segmentation task** (binary fire/no-fire prediction). We should use pre-trained spatial encoders + custom temporal transformer.

**Critical Dataset Constraint:** >50% of sequences have **T_hist=6 or fewer** timesteps. Design for T=6-8, not T=16+.

---

## Current Implementation Status

### ✅ Data Layer (Complete)

**RawFireDataset:**
- Loads static terrain (GeoTIFF + normalization, skips binary bands)
- Fire frames (binary from PNG band)
- Targets (isochrone masks)
- Wind (global min/max normalization)
- Supports variable-length sequences with left-padding + validity masks

**TokenFireDataset:**
- Loads cached patch tokens: static [N,Cs], fire_tokens [T,N], wind [T,2]
- Currently uses only **last fire step** (t_last) - not using history!
- Target: per-patch means from isochrone

**Patch Cache Builder:**
- `build_patch_cache.py`: creates token cache per sequence with metadata
- Supports strict/non-strict padding
- Issue: `valid_tokens.npy` currently all-ones (doesn't reflect spatial padding footprint)

### ✅ Models (Baselines Only)

1. **CopyLast:** Simple baseline
2. **ConvHead2D:** 2D conv over last fire + static + wind
3. **Tiny3D:** 3D conv over time dimension
4. **UNetS:** Small U-Net for grid/pixel predictions

### ✅ Training Scripts

- `train_phase1.py`: Baselines on raw H×W sequences
- `train_stage_c_tokens.py`: UNetS over patch grids, unpatchifies to pixel space for loss
- `train_stage_c_raw.py`: UNetS on full-resolution pixels with weighted BCE
- Performance benchmarks and data sanity checks available

### ❌ Missing Components

- **No transformer architecture**
- **No temporal attention mechanism**
- **No long-range dependency modeling**
- Token pipeline doesn't expose temporal history to models

---

## Recommended Architecture: Hybrid Transformer

### Key Design Decision

Since this is a **segmentation task** (predicting fire boundaries), we should:
- ✅ Use **pre-trained spatial encoder** (ImageNet) for spatial feature extraction
- ✅ Build **custom temporal transformer** (from scratch) for fire spread dynamics
- ✅ Leverage transfer learning where it matters (spatial patterns, boundaries)
- ✅ Design for **T=6-8** (since >50% of dataset has ≤6 history frames)

### Architecture Overview

```
Input: Patch tokens [B, N, T] + static [B, N, Cs] + wind [B, T, 2]
  ↓
[Custom] Temporal Transformer per patch
  - Learns fire spread dynamics over time
  - Per-patch temporal attention
  - Output: [B, N, d] representations
  ↓
Reshape to spatial grid: [B, d, Gy, Gx]
  ↓
[Pre-trained] Spatial Decoder (ImageNet weights)
  - Options: SegFormer, SAM decoder, UNet+ResNet
  - Understands boundaries, spatial structure
  - Output: [B, 1, H, W] segmentation mask
  ↓
Loss: Weighted BCE + (optional) Dice/IoU in pixel space
```

### Pre-trained Model Options

**Option 1: SegFormer (Recommended)**
- Transformer-based segmentation
- Efficient multi-scale features
- HuggingFace: `nvidia/segformer-b0-finetuned-ade-512-512`
- Lightweight, fast inference

**Option 2: Segment Anything (SAM) Decoder**
- State-of-art boundary detection
- Strong zero-shot generalization
- Meta AI: `facebook/sam-vit-base`
- Heavier but more accurate

**Option 3: UNet + ResNet Backbone**
- Battle-tested for segmentation
- torchvision/timm: `resnet34` or `resnet50`
- Most stable option

---

## Implementation Roadmap

### Phase 1: Dataset Extensions (0.5-1 day)

**Task 1.1:** Extend TokenFireDataset for temporal history
```python
# Current: returns fire_tokens[t_last] (single frame)
# New: return fire_tokens[:t_last+1] (full history)
fire_hist = fire_tokens[:t_last+1]  # [T_hist, N]
wind_hist = wind[:t_last+1]         # [T_hist, 2]
static = static_tokens              # [N, Cs]
target = y                          # [patch-level or pixel-level]
```

**Task 1.2:** Add temporal collate function
- Left-pad time dimension to T_max across batch
- Create validity mask: `valid_t [B, T_max]` (True=real, False=pad)
- Pass through grid dimensions (Gy, Gx) via metadata

**Task 1.3:** Fix spatial validity mask (optional but recommended)
- Update `build_patch_cache.py` to compute `valid_tokens` from footprint
- Use `utils.patchify.valid_token_mask_from_footprint`
- Mask loss for padded spatial patches

### Phase 2: Model Implementation (1-2 days)

**Task 2.1:** Create `models/emberformer.py`

**Components:**
1. **Temporal Token Embedder**
   ```python
   # Per-patch, per-timestep embedding
   e_fire = Linear(1 → d_model)          # fire occupancy
   e_wind = MLP(2 → d_model)             # wind vector
   e_pos = PositionalEncoding(T)         # temporal position
   e_static = Linear(Cs → d_model)       # static terrain (broadcast)
   
   token[t,n] = e_fire[t,n] + e_wind[t] + e_pos[t] + e_static[n]
   ```

2. **Temporal Transformer Encoder**
   ```python
   # Per-patch temporal attention
   tokens_flat = rearrange(tokens, 'b n t d -> (b n) t d')
   # Apply transformer with padding mask from valid_t
   encoder_output = TransformerEncoder(tokens_flat, mask=valid_t)
   # Take last valid timestep per patch
   patch_features = encoder_output[:, -1, :]  # [(B×N), d]
   patch_features = rearrange(patch_features, '(b n) d -> b n d', b=B)
   ```

3. **Spatial Decoder (Pre-trained)**
   ```python
   # Reshape to grid
   features = rearrange(patch_features, 'b (gy gx) d -> b d gy gx', 
                        gy=Gy, gx=Gx)
   
   # Option A: SegFormer decoder
   logits = segformer_decoder(features)  # [B, 1, Gy, Gx]
   
   # Option B: UNet + ResNet
   logits = unet_decoder(features)
   
   # Unpatchify to pixel space
   logits_pixels = F.interpolate(logits, scale_factor=patch_size,
                                  mode='nearest')  # [B, 1, H, W]
   ```

**Task 2.2:** Training script `scripts/train_emberformer.py`
- Mirror `train_stage_c_tokens.py` structure
- Use new temporal collate function
- Load pre-trained weights for spatial decoder
- Freeze/unfreeze strategy:
  - Start: freeze pre-trained decoder, train temporal transformer
  - Fine-tune: unfreeze decoder after N epochs

**Task 2.3:** Config `configs/emberformer.yaml`
```yaml
model:
  temporal:
    d_model: 64
    nhead: 4
    num_layers: 3
    dropout: 0.1
    max_seq_len: 8  # Keep low: >50% of dataset has T≤6
  spatial:
    decoder_type: "segformer"  # or "unet_resnet", "sam"
    pretrained: true
    freeze_epochs: 5
    
data:
  token_cache_dir: "data/patch_cache"
  patch_size: 16
  max_history: 6  # Match dataset distribution: >50% have ≤6 frames
  
training:
  batch_size: 8
  lr_temporal: 1e-3
  lr_spatial: 1e-4  # lower for pre-trained
  weight_decay: 0.01
  epochs: 50
```

### Phase 3: Experimentation (0.5-1.5 days)

**Task 3.1:** Sanity checks
- Start with short history (T=6, matches dataset mode)
- Verify loss decreases
- Check temporal attention weights (are early frames attended?)

**Task 3.2:** Baseline comparison
- EmberFormer vs. UNetS (last-frame-only)
- Metrics: Precision, Recall, F1, IoU
- Log W&B visualizations

**Task 3.3:** Ablation studies
```
1. w/ vs w/o pre-trained decoder
2. w/ vs w/o wind embedding
3. w/ vs w/o static terrain embedding
4. 1 vs 3 vs 6 transformer layers
5. History length: 3 vs 6 vs 8 frames (dataset constraint: >50% have ≤6)
```

**Task 3.4:** Handle variable-length sequences
- Test sequences with T=2, T=4, T=6, T=8+ in same batch
- Verify temporal padding mask works correctly
- Monitor performance stratified by sequence length

---

## Hyperparameter Tuning with W&B Sweeps

### Current W&B Integration Status: ✅ READY

The project has mature W&B integration via `utils/wandb_utils.py`:
- ✅ `init_wandb()`: Auto-generates run names, tags, handles offline mode
- ✅ `define_common_metrics()`: Sets up train/val metrics with step tracking
- ✅ `watch_model()`: Logs gradients (optional)
- ✅ `log_grid_preview()`: Visualizes predictions vs targets
- ✅ `save_artifact()`: Saves checkpoints to W&B

All training scripts (`train_stage_c_tokens.py`, `train_stage_c_raw.py`, `train_phase1.py`) use this infrastructure.

### Creating a Sweep for EmberFormer

**Step 1: Create sweep config** (`configs/sweep_emberformer.yaml`):

```yaml
program: scripts/train_emberformer.py
method: bayes  # or 'grid', 'random'
metric:
  name: val/f1
  goal: maximize
parameters:
  # Temporal transformer
  model.temporal.d_model:
    values: [32, 64, 128]
  model.temporal.num_layers:
    values: [2, 3, 4, 6]
  model.temporal.nhead:
    values: [2, 4, 8]
  model.temporal.dropout:
    min: 0.0
    max: 0.3
  
  # Spatial decoder
  model.spatial.decoder_type:
    values: ["segformer", "unet_resnet34"]
  model.spatial.freeze_epochs:
    values: [0, 3, 5]
  
  # Training
  train.lr_temporal:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  train.lr_spatial:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  train.weight_decay:
    values: [0, 1e-5, 1e-4, 1e-3]
  train.batch_size:
    values: [4, 8, 16]
  
  # Data (constrained by dataset)
  data.max_history:
    values: [4, 6, 8]  # >50% have T≤6
```

**Step 2: Initialize sweep**:
```bash
wandb sweep configs/sweep_emberformer.yaml
# Returns: wandb: Created sweep with ID: abc123xyz
```

**Step 3: Launch agents** (can run multiple in parallel):
```bash
# Terminal 1
wandb agent your-entity/emberformer/abc123xyz

# Terminal 2 (parallel)
wandb agent your-entity/emberformer/abc123xyz

# Terminal 3 (on different GPU)
CUDA_VISIBLE_DEVICES=1 wandb agent your-entity/emberformer/abc123xyz
```

### Sweep Integration Requirements

To make training scripts sweep-compatible, they must:
1. ✅ Accept config via YAML (already done)
2. ✅ Use `wandb.config` for hyperparameters (need to add)
3. ✅ Log metrics with consistent keys (already done: `train/loss`, `val/f1`, etc.)
4. ✅ Run for full epochs and log validation metrics (already done)

**Minimal code change needed:**
```python
# In train_emberformer.py, after wandb.init()
run = init_wandb(cfg, context={"script": "train_emberformer"})
if run:
    # Override config with sweep parameters
    cfg = dict(run.config)  # wandb injects sweep params here
```

### Recommended Sweep Strategy

**Phase 1: Architecture search** (Bayes, 30-50 runs)
- Vary: `d_model`, `num_layers`, `nhead`, `decoder_type`
- Fixed: `lr_temporal=1e-3`, `max_history=6`, `batch_size=8`
- Goal: Find best architecture given T=6 constraint

**Phase 2: Learning rate tuning** (Grid, 20 runs)
- Fix best architecture from Phase 1
- Vary: `lr_temporal`, `lr_spatial`, `freeze_epochs`
- Goal: Optimize training dynamics

**Phase 3: Regularization** (Random, 20 runs)
- Vary: `dropout`, `weight_decay`
- Goal: Improve generalization

**Phase 4: Data ablations** (Grid, 12 runs)
- Vary: `max_history` (4/6/8), with/without wind, with/without static
- Goal: Understand what temporal context helps

### Expected Sweep Duration

Assuming:
- 1 epoch ≈ 5 minutes (tokens, T=6, batch=8, single GPU)
- 20 epochs per run
- 50 runs total

**Total GPU time:** ~83 hours (3.5 days on 1 GPU, <1 day on 4 GPUs in parallel)

### Sweep Best Practices

1. **Start small:** Run 5-10 manual experiments first to verify code works
2. **Parallelize:** Use 2-4 agents to speed up Bayesian sweeps
3. **Early stopping:** Add `wandb.log({"val/f1": f1})` and use W&B early stopping
4. **Budget constraints:** Set `max_runs` in sweep config or use `wandb agent --count 20`
5. **Monitor progress:** Check W&B dashboard parallel coordinates plot

---

## Technical Considerations

### Memory Management
- Batch size: Start with 4-8 due to T×N tokens
- Complexity: O(T²) per patch (transformer self-attention)
- If OOM with large T: use causal/local windowed attention

### Masking Correctness
- ✅ Temporal mask: pass `src_key_padding_mask` to TransformerEncoder
- ✅ Spatial mask: use `valid_tokens` when computing loss
- ✅ Ensure left-padding aligns with mask (False=pad)

### Training Strategy
1. **Stage 1 (epochs 0-5):** Freeze pre-trained decoder, train temporal transformer
2. **Stage 2 (epochs 5+):** Unfreeze decoder with lower LR (1e-4 vs 1e-3)
3. **Use differential learning rates:** temporal (1e-3), spatial (1e-4)

### Loss Function
```python
# Weighted BCE (handles class imbalance)
pos_weight = (n_negative / n_positive)
loss_bce = F.binary_cross_entropy_with_logits(logits, target, 
                                               pos_weight=pos_weight)

# Optional: Add Dice loss for boundary quality
loss_dice = dice_loss(torch.sigmoid(logits), target)

loss = loss_bce + 0.1 * loss_dice
```

---

## Advanced Paths (If Needed)

### If Performance Plateaus:
1. **Add spatial attention:** After temporal transformer, add 1-2 layers of cross-patch attention
2. **Multi-scale features:** Use FPN or multi-resolution tokens
3. **Auxiliary losses:** Add edge detection loss, perimeter prediction

### If Memory is Limiting:
1. **Efficient attention:** Use xFormers, Flash Attention 2
2. **Causal attention:** With local window (attend only to last K frames)
3. **Gradient checkpointing:** Trade compute for memory

### If Long Sequences (T>64) Needed:
1. **Hierarchical attention:** Coarse-to-fine temporal modeling
2. **State Space Models:** Consider Mamba/S4 for O(T) complexity
3. **Temporal downsampling:** Skip frames or use temporal pooling

---

## Known Issues

1. **Token cache `valid_tokens.npy`:** Currently all-ones, doesn't reflect spatial padding footprint for non-strict mode
2. **Train script bug:** `train_stage_c_raw.py` prints `train/acc` instead of `train/loss` (cosmetic only)
3. **No transformer implementation:** Core architecture missing

---

## Success Metrics

### Minimum Viable Product (MVP):
- [ ] EmberFormer trains without errors
- [ ] Loss decreases steadily
- [ ] Beats UNetS baseline on validation set
- [ ] Temporal attention visualizations show meaningful patterns

### Research Success:
- [ ] F1 score > 0.7 on test set
- [ ] Captures long-range dependencies (T=16-32)
- [ ] Ablation shows temporal history helps (vs last-frame only)
- [ ] Model generalizes across different fire events

### Publication Ready:
- [ ] State-of-art on benchmark (if exists)
- [ ] Interpretable attention patterns (early vs late fire behavior)
- [ ] Real-time inference capability (<1s per prediction)
- [ ] Ablation studies validate architecture choices

---

## Effort Estimates

| Task | Time | Priority |
|------|------|----------|
| Dataset temporal extension | 0.5-1 day | HIGH |
| Fix spatial validity mask | 0.5 day | MEDIUM |
| Implement EmberFormer model | 1-2 days | HIGH |
| Training script + config | 0.5 day | HIGH |
| Initial experiments (T=8) | 0.5 day | HIGH |
| Ablation studies | 1 day | MEDIUM |
| Scale to long sequences | 0.5-1 day | LOW |

**Total MVP time: 3-5 days**

---

## Next Immediate Steps

1. **Research pre-trained decoders** (1 hour)
   - Test SegFormer vs UNet+ResNet integration
   - Verify compatibility with patch token features
   
2. **Extend TokenFireDataset** (0.5 day)
   - Return full temporal history
   - Create temporal collate function
   
3. **Implement EmberFormer** (1-2 days)
   - Temporal transformer module
   - Integration with pre-trained spatial decoder
   - Training script with differential LR

4. **Run baseline experiment** (0.5 day)
   - Train with T=8, d_model=64, 3 layers
   - Compare against UNetS last-frame baseline
   - Log to W&B

---

## References

- **ClimaX:** Pre-trained weather/climate model (MS Research) - [arXiv:2301.10343](https://arxiv.org/abs/2301.10343)
- **SegFormer:** Efficient transformer segmentation - [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
- **Segment Anything (SAM):** Universal segmentation model - [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- Recent fire prediction papers show transformers work but train from scratch on domain data

---

## Questions to Resolve

- [ ] Which pre-trained decoder: SegFormer vs SAM vs UNet+ResNet?
- [ ] Optimal freeze strategy: freeze encoder, decoder, or both initially?
- [ ] Should we add explicit physics constraints (wind direction bias)?
- [ ] Multi-step forecasting (t+2, t+3) or just t+1?
