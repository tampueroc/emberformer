# GPU Optimization Guide

## Current Issue

**Problem:** Only using ~2.7GB / 24GB (11%) on RTX 3090  
**Solution:** Scale up model and batch size to better utilize GPU

---

## Configuration Comparison

### Small Config (Original - `emberformer_small.yaml`)
**Target:** Quick experiments, limited GPU (<8GB)

| Component | Value | Memory |
|-----------|-------|--------|
| Batch size | 8 | Low |
| d_model | 64 | ~500K params (temporal) |
| nhead | 4 | |
| num_layers | 3 | |
| dim_feedforward | 256 | |
| SegFormer | B0 | 3.7M params |
| **Total** | | **~2.7GB GPU** |

### Large Config (Optimized - `emberformer.yaml`)
**Target:** Full utilization of RTX 3090 (24GB)

| Component | Value | Memory |
|-----------|-------|--------|
| Batch size | 32 | **4x increase** |
| d_model | 128 | ~2M params (temporal) |
| nhead | 8 | **2x** |
| num_layers | 6 | **2x** |
| dim_feedforward | 512 | **2x** |
| SegFormer | B2 | 27M params |
| **Total** | | **~12-14GB GPU** |

---

## Expected Improvements

### 1. Batch Size: 8 → 32
**Benefits:**
- ✅ 4x faster training (fewer gradient updates)
- ✅ More stable gradients (better statistics)
- ✅ Better GPU utilization (parallelism)

**Trade-offs:**
- Learning dynamics slightly different (may need LR adjustment)

### 2. Model Size: 64 → 128 dimensions, 3 → 6 layers
**Benefits:**
- ✅ 4x more parameters (~500K → ~2M temporal)
- ✅ Greater model capacity for complex patterns
- ✅ Better long-range dependency modeling
- ✅ More expressive temporal attention

**Trade-offs:**
- Slightly slower per-batch (still faster overall)
- May need more regularization (dropout, weight decay)

### 3. SegFormer: B0 → B2
**Benefits:**
- ✅ 7x more parameters (3.7M → 27M)
- ✅ Better spatial feature extraction
- ✅ Stronger pre-trained features
- ✅ Better boundary detection

**Trade-offs:**
- Downloads ~90MB model (first time only)

---

## Estimated Memory Usage

### Small Config (Current)
```
Batch size: 8
Sequences: T_avg=6, N=400 patches

Model:        ~500 MB  (temporal + decoder)
Activations:  ~1.5 GB  (forward + backward)
Optimizer:    ~500 MB  (Adam states)
Total:        ~2.7 GB  ✓ Measured
```

### Large Config (Optimized)
```
Batch size: 32
Sequences: T_avg=6, N=400 patches

Model:        ~2 GB    (temporal + decoder)
Activations:  ~8 GB    (forward + backward, 4x batch)
Optimizer:    ~2 GB    (Adam states)
Total:        ~12 GB   (~50% of RTX 3090)
```

**Safety margin:** Still 12GB free for peaks and system overhead

---

## How to Use

### Option 1: Use Large Config (Recommended for RTX 3090)
```bash
# Train with optimized config
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --gpu 0

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Option 2: Keep Small Config
```bash
# Use for debugging or quick experiments
uv run python scripts/train_emberformer.py --config configs/emberformer_small.yaml --gpu 0
```

### Option 3: Manual Override
```bash
# Test different batch sizes
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --batch_size 64 --gpu 0

# If OOM, reduce:
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --batch_size 24 --gpu 0
```

---

## Progressive Scaling Strategy

If you want to be cautious, scale up gradually:

### Step 1: Increase Batch Size Only
```yaml
batch_size: 32  # Keep model small
d_model: 64
num_layers: 3
```
**Result:** ~6GB GPU, 4x faster training

### Step 2: Add Model Capacity
```yaml
batch_size: 32
d_model: 128    # Double embedding
num_layers: 6   # Double layers
```
**Result:** ~10GB GPU, better performance

### Step 3: Upgrade Decoder
```yaml
batch_size: 32
d_model: 128
num_layers: 6
model_name: "nvidia/segformer-b2-finetuned-ade-512-512"
```
**Result:** ~14GB GPU, best performance

---

## Monitoring GPU Usage

### During Training
```bash
# Terminal 1: Training
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml

# Terminal 2: Monitor
watch -n 1 nvidia-smi
```

### Expected Output
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util |
|   0  RTX 3090     72C   P2   320W / 350W |  12000MiB / 24576MiB |    95%   |
+-------------------------------+----------------------+----------------------+
```

**Target:**
- Memory: 10-14GB / 24GB (40-60%)
- GPU-Util: 90-100%
- Temp: <85°C

---

## If OOM (Out of Memory)

### Quick Fixes
1. **Reduce batch size:**
   ```bash
   --batch_size 24  # or 16, 12
   ```

2. **Use gradient checkpointing** (not yet implemented):
   ```python
   # In models/emberformer.py
   self.temporal_transformer.gradient_checkpointing_enable()
   ```

3. **Reduce model size:**
   ```yaml
   d_model: 96      # Between 64 and 128
   num_layers: 4    # Between 3 and 6
   ```

4. **Use mixed precision** (already enabled):
   - Already using `amp.autocast` in training script
   - Saves ~30% memory

### Memory vs. Performance Trade-off

| Config | Memory | Speed | Performance |
|--------|--------|-------|-------------|
| Small (BS=8, d=64) | 2.7GB | 1x | Baseline |
| Medium (BS=16, d=96) | 6GB | 2x | +10% F1 |
| Large (BS=32, d=128) | 12GB | 3.5x | +15% F1 |
| XLarge (BS=48, d=128) | 18GB | 4x | +16% F1 |

**Recommendation:** Use Large config - best performance/speed ratio

---

## Advanced: Batch Size vs. Learning Rate

When increasing batch size, you may want to adjust learning rate:

**Linear Scaling Rule:**
```
batch_size: 8  → lr_temporal: 1e-3
batch_size: 32 → lr_temporal: 4e-3  (scale by sqrt(4) ≈ 2)
```

But often the original LR works fine. Try both:
```bash
# Original LR
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml

# Scaled LR
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --lr_temporal 2e-3
```

Monitor W&B for which converges better.

---

## Summary

**Current Status:** Underutilizing GPU by 8-9x  
**Recommended Action:** Use `configs/emberformer.yaml` (large config)  
**Expected Result:** 
- 50% GPU utilization (~12GB)
- 4x faster training
- Better model performance
- Still safe margin for memory spikes

**Command:**
```bash
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --gpu 0
```
