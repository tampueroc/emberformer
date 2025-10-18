# ✅ Ready to Train DINO Architecture

## Current Status

Branch: `feat-dino-architecture`  
Status: **Ready for training**  
Latest commit: Fixed wandb initialization

---

## Quick Start

```bash
# On your training server (relela-07)
cd ~/repositories/emberformer
git fetch
git checkout feat-dino-architecture
git pull

# Run training
uv run python scripts/train_dino.py \
  --config configs/emberformer_dino.yaml \
  --phase 1 \
  --gpu 0
```

---

## What Just Got Fixed

The error you encountered:
```
TypeError: init_wandb() got an unexpected keyword argument 'script_name'
```

**Fixed by:**
- Using correct `init_wandb(cfg, context={...})` signature
- Properly creating torchmetrics instead of calling non-existent function
- Commit: `798f0ab`

---

## Expected Training Output

```
============================================================
EmberFormer-DINO Training - Phase 1
============================================================
Config: configs/emberformer_dino.yaml
Device: cuda:0
============================================================

✓ W&B initialized: dino-phase1-1018-1432

Loading datasets...
  Train: 800 samples
  Val: 100 samples

Static channels: 7

Creating EmberFormerDINO...
  Total params: 46,421,594
  Trainable params: 2,308,442
  Frozen params: 44,113,152
  Phase: 1 (Frozen DINO)
  Loss: focal_tversky

Starting training for 30 epochs...

Epoch 1/30
------------------------------------------------------------
Training: 100%|███████████████| 100/100 [02:15<00:00]
Validation: 100%|█████████████| 13/13 [00:15<00:00]
  Train Loss: 0.4523 | F1: 0.6234 | IoU: 0.4521
  Val Loss:   0.4892 | F1: 0.5987 | IoU: 0.4298
  ✓ Saved checkpoint: checkpoints/dino_phase1_best.pt

Epoch 2/30
...
```

---

## Monitor Training

**W&B Dashboard:**
https://wandb.ai/tampueroc-university-of-chile/emberformer

**Key metrics to watch:**
- `val/f1` - Main metric (should increase)
- `val/loss` - Should decrease
- `train/loss` vs `val/loss` - Watch for overfitting gap

**Expected performance (Phase 1):**
- Epoch 1: F1 ~0.55-0.60
- Epoch 10: F1 ~0.65-0.70
- Epoch 20: F1 ~0.70-0.75
- Convergence: ~20-30 epochs

---

## If You Hit Issues

### 1. CUDA Out of Memory

Reduce batch size in config:
```yaml
data:
  batch_size: 4  # Down from 8
```

### 2. Data not found

Check path in config:
```yaml
data:
  data_dir: "~/data/deep_crown_dataset/organized_spreads"
```

### 3. Training very slow

- Check GPU usage: `nvidia-smi`
- Increase workers: `num_workers: 12` (if you have cores)
- Check data loading isn't bottleneck

### 4. Model not learning

- Check W&B preview images
- Verify loss is decreasing (even slowly)
- Check learning rate: should be ~1e-3 for Phase 1

---

## After Training Completes

### 1. Check Results

```bash
# Best checkpoint saved at:
checkpoints/dino_phase1_best.pt
```

**In Python:**
```python
import torch
checkpoint = torch.load('checkpoints/dino_phase1_best.pt')
print(f"Best Val F1: {checkpoint['val_f1']:.4f}")
print(f"Best Epoch: {checkpoint['epoch']}")
```

### 2. Compare with Baseline

Look at W&B runs and compare:
- DINO Phase 1 vs SegFormer baseline
- Check F1, IoU, boundary quality
- Compare training curves

### 3. Optional: Phase 2 Fine-tuning

If Phase 1 results are good, fine-tune DINO:

```bash
uv run python scripts/train_dino.py \
  --config configs/emberformer_dino.yaml \
  --phase 2 \
  --gpu 0
```

**Note:** Phase 2 uses much lower LR (1e-5) and trains the DINO encoder.

---

## Architecture Summary

```
Input: Fire frames [B, T, 1, H, W] + Static [B, 7, H, W] + Wind [B, T, 2]
  ↓
DINO Spatial Encoders (frozen, 44M params)
  ↓
Feature Fusion (0.5M params)
  ↓
Temporal Transformer 4 layers (1.5M params)
  ↓
Simple Spatial Decoder (0.2M params)
  ↓
Refinement Decoder (0.1M params)
  ↓
Output: Binary fire prediction [B, 1, H, W]
```

**Total:** 46M params, only 2.3M trainable

---

## Files in This Branch

- ✅ `models/emberformer.py` - DINO components added
- ✅ `scripts/train_dino.py` - Training script
- ✅ `configs/emberformer_dino.yaml` - Configuration
- ✅ `tests/test_dino_forward.py` - Tests (passing)
- ✅ `DINO_ARCHITECTURE.md` - Technical documentation
- ✅ `DINO_TRAINING.md` - Training guide

---

## Expected Timeline

**Phase 1 (Frozen DINO):**
- Time: 2-4 hours on RTX 3090
- Epochs: 20-30 (with early stopping)
- Expected F1: 0.70-0.75

**Phase 2 (Fine-tune DINO) - Optional:**
- Time: 3-5 hours
- Epochs: 15-20
- Expected improvement: +2-5% F1

**Total:** ~5-9 hours for both phases

---

## Support

- **Architecture questions:** See `DINO_ARCHITECTURE.md`
- **Training issues:** See `DINO_TRAINING.md`
- **Code issues:** Check `tests/test_dino_forward.py` passes

---

**Ready to train! 🚀**

Just run the command at the top and monitor W&B.
