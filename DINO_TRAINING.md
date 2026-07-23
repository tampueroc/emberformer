# DINO Training Guide

Quick guide to train EmberFormer with DINO spatial encoding.

## Prerequisites

1. **Install dependencies** (if not already done):
```bash
uv sync
```

2. **Verify DINO installation**:
```bash
uv run python tests/test_dino_forward.py
```

Should see: `✓ All tests passed! ✓ Ready for training!`

---

## Training Phases

### Phase 1: Frozen DINO (Recommended First)

Train with DINO encoders frozen, only train fusion + transformer + decoder.

**Command:**
```bash
uv run python scripts/train_dino.py --config configs/emberformer_dino.yaml --phase 1
```

**What happens:**
- DINO fire encoder: **frozen** (44M params)
- DINO static encoder: **frozen** 
- Trainable: Fusion (0.5M) + Transformer (1.5M) + Decoder (0.3M) = **2.3M params**
- Learning rate: `1e-3` (from config)
- Duration: ~30 epochs (or early stopping)

**Expected results:**
- Faster convergence (~20 epochs)
- F1 score: 0.6-0.75 (depending on data)
- Training time: ~2-4 hours on RTX 3090

---

### Phase 2: Fine-tune DINO (Optional)

After Phase 1 converges, fine-tune DINO fire encoder for domain adaptation.

**Command:**
```bash
uv run python scripts/train_dino.py --config configs/emberformer_dino.yaml --phase 2
```

**What happens:**
- DINO fire encoder: **trainable** (22M params)
- DINO static encoder: **frozen** (stays frozen)
- Trainable: Fire encoder + Fusion + Transformer + Decoder = **24M params**
- Learning rate: `1e-5` (much lower!)
- Duration: ~20 epochs

**Expected results:**
- Marginal improvement: +2-5% F1
- Adapts DINO to fire-specific visual patterns
- Training time: ~3-5 hours on RTX 3090

---

## Configuration

Edit `configs/emberformer_dino.yaml` to customize:

### Common settings:
```yaml
data:
  batch_size: 8        # Reduce to 4 if OOM
  num_workers: 8       # Adjust based on CPU cores

model:
  dino:
    model_name: "facebook/dinov2-small"  # Or dinov2-base for more capacity
  
  temporal:
    d_model: 256         # Embedding dimension
    num_layers: 4        # Transformer depth
    nhead: 8             # Attention heads

train:
  epochs: 30            # Phase 1 epochs
  lr: 1e-3              # Phase 1 learning rate
  
  finetune:
    enabled: false      # Set true for Phase 2
    epochs: 20          # Phase 2 epochs
    lr: 1e-5            # Phase 2 learning rate (much lower!)
```

---

## GPU Options

### Specify GPU:
```bash
uv run python scripts/train_dino.py --config configs/emberformer_dino.yaml --phase 1 --gpu 0
```

### Multi-GPU (data parallel):
```bash
# Not yet implemented - coming soon
```

---

## Memory Optimization

If you get OOM errors:

1. **Reduce batch size** in config:
```yaml
data:
  batch_size: 4  # Down from 8
```

2. **Use smaller DINO model**:
```yaml
model:
  dino:
    model_name: "facebook/dinov2-small"  # Instead of base
```

3. **Reduce transformer size**:
```yaml
model:
  temporal:
    d_model: 128           # Down from 256
    num_layers: 3          # Down from 4
    dim_feedforward: 512   # Down from 1024
```

### Memory estimates (batch_size=8):

| Config | DINO Model | GPU Memory | Training Speed |
|--------|------------|------------|----------------|
| Small | dinov2-small (384-dim) | ~12 GB | Fast |
| Medium | dinov2-small + larger transformer | ~16 GB | Medium |
| Large | dinov2-base (768-dim) | ~20 GB | Slower |

---

## Monitoring with W&B

Training automatically logs to Weights & Biases:

1. **View metrics**: https://wandb.ai/tampueroc-university-of-chile/emberformer

2. **Key metrics to watch**:
   - `val/f1` - Main metric (higher is better)
   - `val/iou` - Intersection over Union
   - `val/loss` - Should decrease steadily
   - `train/loss` vs `val/loss` - Watch for overfitting

3. **Disable W&B** (for testing):
```yaml
wandb:
  enabled: false
```

---

## Checkpoints

Best model saved automatically:

**Phase 1:**
```
checkpoints/dino_phase1_best.pt
```

**Phase 2:**
```
checkpoints/dino_phase2_best.pt
```

### Load checkpoint:
```python
checkpoint = torch.load('checkpoints/dino_phase1_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best Val F1: {checkpoint['val_f1']:.4f}")
```

---

## Troubleshooting

### 1. "transformers package required"
```bash
uv add transformers
```

### 2. "CUDA out of memory"
- Reduce `batch_size` to 4 or 2
- Use `dinov2-small` instead of `dinov2-base`
- Reduce `d_model` or `num_layers`

### 3. "Data directory not found"
Update path in config:
```yaml
data:
  data_dir: "~/data/deep_crown_dataset/organized_spreads"
```

### 4. Training is slow
- Increase `num_workers` (CPU cores)
- Ensure `pin_memory: true`
- Check GPU utilization: `nvidia-smi`

### 5. Model not learning (loss not decreasing)
- Check learning rate (should be ~1e-3 for Phase 1)
- Verify data loading: check W&B preview images
- Try different loss weights in config

---

## Comparison with SegFormer baseline

To compare with old SegFormer approach:

```bash
# DINO (new)
uv run python scripts/train_dino.py --config configs/emberformer_dino.yaml --phase 1

# SegFormer (old, for comparison)
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml
```

**Expected improvements with DINO:**
- Better boundary detection
- Faster convergence
- Higher F1/IoU scores
- More stable training

---

## Next Steps After Training

1. **Evaluate on test set** (coming soon)
2. **Visualize predictions** - check W&B for preview images
3. **Try Phase 2 fine-tuning** if Phase 1 results are good
4. **Experiment with hyperparameters** using W&B sweeps

---

## Quick Start (TL;DR)

```bash
# Test installation
uv run python tests/test_dino_forward.py

# Train Phase 1 (frozen DINO)
uv run python scripts/train_dino.py --config configs/emberformer_dino.yaml --phase 1

# Monitor training
# Visit: https://wandb.ai/tampueroc-university-of-chile/emberformer

# After Phase 1 converges, optionally fine-tune
uv run python scripts/train_dino.py --config configs/emberformer_dino.yaml --phase 2
```

---

## Support

- **Architecture details**: See `DINO_ARCHITECTURE.md`
- **Project overview**: See `PROJECT_STATE.md`
- **General training**: See `TRAINING_GUIDE.md`
