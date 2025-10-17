# EmberFormer Training Guide

## Quick Start

```bash
# Start training with optimized config
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --gpu 0
```

---

## Current Configuration (Optimized for RTX 3090)

### Model Architecture
- **d_model:** 96 (embedding dimension)
- **nhead:** 8 (attention heads)
- **num_layers:** 4 (transformer layers)
- **Spatial decoder:** SegFormer-B1 (pre-trained)
- **Total params:** ~15M

### Training Setup
- **Batch size:** 16 (balanced for 24GB GPU)
- **Epochs:** 50 (with early stopping)
- **LR temporal:** 1e-3 (transformer)
- **LR spatial:** 1e-4 (decoder - lower for pre-trained)
- **Weight decay:** 1e-4

### Loss Function ✅ **NEW: BCE + Dice**
- **BCE weight:** 0.5 (pixel-wise stability)
- **Dice weight:** 0.5 (optimizes F1/IoU directly)
- **Why:** Dice optimizes region overlap, better for boundaries and sparse fire pixels

### Early Stopping ✅ **NEW**
- **Enabled:** Yes
- **Patience:** 10 epochs
- **Monitor:** val/f1
- **Mode:** maximize
- **Min delta:** 0.001 (0.1% improvement)

### Expected GPU Usage
- **Memory:** ~20GB / 24GB (83%)
- **Utilization:** 95-100%
- **Temperature:** 85-90°C
- **Safe margin:** ~4GB headroom

---

## What You'll See

### At Start
```
============================================================
Starting Training: 50 epochs
  Model: EmberFormer (segformer decoder)
  Train samples: 1234
  Val samples: 154
  Batch size: 16
  LR temporal: 1.00e-03, LR spatial: 1.00e-04
  Loss: BCE (0.5) + Dice (0.5)
  Early stopping: enabled (patience=10, monitor=val/f1)
  W&B run: train_emberformer-... (abc123)
  W&B url: https://wandb.ai/.../runs/abc123
============================================================
```

### During Training
```
Epoch 1/50 [Train]: 100%|████████| 77/77 [01:23<00:00, loss=0.4234]
Epoch 1/50 [Val]:   100%|████████| 10/10 [00:08<00:00, loss=0.3891]
[Epoch  1/50] BCE+Dice | train: loss=0.4234 f1=0.623 prec=0.712 | val: loss=0.3891 f1=0.656 prec=0.734 iou=0.487

Epoch 2/50 [Train]: 100%|████████| 77/77 [01:22<00:00, loss=0.3845]
Epoch 2/50 [Val]:   100%|████████| 10/10 [00:08<00:00, loss=0.3567]
[Epoch  2/50] BCE+Dice | train: loss=0.3845 f1=0.678 prec=0.745 | val: loss=0.3567 f1=0.689 prec=0.756 iou=0.524
```

### If Early Stopping Triggers
```
[Epoch 23/50] BCE+Dice | train: loss=0.2134 f1=0.812 prec=0.834 | val: loss=0.2456 f1=0.789 prec=0.801 iou=0.652

🛑 Early stopping triggered!
   Best val/f1: 0.7892 at epoch 13
   No improvement for 10 epochs

============================================================
Training complete!
============================================================
```

---

## W&B Metrics (Comprehensive)

### Loss Components (Every Step)
- `train/loss_step` - Combined BCE + Dice loss
- `train/bce_step` - BCE component
- `train/dice_step` - Dice component

### Performance Metrics (Every 50 Steps)
- `train/prec_step` - **Precision** (running avg)
- `train/rec_step` - Recall
- `train/f1_step` - F1 score
- `train/iou_step` - IoU/Jaccard
- `train/acc_step` - Accuracy

### Epoch-Level Metrics
- `train/loss_epoch`, `val/loss_epoch` - Average loss
- `train/prec_epoch`, `val/prec_epoch` - **Precision**
- `train/f1_epoch`, `val/f1_epoch` - F1 score
- `train/iou_epoch`, `val/iou_epoch` - IoU
- `train/rec_epoch`, `val/rec_epoch` - Recall
- `train/acc_epoch`, `val/acc_epoch` - Accuracy

### Training Dynamics
- `train/lr_temporal` - Temporal transformer learning rate
- `train/lr_spatial` - Spatial decoder learning rate
- `train/T_avg_epoch` - Average sequence length
- `train/batch_T_avg` - Sequence length per batch

### Early Stopping Tracking
- `early_stop/counter` - Epochs without improvement
- `early_stop/best_value` - Best metric value achieved
- `early_stop/best_epoch` - Epoch with best metric

### Images (Every Epoch)
- `val/preview` - Predictions vs targets (2 samples)

---

## Configuration Options

### Change Batch Size
```bash
# Try larger batch (if you have memory)
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --batch_size 20

# Or smaller if OOM
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --batch_size 12
```

### Adjust Loss Weights
Edit `configs/emberformer.yaml`:
```yaml
model:
  loss:
    use_dice: true
    bce_weight: 0.7      # More weight on BCE (if precision drops)
    dice_weight: 0.3     # Less weight on Dice
```

### Early Stopping Settings

**More aggressive (stop sooner):**
```yaml
early_stopping:
  patience: 5          # Stop after 5 epochs
  min_delta: 0.005     # Require 0.5% improvement
```

**More patient (train longer):**
```yaml
early_stopping:
  patience: 15         # Wait 15 epochs
  min_delta: 0.0001    # Accept tiny improvements
```

**Monitor different metric:**
```yaml
early_stopping:
  monitor: "val/iou"   # Stop based on IoU instead of F1
  mode: "max"
```

**Disable early stopping:**
```yaml
early_stopping:
  enabled: false       # Train for full 50 epochs
```

---

## Recommended W&B Dashboard Layout

### Panel 1: Loss Components
- `train/loss_step` (line, smoothing=0.7)
- `train/bce_step` (line, smoothing=0.7)
- `train/dice_step` (line, smoothing=0.7)

### Panel 2: Validation Metrics
- `val/loss_epoch` (line)
- `val/f1_epoch` (line)
- `val/iou_epoch` (line)

### Panel 3: Precision & Recall
- `val/prec_epoch` (line)
- `val/rec_epoch` (line)

### Panel 4: Early Stopping
- `early_stop/counter` (line)
- `early_stop/best_value` (line)

### Panel 5: Learning Rates
- `train/lr_temporal` (line)
- `train/lr_spatial` (line)

### Panel 6: Images
- `val/preview` (media panel)

---

## Troubleshooting

### Early Stopping Too Aggressive

**Symptom:** Stops at epoch 15, but loss still decreasing slowly

**Solution:**
```yaml
early_stopping:
  patience: 15         # Increase patience
  min_delta: 0.0001    # Lower improvement threshold
```

### Loss Not Decreasing

**Check:**
1. **Loss components:** Are BCE and Dice both decreasing?
2. **Learning rates:** Are they appropriate? (temporal > spatial)
3. **Images:** Do predictions look reasonable?
4. **Metrics:** Is F1/IoU improving even if loss plateaus?

**Try:**
```yaml
# Increase BCE weight if unstable
bce_weight: 0.7
dice_weight: 0.3
```

### Precision Too Low

**Symptom:** Many false positives

**Solution:**
```yaml
# Increase BCE weight (penalizes false positives more)
bce_weight: 0.7
dice_weight: 0.3

# Or adjust prediction threshold
metrics:
  pred_thresh: 0.6  # Higher threshold = fewer predictions
```

### IoU Not Improving

**Symptom:** F1 good but IoU stuck

**Solution:**
```yaml
# Increase Dice weight (directly optimizes IoU)
bce_weight: 0.3
dice_weight: 0.7

# Or monitor IoU for early stopping
early_stopping:
  monitor: "val/iou"
```

---

## Training Time Estimates

### With Current Config (Batch=16, RTX 3090)

**Per Epoch:**
- Train: ~1.5 minutes (77 batches)
- Val: ~10 seconds (10 batches)
- **Total:** ~2 minutes/epoch

**Full Training:**
- Without early stopping: 50 epochs × 2min = **~100 minutes (1.7 hours)**
- With early stopping: ~25 epochs × 2min = **~50 minutes**

### Batch Size Impact

| Batch Size | Memory | Time/Epoch | Total (25 epochs) |
|------------|--------|------------|-------------------|
| 8 | 10GB | 3 min | 75 min |
| 12 | 15GB | 2.3 min | 58 min |
| **16** | **20GB** | **2 min** | **50 min** ✓ |
| 20 | 23GB | 1.8 min | 45 min ⚠️ risky |
| 24 | >24GB | - | OOM ❌ |

---

## Advanced: Custom Early Stopping

### Monitor Multiple Metrics

Edit training script to add custom logic:
```python
# Stop if F1 > 0.80 AND IoU > 0.70
if val_metrics['val/f1'] > 0.80 and val_metrics['val/iou'] > 0.70:
    print("🎯 Target metrics achieved!")
    break
```

### Save Best Model Only

The current script saves the final model. To save the best:
```python
# After early_stopper check
if early_stopper and metric_value == early_stopper.best_value:
    # This is the best epoch so far
    best_model_path = f"emberformer_best_ep{ep}.pt"
    torch.save(model.state_dict(), best_model_path)
```

---

## Summary of Improvements

✅ **Loss Function:** BCE + Dice (optimizes for your metrics)  
✅ **Early Stopping:** Prevents overfitting, saves time  
✅ **W&B Logging:** All metrics tracked (loss, precision, F1, IoU)  
✅ **GPU Utilization:** 80% (balanced, safe margin)  
✅ **Progress Bars:** Real-time training feedback  
✅ **Configurable:** All settings in YAML

Your training is now production-ready! 🚀
