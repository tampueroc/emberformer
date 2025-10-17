# W&B Metrics Reference

## What's Logged to Weights & Biases

### Step-Level Metrics (Continuous Updates)

**Loss (Every Step):**
- `train/loss_step` - Training loss for each batch
- Updates: ~every batch (hundreds per epoch)

**Training Metrics (Every 50 Steps):**
- `train/acc_step` - Accuracy (running average)
- `train/prec_step` - **Precision** (running average) ← YOU ASKED FOR THIS
- `train/rec_step` - Recall (running average)
- `train/f1_step` - F1 score (running average)
- `train/iou_step` - IoU/Jaccard (running average)
- Updates: Every 50 batches (configurable via `log_metrics_every`)

**Other:**
- `train/batch_T_avg` - Average sequence length per batch
- `epoch` - Current epoch number

---

### Epoch-Level Metrics (End of Each Epoch)

**Training:**
- `train/loss_epoch` - Average training loss for entire epoch
- `train/acc_epoch` - Training accuracy
- `train/prec_epoch` - **Training precision** ← YOU ASKED FOR THIS
- `train/rec_epoch` - Training recall
- `train/f1_epoch` - Training F1 score
- `train/iou_epoch` - Training IoU
- `train/T_avg_epoch` - Average sequence length for epoch

**Validation:**
- `val/loss_epoch` - Validation loss ← YOU ASKED FOR THIS
- `val/acc_epoch` - Validation accuracy
- `val/prec_epoch` - **Validation precision** ← YOU ASKED FOR THIS
- `val/rec_epoch` - Validation recall
- `val/f1_epoch` - Validation F1 score
- `val/iou_epoch` - Validation IoU

**Learning Rates:**
- `train/lr_temporal` - Learning rate for temporal transformer
- `train/lr_spatial` - Learning rate for spatial decoder

---

### Images (Every Epoch)

**Validation Previews:**
- `val/preview` - Side-by-side comparison:
  - Model predictions (probabilities)
  - Ground truth targets
  - Last fire frame input (context)
- Count: 2 samples by default (configurable via `max_preview_images`)

---

## How to View in W&B Dashboard

### 1. Training Progress (Main View)

**Recommended Charts:**

**Row 1: Loss**
- Left: `train/loss_step` (line chart)
- Right: `val/loss_epoch` (line chart)

**Row 2: Precision & F1**
- Left: `train/prec_step` (line chart)
- Right: `val/prec_epoch` (line chart)

**Row 3: IoU**
- Left: `train/iou_step` (line chart)
- Right: `val/iou_epoch` (line chart)

### 2. Finding Metrics in W&B

If you don't see a metric:

1. **Click "Add panel" → "Line plot"**
2. **Search for metric name:** Type `train/prec` or `val/loss`
3. **Select the metric** from dropdown
4. **Set X-axis:** Usually "step" for step-level, "epoch" for epoch-level

### 3. Smoothing

For step-level metrics (updated frequently), apply smoothing:
- Click on chart → "Smoothing"
- Set to 0.6-0.9 for cleaner curves

### 4. Comparing Runs

To compare multiple experiments:
- Select multiple runs in left sidebar
- Charts will overlay all runs
- Use different colors to distinguish

---

## Metric Naming Convention

### Suffix Meanings

- `_step` - Logged every N steps during training (running average)
- `_epoch` - Logged once per epoch (computed over entire epoch)
- No suffix - Special cases (e.g., `epoch` counter)

### Prefixes

- `train/` - Training set metrics
- `val/` - Validation set metrics

---

## Configuration

### Change Logging Frequency

In `configs/emberformer.yaml`:

```yaml
train:
  log_images_every: 1    # Log images every N epochs (1 = every epoch)
  log_metrics_every: 50  # Log metrics every N steps (50 = every 50 batches)
```

**Examples:**

**More frequent (more data, slower):**
```yaml
log_metrics_every: 10   # Every 10 steps
```

**Less frequent (less data, faster):**
```yaml
log_metrics_every: 100  # Every 100 steps
```

**Image logging:**
```yaml
log_images_every: 5     # Every 5 epochs (saves bandwidth)
```

---

## Troubleshooting

### "I don't see loss in charts"

**Check:**
1. Look for `train/loss_step` (not `train/loss`)
2. Look for `val/loss_epoch` (not `val/loss`)
3. Make sure X-axis is set to "step"

### "I don't see precision"

**Check:**
1. Step-level: `train/prec_step` (updated every 50 steps)
2. Epoch-level: `train/prec_epoch` (updated end of epoch)
3. Validation: `val/prec_epoch` (end of epoch)

### "Charts are too noisy"

**Solution:**
- Apply smoothing (0.6-0.9)
- Or use epoch-level metrics instead of step-level

### "Too much data / slow dashboard"

**Solution:**
Reduce logging frequency in config:
```yaml
log_metrics_every: 200  # Log less often
```

---

## Quick Start

### 1. Start Training

```bash
uv run python scripts/train_emberformer.py --config configs/emberformer.yaml --gpu 0
```

**Look for this in output:**
```
W&B run: train_emberformer-lr1e-3-1017-1234 (abc123xyz)
W&B url: https://wandb.ai/your-entity/emberformer/runs/abc123xyz
```

### 2. Open W&B Dashboard

Click the URL or go to: https://wandb.ai/your-entity/emberformer

### 3. Add Charts

**Loss:**
- Add panel → Line plot → `train/loss_step`
- Add panel → Line plot → `val/loss_epoch`

**Precision:**
- Add panel → Line plot → `train/prec_step`
- Add panel → Line plot → `val/prec_epoch`

**Images:**
- Already visible in "Media" tab or panels

---

## Summary of Changes

**Before:** Only epoch-level metrics, no step-level updates

**After:**
- ✅ Step-level loss (every step)
- ✅ Step-level precision, F1, IoU (every 50 steps)
- ✅ Epoch-level everything (train + val)
- ✅ Clear naming (_step vs _epoch)
- ✅ LR tracking (temporal vs spatial)
- ✅ Images every epoch
- ✅ Configurable frequency

**What you asked for:**
- ✅ "log precision at training" → `train/prec_step` and `train/prec_epoch`
- ✅ "also loss" → `train/loss_step` and `val/loss_epoch`

Both are now logged and visible in W&B! 🎉
