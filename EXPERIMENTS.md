# Parallel Experiments: Focal+Tversky vs BCE+Dice

## Setup

**Hardware:** relela-05 (2× NVIDIA RTX A6000, 48GB each)

**Goal:** Compare loss functions for precision optimization

## Experiments

### Experiment 1: Focal+Tversky (GPU 0) ✨ RECOMMENDED
**Config:** `configs/emberformer.yaml`

**Loss:**
- Focal Loss (70%): Handles 95/5 class imbalance, focuses on hard examples
  - α=0.25, γ=2.0 (standard RetinaNet params)
- Tversky Loss (30%): Direct precision control
  - α=0.7 (heavy FP penalty), β=0.3 (light FN penalty)

**Expected Results:**
- Precision: **70-80%** (up from 26%)
- Recall: **85-90%** (healthy tradeoff from 98%)
- F1: **~0.75-0.80**

### Experiment 2: BCE+Dice (GPU 1) 📊 BASELINE
**Config:** `configs/emberformer_bce_dice.yaml`

**Loss:**
- BCE (90%): Standard binary cross-entropy
- Dice (10%): IoU awareness with sqrt scaling

**Expected Results:**
- Precision: **50-60%** (moderate improvement from 26%)
- Recall: **90-95%** (still high)
- F1: **~0.65-0.70**

## Running Experiments

### Quick Start (Automated)
```bash
./run_experiments.sh
```

### Manual Start
```bash
# Terminal 1: Focal+Tversky on GPU 0
uv run python scripts/train_emberformer.py \
  --config configs/emberformer.yaml \
  --gpu 0

# Terminal 2: BCE+Dice on GPU 1
uv run python scripts/train_emberformer.py \
  --config configs/emberformer_bce_dice.yaml \
  --gpu 1
```

## Monitoring

### Check logs
```bash
tail -f logs/focal_tversky_gpu0.log
tail -f logs/bce_dice_gpu1.log
```

### GPU utilization
```bash
watch -n 1 nvidia-smi
```

### W&B Dashboard
Both runs will appear in your W&B project with tags:
- Experiment 1: `focal_tversky`, `gpu0`
- Experiment 2: `bce_dice`, `gpu1`

## Expected Training Time

- **Dataset:** 65k training samples, 8k validation
- **Batch size:** 4 per GPU
- **Early stopping:** ~10-15 epochs (patience=3)
- **Time per epoch:** ~20-25 minutes
- **Total time:** **3-6 hours** (depending on convergence)

## Comparison Metrics

Key metrics to compare:

| Metric | Focal+Tversky (Expected) | BCE+Dice (Expected) |
|--------|--------------------------|---------------------|
| Precision | 70-80% | 50-60% |
| Recall | 85-90% | 90-95% |
| F1 Score | 0.75-0.80 | 0.65-0.70 |
| IoU | 0.60-0.67 | 0.48-0.54 |

**Winner criteria:** Highest F1 score with precision >65%

## Architecture

Both experiments use:
- EmberFormer with SegFormer spatial decoder
- RefinementDecoder for pixel-precise predictions (learned upsampling)
- Temporal transformer: 3 layers, 4 heads, d_model=64
- Patch size: 8×8
- Total params: ~1.07M

## Post-Training Analysis

After both experiments complete, run:
```bash
# Compare metrics
python scripts/compare_experiments.py \
  --run1 <focal_tversky_run_id> \
  --run2 <bce_dice_run_id>

# Load best model for interpretability
python scripts/inspect_checkpoint.py \
  --checkpoint checkpoints/emberformer_best.pt
```

See `INTERPRETABILITY.md` for detailed analysis plan.
