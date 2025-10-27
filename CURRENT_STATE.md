# EmberFormer Current Project State

**Date:** Oct 25, 2025
**Branch:** main
**Latest Commit:** `f4f87bc` - Use run_name_template with sequence length
**Status:** 🔬 Active experimentation with DINO-based architecture

---

## Executive Summary

EmberFormer has evolved from a transformer-only architecture to a **DINO-ViT + Temporal Transformer** hybrid model for wildfire spread prediction. The project is currently in **active experimentation phase** with two recent completed experiments:

1. ✅ **emberformer-dino-expanding**: Variable-length training (T=2-6) with expanding window strategy
2. ✅ **Fixed T=3**: Baseline with shorter temporal context

**Current Focus:** Exploring optimal temporal context length and transfer learning strategies with pre-trained DINO vision encoders.

---

## Recent Experiments (Last 2)

### Experiment 1: EmberFormer-DINO Expanding Window ✅
**Config:** `configs/emberformer_dino_expanding.yaml`
**Date:** Oct 20, 2025

**Key Features:**
- **Variable-length training**: Uses all subsequences T=2,3,4,5,6 from each sequence
- **DINO encoder**: Frozen `facebook/dinov2-base` for spatial feature extraction
- **Max sequence:** T=6 (covers >50% of dataset)
- **Two-phase training:**
  - Phase 1: 30 epochs, frozen DINO (lr=1e-4)
  - Phase 2: 20 epochs, fine-tune DINO (lr=1e-5)
- **Loss:** Focal (70%) + Tversky (30%) for precision control

**Architecture:**
```
Fire frames [B,T,H,W] → DINO-ViT → patch tokens [B,T,N,d]
                                   ↓
Static terrain [B,H,W] → DINO-ViT → static tokens [B,N,d]
                                   ↓
Wind [B,T,2] → MLP → wind embedding [B,T,d]
                                   ↓
                    Temporal Transformer (4 layers, 8 heads, d=256)
                                   ↓
                    Spatial Decoder (Simple Conv)
                                   ↓
                    Refinement Decoder (14x14 patches → pixels)
                                   ↓
                    Output: [B, 1, H, W] fire prediction
```

**Expected Results:**
- Precision: 70-80%
- Recall: 85-90%
- F1: 0.75-0.80

**Implementation Details:**
- Batch size: 8 (reduced from 16 for T=6 memory)
- Temporal d_model: 256 (high capacity)
- Decoder: Simple conv (128→48 channels)
- Refinement: Learned upsampling from patches to pixels
- Early stopping: patience=5, monitor val/f1

### Experiment 2: Fixed T=3 Baseline ✅
**Config:** `configs/emberformer_dino_t5.yaml` (modified for T=3)
**Date:** Oct 20, 2025

**Key Features:**
- **Fixed temporal context**: T=3 (shorter history)
- **Purpose:** Baseline to compare against expanding window
- **Same architecture** as expanding but with consistent sequence length
- **Higher batch size:** 14 (more memory available with T=3)

**Rationale:**
- Test if variable-length training helps vs fixed-length
- Faster training iterations for debugging
- Lower memory requirements

---

## Current Architecture: EmberFormer-DINO

### Model Components

#### 1. DINO Vision Encoders
**Pre-trained:** `facebook/dinov2-base` (ViT-B/14)
**Purpose:** Extract spatial features from fire frames and static terrain

**Features:**
- Fire encoder: Processes temporal fire frames independently
- Static encoder: Processes terrain (elevation, slope, fuel, etc.)
- Patch size: 14×14 pixels
- Output dimension: 768 → projected to d_model=256
- Freeze strategy: Start frozen, fine-tune in Phase 2

#### 2. Temporal Transformer
**Purpose:** Model fire spread dynamics over time

**Architecture:**
- Layers: 4
- Attention heads: 8
- d_model: 256
- Feedforward: 1024
- Dropout: 0.1
- Max sequence: 32 (flexible for future experiments)

**Input fusion:**
- Fire tokens [B,T,N,768] from DINO
- Static tokens [B,N,768] from DINO (broadcast over time)
- Wind embedding [B,T,32] from MLP
- Positional encoding for temporal order

#### 3. Spatial Decoder
**Type:** Simple convolutional decoder
**Architecture:**
- Input: [B, d, Gy, Gx] patch grid features
- Hidden: 128 channels
- Base: 48 channels
- Output: [B, 1, Gy, Gx] patch-level logits

#### 4. Refinement Decoder
**Purpose:** Upscale from patches to pixels with learned interpolation

**Architecture:**
- Patch size: 14×14
- Base channels: 32
- Conv layers with bilinear upsampling
- Output: [B, 1, H, W] pixel-level predictions

### Model Statistics
- **Total parameters:** ~90M (DINO: ~86M, Custom: ~4M)
- **Trainable (Phase 1):** ~4M (DINO frozen)
- **Trainable (Phase 2):** ~90M (DINO unfrozen)
- **Memory (T=6, batch=8):** ~8-10GB GPU
- **Input size:** 406×406 pixels

---

## Data Pipeline Status

### ✅ Completed
- **RawFireDataset:** Loads pixel-level data with temporal sequences
- **Expanding window support:** Creates all valid subsequences from each sequence
- **Variable-length batching:** Left-padding with validity masks
- **Multi-channel static:** 7 terrain features (elevation, slope, aspect, fuel types)
- **Wind normalization:** Global min/max scaling
- **Data splits:** 80/10/10 train/val/test

### Dataset Statistics
- **Total sequences:** ~73K training samples
- **Sequence lengths:** Variable (T=2 to T=20+)
- **>50% sequences:** Have T≤6 timesteps
- **Image size:** 406×406 pixels
- **Class imbalance:** ~95/5 (no-fire/fire)

### Data Augmentation
Currently **not implemented** but planned:
- Random rotations
- Flips
- Color jitter for fire intensity

---

## Training Infrastructure

### W&B Integration ✅
**Project:** `emberformer`
**Entity:** `tampueroc-university-of-chile`

**Logged Metrics:**
- Train: loss, accuracy, precision, recall, f1, iou
- Val: loss, accuracy, precision, recall, f1, iou
- Learning rate schedule
- Gradient norms
- Sample predictions (images)

**Run Naming:**
- Template: `{script}-dino-{variant}-t{T}-{timestamp}`
- Tags: Auto-generated (dino, expanding-window, focal-tversky, etc.)

### Checkpointing ✅
**Directory:** `~/data/emberformer/checkpoints/`
**Naming:** `{config_name}_{run_id}_best.pt`

**Saved State:**
- Model weights
- Optimizer state
- Epoch number
- Best validation metrics
- Config snapshot

### Loss Functions

#### Focal + Tversky (Current) ✅
**Purpose:** Precision-optimized for imbalanced data

**Focal Loss (70%):**
- α=0.25: Down-weight easy negatives
- γ=2.0: Focus on hard examples
- Handles 95/5 class imbalance

**Tversky Loss (30%):**
- α=0.3: False negative penalty
- β=0.7: False positive penalty (heavy)
- Direct precision control

**Expected Impact:**
- Precision: 26% → 70-80%
- Recall: 98% → 85-90%
- F1: Better balance

#### BCE + Dice (Previous baseline)
**Purpose:** Standard segmentation loss

**Components:**
- BCE (90%): Binary cross-entropy with pos_weight
- Dice (10%): IoU-aware boundary quality

---

## Training Scripts

### 1. `train_dino.py` (Primary) ✅
**Purpose:** Train EmberFormer with DINO encoders

**Features:**
- Two-phase training (frozen → fine-tune)
- Expanding window data loading
- Differential learning rates
- Early stopping
- Mixed precision training
- Gradient clipping

**Usage:**
```bash
uv run python scripts/train_dino.py \
  --config configs/emberformer_dino_expanding.yaml \
  --gpu 0
```

### 2. `train_emberformer.py` (Legacy)
**Purpose:** Original transformer-only version

**Status:** Superseded by DINO variant

### 3. Supporting Scripts
- `build_patch_cache.py`: Pre-compute patch tokens (not used with DINO)
- `test_expanding_window.py`: Verify variable-length data loading
- `inspect_checkpoint.py`: Load and analyze saved models

---

## Configuration Files

### Active Configs

**1. `emberformer_dino_expanding.yaml`**
- Expanding window training (T=2-6)
- Two-phase training
- Focal+Tversky loss
- Batch size: 8

**2. `emberformer_dino_t5.yaml`**
- Fixed T=5 context
- Continue from Phase 2 checkpoint
- Lower LR for fine-tuning
- Batch size: 14

**3. `emberformer_dino_t6.yaml`**
- Fixed T=6 (maximum common length)
- Batch size: 10

**4. `emberformer_dino.yaml`**
- Original DINO config (T=4)
- Baseline reference

### Legacy Configs
- `emberformer.yaml`: Original transformer with SegFormer
- `emberformer_bce_dice.yaml`: BCE+Dice loss variant
- `stage_c_raw.yaml`, `stage_c.yaml`: Phase 1 baselines

---

## Interpretability & Analysis

### Available Tools

**1. Attention Entropy Analysis** (`analysis/attention_entropy.py`)
- Analyzes DINO encoder attention patterns
- Measures focus vs diffusion over temporal sequences
- Visualizes which patches attend to fire spread

**2. DINO Self-Attention Visualization**
- Hooks into DINO encoder layers
- Extracts attention maps
- Correlates with fire progression

**3. Checkpoint Inspection** (`scripts/inspect_checkpoint.py`)
- Load saved models
- Examine learned weights
- Replay predictions

### Planned Analysis (See `INTERPRETABILITY_DINO.md`)
- Temporal attention flow: Which past frames matter most?
- Spatial attention patterns: Do models focus on fire boundaries?
- Feature evolution: How do patch embeddings change over time?
- Physics alignment: Does attention follow wind direction?

---

## Known Issues & Limitations

### 1. Memory Constraints ⚠️
- **T=6, batch=8:** Uses ~8-10GB GPU
- **Solution:** Reduced batch size from 16 to 8
- **Future:** Gradient checkpointing for longer sequences

### 2. DINO Fine-tuning ⚠️
- **Issue:** Fine-tuning 86M parameters is slow
- **Current:** Two-phase training (frozen → unfreeze)
- **Future:** Layer-wise LR decay, LoRA adapters

### 3. Class Imbalance 🔄
- **Problem:** 95/5 no-fire/fire ratio
- **Current:** Focal+Tversky loss
- **Status:** Awaiting experiment results

### 4. No Checkpoint Manager 📋
- **Issue:** Manual checkpoint file management
- **Impact:** Hard to track best models across experiments
- **TODO:** Implement automatic checkpoint cleanup

### 5. Dataset Size Uncertainty ❓
- **Issue:** Exact number of training samples unknown
- **Impact:** Epoch timing estimates inaccurate
- **TODO:** Add dataset size counter

---

## Next Immediate Steps

### 1. Evaluate Recent Experiments (PRIORITY)
```bash
# Check W&B dashboard for results
# Compare expanding window vs fixed T=3
# Analyze precision/recall tradeoff
```

**Expected Decision:** Choose best temporal strategy for next experiments

### 2. Longer Context Experiments (if expanding wins)
**Options:**
- T=8: Test if more history helps
- T=10: Upper bound for common sequences
- Hierarchical: Coarse-to-fine temporal modeling

### 3. Augmentation Experiments
**Add to data pipeline:**
- Random rotations (90°, 180°, 270°)
- Horizontal/vertical flips
- Fire intensity jitter

**Expected Impact:** +5-10% F1 from better generalization

### 4. Advanced Loss Functions (if precision still low)
**Try:**
- Combo loss: Focal + Tversky + Dice
- Boundary loss: Edge-aware penalties
- Asymmetric loss: Different weights per fire stage

### 5. Interpretability Analysis
**After best model chosen:**
- Run attention entropy analysis
- Visualize temporal attention weights
- Generate failure case analysis
- Create physics-alignment metrics

---

## Success Metrics

### Current Baseline (UNetS last-frame)
- Precision: 26%
- Recall: 98%
- F1: ~0.41
- IoU: ~0.26

### Target Metrics (EmberFormer-DINO)

**Minimum Viable (MVP):**
- ✅ Trains without errors
- ✅ Loss decreases steadily
- [ ] Precision > 60%
- [ ] F1 > 0.65
- [ ] Beats last-frame baseline

**Research Success:**
- [ ] Precision > 70%
- [ ] F1 > 0.75
- [ ] IoU > 0.60
- [ ] Temporal ablation shows history helps
- [ ] Generalizes across fire events

**Publication Ready:**
- [ ] F1 > 0.80
- [ ] Interpretable attention patterns
- [ ] Real-time inference (<1s per prediction)
- [ ] Ablation validates all architecture choices
- [ ] State-of-art on benchmark (if exists)

---

## Experiment Tracking

### Completed Experiments

| Experiment | Config | T | Batch | Status | F1 | Notes |
|------------|--------|---|-------|--------|----|----|
| DINO Expanding | `emberformer_dino_expanding.yaml` | 2-6 | 8 | ✅ Done | TBD | Variable-length training |
| Fixed T=3 | `emberformer_dino_t5.yaml` (mod) | 3 | 14 | ✅ Done | TBD | Baseline comparison |

### Planned Experiments

| Experiment | Config | T | Purpose |
|------------|--------|---|---------|
| DINO T=6 | `emberformer_dino_t6.yaml` | 6 | Maximum common length |
| DINO T=8 | `emberformer_dino_t8.yaml` (new) | 8 | Test longer context |
| With Augmentation | `emberformer_dino_aug.yaml` | 2-6 | Data augmentation impact |
| LoRA Fine-tune | `emberformer_dino_lora.yaml` (new) | 2-6 | Efficient DINO adaptation |

---

## Technical Debt & TODOs

### High Priority
- [ ] **Evaluate recent experiments:** Check W&B results, compare metrics
- [ ] **Add dataset size logging:** Know exact training samples
- [ ] **Implement checkpoint cleanup:** Auto-delete old checkpoints
- [ ] **Add learning rate scheduler:** Cosine annealing for better convergence

### Medium Priority
- [ ] **Data augmentation pipeline:** Rotations, flips, jitter
- [ ] **Gradient checkpointing:** Enable longer sequences (T>8)
- [ ] **LoRA adapters:** Efficient DINO fine-tuning
- [ ] **Multi-GPU training:** Distributed data parallel

### Low Priority
- [ ] **Attention visualization UI:** Interactive attention map explorer
- [ ] **Model compression:** Quantization, pruning for deployment
- [ ] **Multi-step forecasting:** Predict t+2, t+3, etc.
- [ ] **Uncertainty estimation:** Bayesian dropout, ensembles

---

## Repository Structure

```
emberformer/
├── configs/                    # YAML configuration files
│   ├── emberformer_dino_expanding.yaml  # Active: Variable-length
│   ├── emberformer_dino_t5.yaml         # Active: Fixed T=5
│   └── *.yaml                           # Other variants
├── data/                       # Data loading & preprocessing
│   ├── datasets.py            # RawFireDataset, expanding window
│   └── collate.py             # Batch collation with padding
├── models/                     # Model architectures
│   ├── emberformer_dino.py    # DINO + Temporal Transformer
│   ├── emberformer.py         # Legacy transformer-only
│   └── baselines.py           # UNetS, ConvNet baselines
├── scripts/                    # Training & analysis scripts
│   ├── train_dino.py          # Main training script
│   ├── test_expanding_window.py  # Data loading tests
│   └── inspect_checkpoint.py  # Model analysis
├── analysis/                   # Interpretability tools
│   └── attention_entropy.py   # DINO attention analysis
├── utils/                      # Shared utilities
│   ├── wandb_utils.py         # W&B logging helpers
│   └── metrics.py             # Precision, recall, F1, IoU
├── tests/                      # Unit tests
├── results/                    # Experiment outputs
└── docs/                       # Documentation
    ├── CURRENT_STATE.md       # This file
    ├── PROJECT_STATE.md       # Original architecture design
    ├── PHASE2_COMPLETE.md     # Transformer implementation
    ├── INTERPRETABILITY_DINO.md  # Analysis plan
    └── *.md                   # Other guides
```

---

## Key References

### Documentation
- **PROJECT_STATE.md:** Original transformer architecture design and roadmap
- **PHASE2_COMPLETE.md:** EmberFormer (transformer-only) implementation
- **INTERPRETABILITY_DINO.md:** DINO attention analysis guide
- **DINO_ARCHITECTURE.md:** DINO integration details
- **DINO_TRAINING.md:** Two-phase training strategy

### External
- **DINO v2:** [Meta AI DinoV2](https://github.com/facebookresearch/dinov2)
- **SegFormer:** [NVidia SegFormer](https://github.com/NVlabs/SegFormer)
- **Focal Loss:** [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- **Tversky Loss:** [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)

---

## Questions to Resolve

### Immediate
- [ ] Did expanding window training beat fixed T=3? (Check W&B)
- [ ] What is optimal temporal context length? (T=3 vs T=5 vs T=6)
- [ ] Is Focal+Tversky better than BCE+Dice? (Precision comparison)

### Short-term
- [ ] Should we add data augmentation? (Expected +5-10% F1)
- [ ] Is DINO fine-tuning worth the compute? (Phase 1 vs Phase 2 comparison)
- [ ] Can we scale to longer sequences (T>8) with current memory?

### Long-term
- [ ] Multi-step forecasting: Predict multiple future timesteps?
- [ ] Physics constraints: Should we encode wind direction bias?
- [ ] Real-time deployment: Can we compress model for inference?
- [ ] Generalization: Does model transfer to different fire datasets?

---

## Team & Resources

**Hardware:** relela-05 server
- 2× NVIDIA RTX A6000 (48GB each)
- Enough memory for batch=8, T=6 with DINO

**W&B Project:**
- Project: `emberformer`
- Entity: `tampueroc-university-of-chile`
- URL: [W&B Dashboard](https://wandb.ai/tampueroc-university-of-chile/emberformer)

**Contact:**
- Project lead: tampueroc
- Institution: University of Chile
- Focus: Wildfire spread prediction with transformers

---

## Summary

EmberFormer is transitioning from a custom-built transformer to a **DINO-ViT + Temporal Transformer hybrid** that leverages pre-trained vision encoders for better spatial understanding. Recent experiments focus on finding the optimal temporal context length and training strategy (variable-length vs fixed).

**Current State:**
- ✅ DINO integration complete
- ✅ Expanding window training implemented
- ✅ Two experiments completed (awaiting results)
- 🔄 Evaluating precision/recall tradeoff with Focal+Tversky loss
- 🎯 Next: Choose best strategy, scale to longer sequences

**Key Innovation:** Variable-length training with expanding windows allows the model to see all possible subsequences, potentially learning better temporal dynamics than fixed-length training.

---

**Last Updated:** Oct 25, 2025
**Next Review:** After evaluating experiments #1 and #2 results
