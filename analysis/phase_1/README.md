# Phase 1 Interpretability Analysis

Comprehensive analysis suite for understanding EmberFormer-DINO Phase 1 (frozen DINO) model behavior.

## Prerequisites

- ✅ Phase 1 training complete (F1 = 68.9%, epoch 24)
- ✅ Checkpoint exists: `checkpoints/dino_phase1_best.pt`
- ✅ Validation dataset available

## Quick Start

### Run All Analyses

```bash
cd ~/Code/emberformer
python analysis/phase_1/run_analysis.py
```

### Run Specific Analyses

```bash
# Skip slow feature ablation
python analysis/phase_1/run_analysis.py --skip features

# Run only spatial and temporal
python analysis/phase_1/run_analysis.py --skip features wind extreme

# Use fewer samples for speed
python analysis/phase_1/run_analysis.py --num-samples 50
```

### Custom Configuration

```bash
python analysis/phase_1/run_analysis.py \
    --checkpoint checkpoints/dino_phase1_best.pt \
    --config configs/emberformer_dino.yaml \
    --output-dir analysis/phase_1 \
    --num-samples 100 \
    --device cuda
```

## Analyses Included

### 1. Spatial Importance (~ 5-10 min)
**What:** Gradient-based attribution showing which input pixels drive predictions

**Output:**
- `spatial/spatial_importance_sample_*.png` - Individual sample visualizations
- `spatial/spatial_importance_statistics.png` - Aggregate statistics

**Key Questions:**
- Does model focus on fire boundaries or interior?
- Is attention distributed or concentrated?

### 2. Temporal Importance (~10-15 min)
**What:** Measures which timesteps in history are most important

**Output:**
- `temporal/temporal_importance.png` - Importance by sequence length
- `temporal/temporal_importance_data.pkl` - Raw data

**Key Questions:**
- Does recent frame (t-1) dominate?
- Is full history (3-4 frames) utilized?

### 3. Feature Ablation (~30-60 min)
**What:** Tests importance of each terrain feature by removal

**Output:**
- `features/feature_importance.png` - Importance ranking
- `features/feature_importance.json` - Detailed results

**Key Questions:**
- Which terrain features matter most?
- Does model learn fire physics (slope, fuel)?

### 4. Wind Alignment (~15-20 min)
**What:** Tests if predictions align with wind direction

**Output:**
- `wind/wind_alignment.png` - Alignment analysis
- `wind/wind_alignment_data.csv` - Per-sample data

**Key Questions:**
- Does model learn wind-driven spread?
- How well aligned are predictions?

### 5. Extreme Events (~20-30 min)
**What:** Identifies and analyzes challenging extreme spread events

**Output:**
- `extreme_events/extreme_event_*.png` - Top 10 visualizations
- `extreme_events/spread_distribution.png` - Distribution analysis
- `extreme_events/extreme_events_summary.json` - Summary stats

**Key Questions:**
- How does model handle extreme spreads?
- Does it overpredict or underpredict?

## Expected Results

Based on Phase 1 performance (F1=68.9%, Precision=63.5%, Recall=75.2%):

### Spatial Importance
- **Expected:** Distributed attention across fire region
- **Interpretation:** Frozen DINO captures general fire patterns but not fine-grained boundaries

### Temporal Importance
- **Expected:** 60-70% weight on t-1 (recency bias)
- **Interpretation:** Model relies heavily on most recent frame, partial history use

### Feature Importance (Expected Ranking)
1. **Slope** (~0.02-0.04 F1 drop) - Directional spread
2. **Fuel Load** (~0.01-0.03) - Spread magnitude
3. **Wind Speed** (~0.01-0.02) - Rate modulation
4. **Elevation** (~0.005-0.01) - Indirect effects

### Wind Alignment
- **Expected:** 40-60% well-aligned (< 45°)
- **Interpretation:** Model learned some wind physics but imperfect

### Extreme Events
- **Expected:** Underprediction bias (conservative)
- **Interpretation:** High recall (75%) but lower precision (63%) - catches fire but overpredicts extent

## Comparison with Phase 2

After Phase 2 training completes, re-run this analysis:

```bash
python analysis/phase_1/run_analysis.py \
    --checkpoint checkpoints/dino_phase2_best.pt \
    --output-dir analysis/phase_2
```

**Expected improvements in Phase 2:**
- Sharper spatial attention on boundaries
- Better feature importance (higher terrain sensitivity)
- Improved wind alignment (>60% well-aligned)
- Better extreme event handling

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in feature_ablation.py (line 69)
# Or reduce num_samples
python analysis/phase_1/run_analysis.py --num-samples 50 --skip features
```

### Slow Execution
```bash
# Run only fast analyses
python analysis/phase_1/run_analysis.py --skip features extreme
```

### Missing Checkpoint
```bash
# Verify checkpoint exists
ls -lh checkpoints/dino_phase1_best.pt

# If missing, retrain Phase 1
uv run scripts/train_dino.py --phase 1 --config configs/emberformer_dino.yaml --gpu 0
```

## Output Structure

```
analysis/phase_1/
├── README.md
├── run_analysis.py          # Main entry point
├── spatial_importance.py    # Module 1
├── temporal_importance.py   # Module 2
├── feature_ablation.py      # Module 3
├── wind_analysis.py          # Module 4
├── extreme_events.py         # Module 5
├── analysis_metadata.yaml   # Run metadata
├── spatial/                 # Results folder 1
├── temporal/                # Results folder 2
├── features/                # Results folder 3
├── wind/                    # Results folder 4
└── extreme_events/          # Results folder 5
```

## Citation

If using these analyses in publications, please reference:

```
EmberFormer-DINO Interpretability Analysis
Phase 1 (Frozen DINO Encoder)
Model: DinoV2-base + Temporal Transformer + Spatial Decoder
Performance: F1=0.689, Precision=0.635, Recall=0.752
Checkpoint: dino_phase1_best.pt (Epoch 24)
```
