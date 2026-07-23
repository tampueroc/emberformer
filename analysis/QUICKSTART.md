# Analysis Quick Start Guide

## Run Phase 1 Analysis Now

Your Phase 1 model is ready for analysis! Here's how to run it:

### 1. Simple Run (Recommended)

```bash
cd ~/Code/emberformer
python analysis/phase_1/run_analysis.py
```

**Time:** ~1.5-2 hours total  
**Output:** `analysis/phase_1/{spatial,temporal,features,wind,extreme_events}/`

### 2. Fast Run (Skip Slow Analyses)

```bash
# Skip feature ablation (slowest part ~1 hour)
python analysis/phase_1/run_analysis.py --skip features

# Only spatial and temporal (~25 min total)
python analysis/phase_1/run_analysis.py --skip features wind extreme
```

### 3. Debug Run (Few Samples)

```bash
python analysis/phase_1/run_analysis.py --num-samples 20 --skip features extreme
```

## What You'll Get

### Spatial Importance Maps
![Example](spatial/spatial_importance_sample_000.png)
- Shows which input pixels drive predictions
- 10 sample visualizations + statistics

### Temporal Patterns
![Example](temporal/temporal_importance.png)
- Which timesteps matter most (t-1 vs t-2 vs t-3)
- Reveals recency bias

### Feature Importance Ranking
![Example](features/feature_importance.png)
- Which terrain features matter: slope > fuel > wind > elevation
- Validates fire physics learning

### Wind Alignment
![Example](wind/wind_alignment.png)
- Does model follow wind direction?
- Measures directional accuracy

### Extreme Events
![Example](extreme_events/extreme_event_00_*.png)
- Top 10 most challenging scenarios
- Error analysis (FP vs FN)

## Key Questions Answered

1. **Spatial:** Does frozen DINO capture fire boundaries?
2. **Temporal:** Does model use full history or just t-1?
3. **Features:** Which terrain features drive predictions?
4. **Wind:** Did model learn wind physics?
5. **Extreme:** Why does it fail on large spreads?

## While Phase 2 Trains

**Timeline:**
- Phase 2 training: ~2-3 hours (20 epochs)
- Phase 1 analysis: ~1.5-2 hours

**Run them in parallel:**

Terminal 1:
```bash
ssh relela-05
cd ~/Code/emberformer
uv run scripts/train_dino.py --phase 2 --config configs/emberformer_dino.yaml --gpu 0
```

Terminal 2:
```bash
ssh relela-05
cd ~/Code/emberformer
python analysis/phase_1/run_analysis.py
```

## After Phase 2 Completes

Re-run same analysis on Phase 2:

```bash
python analysis/phase_1/run_analysis.py \
    --checkpoint checkpoints/dino_phase2_best.pt \
    --output-dir analysis/phase_2
```

Then compare:
```bash
# Side-by-side comparison
ls -lh analysis/phase_1/spatial/*.png
ls -lh analysis/phase_2/spatial/*.png

# Metrics comparison
cat analysis/phase_1/features/feature_importance.json
cat analysis/phase_2/features/feature_importance.json
```

## Troubleshooting

### "No module named 'models'"
```bash
# Make sure you're in project root
cd ~/Code/emberformer
python analysis/phase_1/run_analysis.py
```

### "Checkpoint not found"
```bash
# Verify checkpoint exists
ls -lh checkpoints/dino_phase1_best.pt

# If on remote server, sync first
rsync -avz relela-05:~/Code/emberformer/checkpoints/ checkpoints/
```

### "CUDA out of memory"
```bash
# Use CPU instead (slower)
python analysis/phase_1/run_analysis.py --device cpu

# Or reduce samples
python analysis/phase_1/run_analysis.py --num-samples 50 --skip features
```

## Need Help?

See detailed documentation:
- [Phase 1 README](phase_1/README.md)
- [INTERPRETABILITY_DINO.md](../INTERPRETABILITY_DINO.md)
